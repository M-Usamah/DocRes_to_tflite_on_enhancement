import os
import cv2
import torch
import numpy as np
from models import restormer_arch

def create_patches(img, patch_size=256, overlap=32):
    """Create overlapping patches from an image."""
    h, w = img.shape[:2]
    patches = []
    positions = []
    
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # Pad image if necessary
    if pad_h != 0 or pad_w != 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    stride = patch_size - overlap
    h, w = img.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    return patches, positions, (h, w), (pad_h, pad_w)

def merge_patches(patches, positions, img_size, patch_size=256, overlap=32):
    """Merge overlapping patches back into a single image."""
    h, w = img_size
    merged = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    stride = patch_size - overlap
    
    # Create weight matrix for blending
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    if overlap > 0:
        # Create smoother transition in overlap regions
        for i in range(overlap):
            weight[i, :] *= (i + 1) / (overlap + 1)
            weight[-i-1, :] *= (i + 1) / (overlap + 1)
            weight[:, i] *= (i + 1) / (overlap + 1)
            weight[:, -i-1] *= (i + 1) / (overlap + 1)
    
    for patch, (y, x) in zip(patches, positions):
        merged[y:y + patch_size, x:x + patch_size] += patch * weight[:, :, np.newaxis]
        weights[y:y + patch_size, x:x + patch_size] += weight
    
    # Add small epsilon to avoid division by zero
    weights = weights[:, :, np.newaxis] + 1e-8
    merged = merged / weights
    
    # Clip values to valid range
    merged = np.clip(merged, 0, 255)
    return merged

def load_model(checkpoint_path, device):
    # Initialize model with same parameters as training
    model = restormer_arch.Restormer(
        inp_channels=6,
        out_channels=3,
        dim=24,
        num_blocks=[1,1,2,2],
        num_refinement_blocks=2,
        heads=[1,2,2,4],
        ffn_expansion_factor=1.5,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True
    )
    
    # Load checkpoint with appropriate map_location
    if device.type == 'cpu':
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    # Convert to float32 for CPU or keep as is for GPU
    if device.type == 'cpu':
        model = model.float()
    
    return model

def appearance_dtsprompt(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    
    for plane in rgb_planes:
        # Increase kernel size for better background estimation
        dilated_img = cv2.dilate(plane, np.ones((11,11), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 31)  # Increased blur size
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        
        # Enhance contrast in the difference image
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Apply additional contrast enhancement
        norm_img = cv2.convertScaleAbs(norm_img, alpha=1.5, beta=0)
        result_norm_planes.append(norm_img)
    
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm, (w, h))
    return result_norm

def preprocess_patch(patch, size=256):
    # Apply initial denoising
    patch = cv2.fastNlMeansDenoisingColored(patch, None, 10, 10, 7, 21)
    
    # Enhance contrast globally
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    patch = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Generate enhancement prompt
    enhance_prompt = appearance_dtsprompt(patch)
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
    prompt_tensor = torch.from_numpy(enhance_prompt.transpose(2, 0, 1)).float() / 255.0
    
    return torch.cat((img_tensor, prompt_tensor), 0).unsqueeze(0)

def postprocess_patch(tensor_output):
    # Convert tensor to numpy array
    output = tensor_output.squeeze().cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    
    # Apply post-processing enhancements
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab = cv2.merge([l,a,b])
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(output, -1, kernel)
    
    return output

def enhance_document(model, input_path, output_path, device, patch_size=256, overlap=32):
    # Read original image
    original_img = cv2.imread(input_path)
    if original_img is None:
        raise ValueError(f"Could not read image at {input_path}")
    
    original_shape = original_img.shape
    
    # Create patches
    patches, positions, padded_size, (pad_h, pad_w) = create_patches(
        original_img, patch_size=patch_size, overlap=overlap
    )
    
    # Process each patch
    enhanced_patches = []
    total_patches = len(patches)
    
    # Adjust batch processing based on device
    batch_size = 1 if device.type == 'cpu' else 4
    
    print(f"Processing {total_patches} patches on {device.type.upper()}...")
    
    # Process patches in batches
    for i in range(0, total_patches, batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_tensors = []
        
        # Prepare batch
        for patch in batch_patches:
            input_tensor = preprocess_patch(patch)
            batch_tensors.append(input_tensor)
        
        # Concatenate batch tensors
        if len(batch_tensors) > 1:
            input_batch = torch.cat(batch_tensors, dim=0)
        else:
            input_batch = batch_tensors[0]
        
        input_batch = input_batch.to(device)
        
        # Inference
        with torch.no_grad():
            output_batch = model(input_batch, 'appearance')
        
        # Process each output in the batch
        for j in range(len(batch_patches)):
            output_tensor = output_batch[j:j+1]
            enhanced_patch = postprocess_patch(output_tensor)
            enhanced_patches.append(enhanced_patch)
        
        if (i + len(batch_patches)) % 10 == 0:
            print(f"Processed {i + len(batch_patches)}/{total_patches} patches")
            
        # Clear GPU memory if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("Merging patches...")
    # Merge patches
    merged_result = merge_patches(enhanced_patches, positions, padded_size, 
                                patch_size=patch_size, overlap=overlap)
    
    # Remove padding if any
    if pad_h > 0 or pad_w > 0:
        merged_result = merged_result[:-pad_h if pad_h > 0 else None, 
                                    :-pad_w if pad_w > 0 else None]
    
    # Ensure output matches original shape
    if merged_result.shape[:2] != original_shape[:2]:
        merged_result = cv2.resize(merged_result, (original_shape[1], original_shape[0]))
    
    # Final cleanup of any remaining invalid values
    merged_result = np.nan_to_num(merged_result, nan=0.0, posinf=255.0, neginf=0.0)
    merged_result = np.clip(merged_result, 0, 255)
    
    # Save result
    cv2.imwrite(output_path, merged_result.astype(np.uint8))
    print(f"Enhanced image saved to {output_path}")
    return merged_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Document Enhancement Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of image patches')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between patches')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU inference even if CUDA is available')
    args = parser.parse_args()
    
    # Device selection
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)
    enhance_document(model, args.input, args.output, device, 
                    patch_size=args.patch_size, overlap=args.overlap)