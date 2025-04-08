import os
import sys
import numpy as np
import cv2
import tensorflow as tf

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

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

def appearance_dtsprompt(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, 1024))
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((11,11), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 31)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        norm_img = cv2.convertScaleAbs(norm_img, alpha=1.5, beta=0)
        result_norm_planes.append(norm_img)
    
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm, (w, h))
    return result_norm

def preprocess_patch(patch, size=256):
    """Preprocess a single patch for inference."""
    # Resize to model's expected input size if necessary
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size))
    
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
    
    # Normalize
    img_array = patch.astype(np.float32) / 255.0
    prompt_array = enhance_prompt.astype(np.float32) / 255.0
    
    # Stack along the channel dimension
    combined = np.concatenate([img_array, prompt_array], axis=-1)
    
    # Reshape to NCHW format for the model
    combined = np.transpose(combined, (2, 0, 1))
    combined = np.expand_dims(combined, axis=0)
    
    return combined, patch.shape[:2]

def postprocess_patch(output_array):
    """Postprocess a single output patch."""
    # Convert from NCHW to HWC format
    output = np.transpose(output_array, (1, 2, 0))
    
    # Scale back to 0-255 range
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    # Apply post-processing
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(output, -1, kernel)
    
    return output

def enhance_document(model, input_path, output_path, patch_size=256, overlap=32):
    """Enhance document using patch-based processing."""
    # Read original image
    original_img = cv2.imread(input_path)
    if original_img is None:
        raise ValueError(f"Could not read image at {input_path}")
    
    original_shape = original_img.shape
    
    # Create patches
    patches, positions, padded_size, (pad_h, pad_w) = create_patches(
        original_img, patch_size=patch_size, overlap=overlap
    )
    
    # Get the inference function
    infer = model.signatures["serving_default"]
    
    # Process each patch
    enhanced_patches = []
    patch_sizes = []
    total_patches = len(patches)
    
    print(f"Processing {total_patches} patches...")
    for i, patch in enumerate(patches):
        # Preprocess patch
        input_tensor, orig_patch_size = preprocess_patch(patch, size=256)
        patch_sizes.append(orig_patch_size)
        
        # Run inference
        output = infer(input=tf.constant(input_tensor, dtype=tf.float32))
        
        # Get the output tensor name from the model output
        output_name = list(output.keys())[0]
        output_tensor = output[output_name].numpy()
        
        # Postprocess patch
        enhanced_patch = postprocess_patch(output_tensor[0])
        
        # Resize back to original patch size if needed
        if enhanced_patch.shape[:2] != patch.shape[:2]:
            enhanced_patch = cv2.resize(enhanced_patch, (patch.shape[1], patch.shape[0]))
        
        enhanced_patches.append(enhanced_patch)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_patches} patches")
    
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TensorFlow Document Enhancement Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--model_dir', type=str, default='tensorflow_model', help='TensorFlow model directory')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of image patches (will be resized to 256x256 internally)')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between patches')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading TensorFlow model from {args.model_dir}...")
    model = tf.saved_model.load(args.model_dir)
    
    print(f"Model loaded. Processing input image: {args.input}")
    print(f"Note: Patches will be created with size {args.patch_size}x{args.patch_size} and then resized to 256x256 for inference")
    
    # Enhance document
    enhance_document(model, args.input, args.output, 
                    patch_size=args.patch_size, overlap=args.overlap)

if __name__ == "__main__":
    main()
