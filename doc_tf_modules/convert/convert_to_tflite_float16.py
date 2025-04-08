import os
import sys
import numpy as np
import tensorflow as tf

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

def convert_to_tflite_float16(saved_model_dir, tflite_model_path):
    """
    Convert TensorFlow SavedModel to TFLite format with float16 quantization
    
    Args:
        saved_model_dir: Directory containing the SavedModel
        tflite_model_path: Path to save the TFLite model
    """
    # Load the SavedModel
    print(f"Loading SavedModel from {saved_model_dir}...")
    model = tf.saved_model.load(saved_model_dir)
    
    # Get the concrete function
    concrete_func = model.signatures["serving_default"]
    
    # Define converter options
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable TF Select to support operations not natively supported in TFLite
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # This enables operations like Conv2D, Erf, RealDiv
    ]
    
    # Enable float16 quantization for smaller model size and better performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Set the inference input and output type
    converter.inference_input_type = tf.float32  # Keep input as float32
    converter.inference_output_type = tf.float32  # Keep output as float32
    
    # Enable experimental features for better optimization
    converter.experimental_new_converter = True
    
    # Convert the model
    print("Converting model to TFLite format with float16 quantization...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calculate model size
    model_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
    print(f"Float16 quantized TFLite model saved to {tflite_model_path}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Create inference script
    create_tflite_inference_script()
    
    return tflite_model_path

def create_tflite_inference_script():
    """
    Creates a Python script for running inference with the float16 quantized TFLite model
    """
    script_path = 'tflite_inference_float16.py'
    
    with open(script_path, 'w') as f:
        f.write('''
import os
import numpy as np
import cv2
import tensorflow as tf
import argparse
from time import time
import math

def create_patches(image, patch_size=256, overlap=64):
    """
    Create overlapping patches from an input image.
    
    Args:
        image: Input image
        patch_size: Size of each patch
        overlap: Overlap between patches
    
    Returns:
        patches: List of patches
        positions: List of (x, y) positions for each patch
    """
    h, w = image.shape[:2]
    patches = []
    positions = []
    
    stride = patch_size - overlap
    
    # Calculate number of patches in each dimension
    x_steps = max(1, math.ceil((w - patch_size) / stride) + 1)
    y_steps = max(1, math.ceil((h - patch_size) / stride) + 1)
    
    for y in range(y_steps):
        for x in range(x_steps):
            # Calculate patch coordinates
            x_start = min(x * stride, w - patch_size)
            y_start = min(y * stride, h - patch_size)
            
            # Extract patch
            patch = image[y_start:y_start + patch_size, x_start:x_start + patch_size]
            
            # If patch is smaller than patch_size (at edges), pad it
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            patches.append(patch)
            positions.append((x_start, y_start))
    
    return patches, positions

def merge_patches(patches, positions, original_size, patch_size=256, overlap=64):
    """
    Merge overlapping patches back into a single image with smooth blending.
    
    Args:
        patches: List of processed patches
        positions: List of (x, y) positions for each patch
        original_size: Original image size (h, w)
        patch_size: Size of each patch
        overlap: Overlap between patches
    
    Returns:
        merged: Merged image
    """
    h, w = original_size
    merged = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    # Create weight map for blending
    weight_map = np.ones((patch_size, patch_size), dtype=np.float32)
    
    # Apply tapering to the edges for smooth blending
    if overlap > 0:
        for i in range(overlap):
            weight_factor = (i + 1) / (overlap + 1)
            weight_map[i, :] *= weight_factor
            weight_map[-i-1, :] *= weight_factor
            weight_map[:, i] *= weight_factor
            weight_map[:, -i-1] *= weight_factor
    
    # Merge patches using weighted blending
    for patch, (x, y) in zip(patches, positions):
        # Get the actual patch dimensions to handle edge cases
        patch_h = min(patch_size, h - y)
        patch_w = min(patch_size, w - x)
        
        # Extract valid region
        valid_patch = patch[:patch_h, :patch_w]
        valid_weight = weight_map[:patch_h, :patch_w]
        
        # Apply weighted blending
        merged[y:y+patch_h, x:x+patch_w] += valid_patch * valid_weight[:, :, np.newaxis]
        weights[y:y+patch_h, x:x+patch_w] += valid_weight
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-10)  # Avoid division by zero
    merged = merged / weights[:, :, np.newaxis]
    
    return np.clip(merged, 0, 255).astype(np.uint8)

def preprocess_patch(patch):
    """
    Preprocess patch for the model.
    Expected input shape: [1, 6, 256, 256]
    """
    # Resize to 256x256 if needed
    if patch.shape[0] != 256 or patch.shape[1] != 256:
        patch = cv2.resize(patch, (256, 256))
    
    # Convert to float32 and normalize to [0, 1]
    patch = patch.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format
    patch = np.transpose(patch, (2, 0, 1))
    
    # Add batch dimension and duplicate channels to match expected input
    patch = np.expand_dims(patch, axis=0)  # Add batch dimension
    patch = np.repeat(patch, 2, axis=1)    # Duplicate channels to get 6 channels
    
    return patch

def postprocess_patch(patch):
    """
    Postprocess model output.
    Input shape: [1, 3, 256, 256]
    Output shape: [256, 256, 3]
    """
    # Remove batch dimension and convert from CHW to HWC
    patch = np.squeeze(patch, axis=0)
    patch = np.transpose(patch, (1, 2, 0))
    
    # Denormalize to [0, 255]
    patch = patch * 255.0
    
    # Clip values and convert to uint8
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    
    return patch

def enhance_document(image, interpreter, patch_size=256, overlap=64):
    """
    Enhance document image using float16 quantized TFLite model with patch-based processing.
    
    Args:
        image: Input image
        interpreter: TFLite interpreter
        patch_size: Size of each patch
        overlap: Overlap between patches
    
    Returns:
        enhanced_image: Enhanced image
    """
    # Get model input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create patches
    print(f"Creating patches with size {patch_size}x{patch_size} and overlap {overlap}...")
    patches, positions = create_patches(image, patch_size, overlap)
    print(f"Created {len(patches)} patches")
    
    # Process each patch
    enhanced_patches = []
    for i, patch in enumerate(patches):
        # Preprocess patch
        processed_patch = preprocess_patch(patch)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_patch)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_patch = interpreter.get_tensor(output_details[0]['index'])
        
        # Postprocess patch
        enhanced_patch = postprocess_patch(output_patch)
        
        # Resize back to original patch size if it was resized
        if patch.shape[:2] != (256, 256):
            enhanced_patch = cv2.resize(enhanced_patch, (patch.shape[1], patch.shape[0]))
        
        enhanced_patches.append(enhanced_patch)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(patches)} patches")
    
    # Merge patches
    print("Merging patches...")
    enhanced_image = merge_patches(enhanced_patches, positions, image.shape[:2], patch_size, overlap)
    
    return enhanced_image

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Float16 Quantized TFLite Document Enhancement')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--model', default='docres_model_float16.tflite', help='Float16 quantized TFLite model path')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between patches')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking mode')
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image {args.input}")
        return
    
    print(f"Loaded image with shape {image.shape}")
    
    # Load TFLite model and allocate tensors
    print(f"Loading float16 quantized TFLite model from {args.model}...")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    
    # Print input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Model input shape: {input_details[0]['shape']}")
    print(f"Model output shape: {output_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    print(f"Note: Using patch size of {args.patch_size} with overlap {args.overlap}")
    if args.patch_size != 256:
        print("Warning: Model expects 256x256 inputs. Patches will be resized for inference.")
    
    # Run benchmark if requested
    if args.benchmark:
        print("Running benchmark...")
        num_runs = 5
        total_time = 0
        
        # Warm-up run
        _ = enhance_document(image, interpreter, args.patch_size, args.overlap)
        
        # Benchmark runs
        for i in range(num_runs):
            start_time = time()
            _ = enhance_document(image, interpreter, args.patch_size, args.overlap)
            end_time = time()
            run_time = end_time - start_time
            total_time += run_time
            print(f"Run {i+1}/{num_runs}: {run_time:.2f} seconds")
        
        avg_time = total_time / num_runs
        print(f"Average inference time: {avg_time:.2f} seconds")
    
    # Enhance document
    start_time = time()
    enhanced_image = enhance_document(image, interpreter, args.patch_size, args.overlap)
    end_time = time()
    
    print(f"Enhancement completed in {end_time - start_time:.2f} seconds")
    
    # Save enhanced image
    cv2.imwrite(args.output, enhanced_image)
    print(f"Enhanced image saved to {args.output}")

if __name__ == "__main__":
    main()
''')
    
    print(f"Float16 TFLite inference script created at 'tflite_inference_float16.py'")
    print("Run with: python tflite_inference_float16.py --input <input_image> --output <output_image> --model docres_model_float16.tflite")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to float16 quantized TFLite')
    parser.add_argument('--saved_model_dir', default='tensorflow_model', help='SavedModel directory')
    parser.add_argument('--tflite_model_path', default='docres_model_float16.tflite', help='Output float16 quantized TFLite model path')
    
    args = parser.parse_args()
    
    tflite_model_path = convert_to_tflite_float16(args.saved_model_dir, args.tflite_model_path) 