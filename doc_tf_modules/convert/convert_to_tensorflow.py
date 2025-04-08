import os
import torch
import numpy as np
import onnx
import onnx_tf
import tensorflow as tf
import sys
import os.path

# Add the project root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
from models import restormer_arch

def load_pytorch_model(checkpoint_path):
    """Load the PyTorch model and its weights"""
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model

def convert_to_onnx(model, onnx_path, input_shape=(1, 6, 256, 256)):
    """Convert PyTorch model to ONNX format"""
    # Create a sample input tensor
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    
    # Export the model to ONNX format
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input
        onnx_path,                  # where to save the model
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=12,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes=dynamic_axes   # variable length axes
    )
    
    print(f"PyTorch model converted to ONNX and saved at: {onnx_path}")
    return onnx_path

def test_onnx_model(onnx_path, input_shape=(1, 6, 256, 256)):
    """Test the ONNX model with a random input tensor"""
    import onnxruntime
    
    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Create a random input tensor
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX model tested successfully. Output shape: {ort_outputs[0].shape}")
    return True

def convert_onnx_to_tensorflow(onnx_path, tf_dir):
    """Convert ONNX model to TensorFlow format"""
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert the ONNX model to TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Create directory if it doesn't exist
    os.makedirs(tf_dir, exist_ok=True)
    
    # Save the TensorFlow model
    tf_rep.export_graph(tf_dir)
    
    print(f"ONNX model converted to TensorFlow and saved at: {tf_dir}")
    return tf_dir

def test_tensorflow_model(tf_dir, input_shape=(1, 6, 256, 256)):
    """Test the TensorFlow model with a random input tensor"""
    # Load the TensorFlow model
    tf_model = tf.saved_model.load(tf_dir)
    
    # Create a random input tensor
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    infer = tf_model.signatures["serving_default"]
    output = infer(tf.constant(input_data))
    
    # Get the output tensor name from the model output
    output_name = list(output.keys())[0]
    output_tensor = output[output_name]
    
    print(f"TensorFlow model tested successfully. Output shape: {output_tensor.shape}")
    return True

def create_tensorflow_inference_script(tf_dir, script_path):
    """Create a sample inference script for the TensorFlow model"""
    code = """
import os
import numpy as np
import cv2
import tensorflow as tf

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

def preprocess_image(image_path, size=256):
    # Read and resize image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {{image_path}}")
    
    # Apply initial denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Enhance contrast globally
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Resize maintaining aspect ratio
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    
    # Create canvas of target size
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    
    # Generate enhancement prompt
    enhance_prompt = appearance_dtsprompt(canvas)
    
    # Normalize
    img_array = canvas.astype(np.float32) / 255.0
    prompt_array = enhance_prompt.astype(np.float32) / 255.0
    
    # Stack along the channel dimension
    combined = np.concatenate([img_array, prompt_array], axis=-1)
    
    # Reshape to NCHW format for the model
    combined = np.transpose(combined, (2, 0, 1))
    combined = np.expand_dims(combined, axis=0)
    
    return combined, canvas.shape[:2]

def postprocess_image(output_array, original_shape):
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TensorFlow Document Enhancement Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--model_dir', type=str, default='{0}', help='TensorFlow model directory')
    args = parser.parse_args()
    
    # Load model
    model = tf.saved_model.load(args.model_dir)
    infer = model.signatures["serving_default"]
    
    # Preprocess image
    input_tensor, original_shape = preprocess_image(args.input)
    
    # Run inference
    output = infer(tf.constant(input_tensor, dtype=tf.float32))
    
    # Get the output tensor name from the model output
    output_name = list(output.keys())[0]
    output_tensor = output[output_name].numpy()
    
    # Postprocess and save
    enhanced_img = postprocess_image(output_tensor[0], original_shape)
    cv2.imwrite(args.output, enhanced_img)
    print(f"Enhanced image saved to {{args.output}}")

if __name__ == "__main__":
    main()
""".format(tf_dir)
    
    with open(script_path, 'w') as f:
        f.write(code)
    
    print(f"TensorFlow inference script created at: {script_path}")
    return script_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DocRes model from PyTorch to TensorFlow')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to PyTorch checkpoint (.pkl file)')
    parser.add_argument('--onnx_path', type=str, default='docres_model.onnx',
                        help='Path to save the ONNX model')
    parser.add_argument('--tf_dir', type=str, default='tensorflow_model',
                        help='Directory to save the TensorFlow model')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size for the model')
    args = parser.parse_args()
    
    # Step 1: Load PyTorch model
    print("Step 1: Loading PyTorch model...")
    pytorch_model = load_pytorch_model(args.checkpoint)
    
    # Step 2: Convert PyTorch model to ONNX
    print("Step 2: Converting PyTorch model to ONNX...")
    input_shape = (1, 6, args.input_size, args.input_size)
    onnx_path = convert_to_onnx(pytorch_model, args.onnx_path, input_shape)
    
    # Step 3: Test ONNX model
    print("Step 3: Testing ONNX model...")
    test_result = test_onnx_model(onnx_path, input_shape)
    if not test_result:
        print("ONNX model test failed!")
        exit(1)
    
    # Step 4: Convert ONNX model to TensorFlow
    print("Step 4: Converting ONNX model to TensorFlow...")
    tf_dir = convert_onnx_to_tensorflow(onnx_path, args.tf_dir)
    
    # Step 5: Test TensorFlow model
    print("Step 5: Testing TensorFlow model...")
    tf_test_result = test_tensorflow_model(tf_dir, input_shape)
    if not tf_test_result:
        print("TensorFlow model test failed!")
        exit(1)
    
    # Step 6: Create inference script
    print("Step 6: Creating TensorFlow inference script...")
    inference_script = create_tensorflow_inference_script(tf_dir, "tf_inference.py")
    
    print("\nConversion completed successfully!")
    print(f"PyTorch model: {args.checkpoint}")
    print(f"ONNX model: {onnx_path}")
    print(f"TensorFlow model: {tf_dir}")
    print(f"TensorFlow inference script: {inference_script}")
    print("\nYou can run inference with:")
    print("python tf_inference.py --input your_image.jpg --output enhanced_output.jpg") 