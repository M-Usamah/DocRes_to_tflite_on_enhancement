import os
import sys
import argparse
import numpy as np
import tensorflow as tf
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)


def human_readable_size(size_bytes):
    """Convert size in bytes to human readable format (KB, MB, GB)"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def analyze_tflite_model(model_path, baseline_size=None, verbose=True):
    """
    Analyze a TFLite model to check quantization status and size
    
    Args:
        model_path: Path to the TFLite model
        baseline_size: Size of the baseline model in bytes for compression ratio calculation
        verbose: Whether to print detailed information
        
    Returns:
        dict: Model analysis results
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return None
    
    # Get model size
    size_bytes = os.path.getsize(model_path)
    size_str = human_readable_size(size_bytes)
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Model: {model_path}")
        print(f"Size: {size_str} ({size_bytes:,} bytes)")
        
        if baseline_size:
            compression_ratio = baseline_size / size_bytes
            size_reduction = (1 - size_bytes / baseline_size) * 100
            print(f"Compression ratio: {compression_ratio:.2f}x (reduced by {size_reduction:.2f}%)")
    
    # Prepare results dictionary
    results = {
        'path': model_path,
        'size_bytes': size_bytes,
        'size_str': size_str,
        'quantized': False,
        'quantization_type': None,
        'has_flex_ops': False,
        'tensor_count': 0,
        'dtype_counts': {}
    }
    
    # Load the model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        
        results['tensor_count'] = len(tensor_details)
        results['input_count'] = len(input_details)
        results['output_count'] = len(output_details)
        
        if verbose:
            print(f"\nModel details:")
            print(f"Number of tensors: {len(tensor_details)}")
            print(f"Input tensor(s): {len(input_details)}")
            print(f"Output tensor(s): {len(output_details)}")
        
        # Check for quantization
        has_int8 = False
        has_int16 = False
        has_float16 = False
        has_tf_ops = False
        
        # Count tensor types
        dtype_counts = {}
        
        for tensor in tensor_details:
            dtype = tensor['dtype'].__name__
            if dtype not in dtype_counts:
                dtype_counts[dtype] = 0
            dtype_counts[dtype] += 1
            
            if dtype == 'int8':
                has_int8 = True
            elif dtype == 'int16':
                has_int16 = True
            elif dtype == 'float16':
                has_float16 = True
        
        results['dtype_counts'] = dtype_counts
        
        # Check for TF Lite Flex ops (custom ops)
        metadata = interpreter._get_ops_details()
        if metadata:
            for op in metadata:
                if 'Flex' in op.get('custom_name', ''):
                    has_tf_ops = True
                    results['has_flex_ops'] = True
                    break
        
        if verbose:
            # Print tensor type distribution
            print("\nTensor type distribution:")
            for dtype, count in sorted(dtype_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {dtype}: {count} tensors")
            
            # Determine quantization status
            print("\nQuantization status:")
            
            # Input/output types
            print(f"Input tensor type: {input_details[0]['dtype'].__name__}")
            print(f"Output tensor type: {output_details[0]['dtype'].__name__}")
        
        results['input_type'] = input_details[0]['dtype'].__name__
        results['output_type'] = output_details[0]['dtype'].__name__
        
        # Overall quantization assessment
        if has_int8:
            results['quantized'] = True
            results['quantization_type'] = 'INT8'
            if verbose:
                print("Model uses INT8 quantization")
        if has_int16:
            results['quantized'] = True
            results['quantization_type'] = 'INT16'
            if verbose:
                print("Model uses INT16 quantization")
        if has_float16:
            results['quantized'] = True
            results['quantization_type'] = 'FLOAT16'
            if verbose:
                print("Model uses FLOAT16 quantization")
        if has_tf_ops and verbose:
            print("Model uses TensorFlow Flex ops (may require the TF Select delegate)")
        
        if not (has_int8 or has_int16 or has_float16):
            if verbose:
                print("Model is not quantized (uses full precision FP32)")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Error analyzing model: {str(e)}")
        return None

def compare_models(model_paths):
    """Compare multiple TFLite models"""
    if len(model_paths) < 2:
        print("Please provide at least two models to compare")
        return
    
    results = []
    # Find the largest model to use as baseline
    baseline_size = max(os.path.getsize(path) for path in model_paths if os.path.exists(path))
    
    # Analyze each model
    for model_path in model_paths:
        result = analyze_tflite_model(model_path, baseline_size, verbose=False)
        if result:
            results.append(result)
    
    if not results:
        print("No valid models to compare")
        return
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(f"{'Model':<30} {'Size':<15} {'Quantized':<10} {'Type':<10} {'Compression':<15} {'Tensors':<10}")
    print("-" * 100)
    
    for result in results:
        model_name = os.path.basename(result['path'])
        compression = "baseline" if result['size_bytes'] == baseline_size else f"{baseline_size/result['size_bytes']:.2f}x"
        quantized = "Yes" if result['quantized'] else "No"
        quant_type = result['quantization_type'] if result['quantization_type'] else "-"
        
        print(f"{model_name:<30} {result['size_str']:<15} {quantized:<10} {quant_type:<10} {compression:<15} {result['tensor_count']:<10}")
    
    print("\nDETAILED TENSOR TYPE DISTRIBUTION")
    print("-" * 100)
    
    # Get all unique dtypes
    all_dtypes = set()
    for result in results:
        all_dtypes.update(result['dtype_counts'].keys())
    
    # Print header
    header = "Dtype"
    for result in results:
        header += f" | {os.path.basename(result['path'])}"
    print(header)
    print("-" * len(header))
    
    # Print counts for each dtype
    for dtype in sorted(all_dtypes):
        line = f"{dtype:<6}"
        for result in results:
            count = result['dtype_counts'].get(dtype, 0)
            line += f" | {count:<{len(os.path.basename(result['path']))}}"
        print(line)
    
    print("\nINPUT/OUTPUT TYPES")
    print("-" * 100)
    print(f"{'Model':<30} {'Input Type':<15} {'Output Type':<15}")
    print("-" * 100)
    
    for result in results:
        model_name = os.path.basename(result['path'])
        print(f"{model_name:<30} {result['input_type']:<15} {result['output_type']:<15}")
    
    print("\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze TFLite model to check quantization status and size')
    parser.add_argument('model_paths', nargs='+', help='Path(s) to TFLite model file(s)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    args = parser.parse_args()
    
    if args.compare or len(args.model_paths) > 1:
        compare_models(args.model_paths)
    else:
        analyze_tflite_model(args.model_paths[0])

if __name__ == "__main__":
    main() 