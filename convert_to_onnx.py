import torch
import torch.nn as nn
import os
import sys
import numpy as np
from model_architecture import ForgeryDetectionNet, ForgeryDetectionNetAlternative
from inference import load_model

def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, device='cpu', feature_based=False):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path: Path to the PyTorch model
        onnx_model_path: Path to save the ONNX model
        device: Device to load the model on
        feature_based: Whether the model is feature-based or image-based
    """
    print(f"Converting PyTorch model at {pytorch_model_path} to ONNX format...")
    
    try:
        # Try loading with standard architecture first
        if not feature_based:
            model = load_model(pytorch_model_path, device=device)
            print("Loaded standard model architecture for conversion")
            
            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 299, 299, device=device)
            
            # Export the model
            torch.onnx.export(
                model,                      # model being run
                dummy_input,                # model input (or a tuple for multiple inputs)
                onnx_model_path,            # where to save the model
                export_params=True,         # store the trained parameter weights inside the model file
                opset_version=12,           # the ONNX version to export the model to
                do_constant_folding=True,   # whether to execute constant folding for optimization
                input_names=['input'],      # the model's input names
                output_names=['output', 'segmentation', 'attention'],  # the model's output names
                dynamic_axes={
                    'input': {0: 'batch_size'},    # variable length axes
                    'output': {0: 'batch_size'},
                    'segmentation': {0: 'batch_size'},
                    'attention': {0: 'batch_size'}
                }
            )
        else:
            # For feature-based model
            model = ForgeryDetectionNetAlternative(num_classes=2, feature_dim=768)
            model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
            model.to(device)
            model.eval()
            print("Loaded alternative model architecture for conversion")
            
            # Create a dummy input tensor for features
            dummy_input = torch.randn(1, 768, device=device)
            
            # Export the model
            torch.onnx.export(
                model,                      # model being run
                dummy_input,                # model input (or a tuple for multiple inputs)
                onnx_model_path,            # where to save the model
                export_params=True,         # store the trained parameter weights inside the model file
                opset_version=12,           # the ONNX version to export the model to
                do_constant_folding=True,   # whether to execute constant folding for optimization
                input_names=['input'],      # the model's input names
                output_names=['output', 'segmentation', 'attention'],  # the model's output names
                dynamic_axes={
                    'input': {0: 'batch_size'},    # variable length axes
                    'output': {0: 'batch_size'},
                    'segmentation': {0: 'batch_size'},
                    'attention': {0: 'batch_size'}
                }
            )
        
        print(f"Successfully converted model to ONNX format. Saved at: {onnx_model_path}")
        return True
    
    except Exception as e:
        print(f"Error during model conversion: {e}")
        return False

if __name__ == "__main__":
    # Set paths
    pytorch_model_path = 'forgery_detection_model_120.pth'
    onnx_model_path = 'forgery_detection_model.onnx'
    
    # Check if PyTorch model exists
    if not os.path.exists(pytorch_model_path):
        print(f"Error: PyTorch model not found at {pytorch_model_path}")
        sys.exit(1)
    
    # Try standard conversion first
    success = convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, feature_based=False)
    
    # If standard conversion fails, try feature-based conversion
    if not success:
        print("Trying feature-based conversion...")
        success = convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, feature_based=True)
    
    if success:
        print("Conversion completed successfully.")
    else:
        print("Conversion failed. Please check the error messages above.")
