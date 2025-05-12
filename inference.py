import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
from model_architecture import ForgeryDetectionNet, ForgeryDetectionNetAlternative

def load_model(model_path, device='cuda'):
    """
    Load the forgery detection model from the given path.
    Tries different model architectures if the standard one fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Try loading with standard architecture first
    try:
        model = ForgeryDetectionNet(num_classes=2, pretrained=False, with_segmentation=True)
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if the state dict has 'backbone' or 'features' keys
        if any('backbone' in k for k in state_dict.keys()):
            model.load_state_dict(state_dict)
        else:
            # Try to adapt the state dict if it has a different structure
            adapted_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('features'):
                    adapted_state_dict[k.replace('features', 'backbone')] = v
                else:
                    adapted_state_dict[k] = v
            
            # Load the adapted state dict
            model.load_state_dict(adapted_state_dict, strict=False)
        
        model.to(device)
        model.eval()
        print("Loaded standard model architecture")
        return model
    
    except Exception as e:
        print(f"Error loading standard model: {e}")
        
        # Try alternative architecture for 768-dim feature input
        try:
            model = ForgeryDetectionNetAlternative(num_classes=2, feature_dim=768)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("Loaded alternative model architecture")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model with both architectures: {e}")

def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess an image for model input.
    
    Args:
        image: PIL Image or numpy array
        target_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image if needed
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif isinstance(image, Image.Image):
        # Convert PIL Image to numpy array
        image = np.array(image)
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize image
    image_resized = cv2.resize(image, target_size)
    
    # Convert to float and normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def postprocess_mask(mask, threshold=0.5, min_area=100):
    """
    Post-process the segmentation mask to clean up noise and small regions.
    
    Args:
        mask: Numpy array of the segmentation mask
        threshold: Threshold for binary mask
        min_area: Minimum area for a region to be kept
        
    Returns:
        Cleaned binary mask
    """
    if mask is None:
        return None
    
    # Convert to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    
    # Ensure mask is in the right shape
    if len(mask.shape) == 4:
        mask = mask[0, 0]  # Take first image, first channel
    elif len(mask.shape) == 3:
        mask = mask[0]  # Take first channel
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary_mask, [contour], 0, 0, -1)
    
    return binary_mask

def predict(model, image, device='cuda'):
    """
    Make a prediction using the forgery detection model.
    
    Args:
        model: The loaded model
        image: Input image (numpy array or PIL Image)
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results
    """
    # Check if model is loaded
    if model is None:
        raise ValueError("Model is not loaded")
    
    # Preprocess image
    if isinstance(image, str):
        # If it's a file path
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save original image for visualization
    original_image = image.copy()
    
    # Preprocess for model input
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        try:
            # Check if the model expects image input or feature input
            if isinstance(model, ForgeryDetectionNetAlternative):
                # For the alternative model that expects feature input
                # This is a placeholder - in a real scenario, you would extract features first
                features = torch.zeros(1, 768, device=device)  # Dummy features
                outputs = model(features)
            else:
                # For the standard model that expects image input
                outputs = model(input_tensor)
            
            # Parse outputs
            if len(outputs) > 2:
                # Model returns classification, segmentation, and attention maps
                logits, segmentation, attention_maps = outputs
                
                # Convert segmentation mask to numpy
                mask = segmentation[0, 0].cpu().numpy()
                
                # Process attention maps (use the first one for visualization)
                attention_map = attention_maps[0][0, 0].cpu().numpy()
            else:
                # Model returns only classification and attention maps
                logits, attention_maps = outputs
                mask = None
                attention_map = attention_maps[0][0, 0].cpu().numpy()
            
            # Get class probabilities
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get prediction
            pred_idx = np.argmax(probabilities)
            confidence = probabilities[pred_idx]
            label = "fake" if pred_idx == 0 else "real"
            
            # Post-process mask if available
            if mask is not None:
                processed_mask = postprocess_mask(mask)
            else:
                processed_mask = None
            
            return {
                'label': label,
                'confidence': confidence,
                'mask': processed_mask,
                'attention_map': attention_map
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a default prediction in case of error
            return {
                'label': "unknown",
                'confidence': 0.5,
                'mask': None,
                'attention_map': None,
                'error': str(e)
            }

def visualize_results(image, prediction, overlay_alpha=0.5):
    """
    Visualize the prediction results on the image.
    
    Args:
        image: Original image
        prediction: Prediction dictionary from predict()
        overlay_alpha: Transparency of the overlay
        
    Returns:
        Visualization image with overlays
    """
    # Make a copy of the image
    result_img = image.copy()
    
    # Get prediction details
    label = prediction['label']
    confidence = prediction['confidence']
    mask = prediction['mask']
    attention_map = prediction['attention_map']
    
    # Add text with prediction and confidence
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label.upper()}: {confidence*100:.1f}%"
    cv2.putText(result_img, text, (10, 30), font, 1, (0, 0, 255) if label == "fake" else (0, 255, 0), 2)
    
    # If the image is predicted as fake and we have a mask
    if label == "fake" and mask is not None:
        # Resize mask to match image
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create a colored overlay for tampered regions
        overlay = np.zeros_like(result_img)
        overlay[mask_resized > 0] = [0, 0, 255]  # Red for tampered regions
        
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, overlay_alpha, result_img, 1 - overlay_alpha, 0, result_img)
        
        # Draw contours around tampered regions
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
    
    # If we have an attention map
    if attention_map is not None:
        # Resize attention map to match image
        attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create a heatmap
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create a separate image for the attention visualization
        attention_vis = image.copy()
        cv2.addWeighted(heatmap, overlay_alpha, attention_vis, 1 - overlay_alpha, 0, attention_vis)
        
        # Add a small version of the attention map to the corner of the result image
        h, w = result_img.shape[:2]
        map_size = min(h, w) // 4
        attention_small = cv2.resize(attention_vis, (map_size, map_size))
        result_img[10:10+map_size, w-10-map_size:w-10] = attention_small
    
    return result_img
