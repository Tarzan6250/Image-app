import numpy as np
import cv2
from PIL import Image
import io
import os

class ForensicFeatureExtractor:
    """Class for extracting forensic features from images for forgery detection"""
    
    def __init__(self):
        """Initialize the feature extractor"""
        pass
    
    def extract_features(self, image, feature_dim=768):
        """
        Extract forensic features from an image for forgery detection.
        
        Args:
            image: Input image (numpy array or PIL Image)
            feature_dim: Dimension of the output feature vector
            
        Returns:
            Feature vector of dimension feature_dim
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Extract various forensic features
        features = []
        
        # 1. Error Level Analysis (ELA) features
        ela_features = self._extract_ela_features(image)
        features.extend(ela_features)
        
        # 2. Noise analysis features
        noise_features = self._extract_noise_features(image)
        features.extend(noise_features)
        
        # 3. JPEG compression features
        jpeg_features = self._extract_jpeg_features(image)
        features.extend(jpeg_features)
        
        # 4. Color features
        color_features = self._extract_color_features(image)
        features.extend(color_features)
        
        # 5. Texture features
        texture_features = self._extract_texture_features(image)
        features.extend(texture_features)
        
        # Ensure the feature vector has the correct dimension
        features = np.array(features, dtype=np.float32)
        
        # If the feature vector is too long, truncate it
        if len(features) > feature_dim:
            features = features[:feature_dim]
        
        # If the feature vector is too short, pad it with zeros
        elif len(features) < feature_dim:
            features = np.pad(features, (0, feature_dim - len(features)), 'constant')
        
        return features
    
    def _extract_ela_features(self, image):
        """Extract Error Level Analysis features"""
        # Convert to PIL Image for JPEG compression
        pil_img = Image.fromarray(image)
        
        # Save with a specific JPEG quality
        quality = 90
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load the compressed image
        compressed_img = np.array(Image.open(buffer))
        
        # Calculate the ELA (difference between original and compressed)
        ela = np.abs(image.astype(np.float32) - compressed_img.astype(np.float32))
        
        # Scale for better visualization
        ela_scaled = np.clip(ela * 10, 0, 255).astype(np.uint8)
        
        # Convert to grayscale for feature extraction
        if len(ela_scaled.shape) == 3:
            ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_scaled
        
        # Extract statistical features from the ELA image
        features = []
        
        # Mean and standard deviation
        features.append(np.mean(ela_gray))
        features.append(np.std(ela_gray))
        
        # Histogram features (10 bins)
        hist = cv2.calcHist([ela_gray], [0], None, [10], [0, 256])
        hist = hist.flatten() / np.sum(hist)  # Normalize
        features.extend(hist)
        
        return features
    
    def _extract_noise_features(self, image):
        """Extract noise pattern features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply noise extraction filter
        noise = cv2.medianBlur(gray, 5) - gray
        
        # Extract statistical features from the noise
        features = []
        
        # Mean and standard deviation
        features.append(np.mean(noise))
        features.append(np.std(noise))
        
        # Histogram features (10 bins)
        hist = cv2.calcHist([noise], [0], None, [10], [0, 256])
        hist = hist.flatten() / np.sum(hist)  # Normalize
        features.extend(hist)
        
        return features
    
    def _extract_jpeg_features(self, image):
        """Extract JPEG compression artifacts features"""
        # Convert to YCrCb color space
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            y_channel = ycrcb[:, :, 0]
        else:
            y_channel = image
        
        # Apply DCT transform
        h, w = y_channel.shape
        h_pad = h - (h % 8)
        w_pad = w - (w % 8)
        y_channel = y_channel[:h_pad, :w_pad]
        
        # Compute DCT coefficients for 8x8 blocks
        features = []
        for i in range(0, h_pad, 8):
            for j in range(0, w_pad, 8):
                block = y_channel[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                # Use the first few AC coefficients
                features.append(dct_block[0, 1])
                features.append(dct_block[1, 0])
                features.append(dct_block[1, 1])
                features.append(dct_block[2, 0])
                
                # Break after processing a few blocks to limit feature size
                if len(features) >= 100:
                    break
            if len(features) >= 100:
                break
        
        # Ensure we have exactly 100 features
        features = np.array(features[:100], dtype=np.float32)
        if len(features) < 100:
            features = np.pad(features, (0, 100 - len(features)), 'constant')
        
        return features
    
    def _extract_color_features(self, image):
        """Extract color-based features"""
        features = []
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Split into channels
        r, g, b = cv2.split(image)
        
        # Calculate mean and std for each channel
        for channel in [r, g, b]:
            features.append(np.mean(channel))
            features.append(np.std(channel))
        
        # Calculate color correlation
        features.append(np.corrcoef(r.flat, g.flat)[0, 1])
        features.append(np.corrcoef(r.flat, b.flat)[0, 1])
        features.append(np.corrcoef(g.flat, b.flat)[0, 1])
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features using Haralick texture features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Reduce to 8 gray levels to speed up calculation
        gray = (gray / 32).astype(np.uint8)
        
        # Calculate GLCM (Gray-Level Co-occurrence Matrix)
        glcm = np.zeros((8, 8), dtype=np.float32)
        h, w = gray.shape
        
        # Simple GLCM calculation for horizontal adjacency
        for i in range(h):
            for j in range(w-1):
                glcm[gray[i, j], gray[i, j+1]] += 1
        
        # Normalize GLCM
        glcm = glcm / np.sum(glcm)
        
        # Calculate Haralick features
        features = []
        
        # Energy
        features.append(np.sum(glcm**2))
        
        # Contrast
        contrast = 0
        for i in range(8):
            for j in range(8):
                contrast += (i-j)**2 * glcm[i, j]
        features.append(contrast)
        
        # Homogeneity
        homogeneity = 0
        for i in range(8):
            for j in range(8):
                homogeneity += glcm[i, j] / (1 + abs(i-j))
        features.append(homogeneity)
        
        # Correlation
        mu_i = np.sum(np.arange(8) * np.sum(glcm, axis=1))
        mu_j = np.sum(np.arange(8) * np.sum(glcm, axis=0))
        sigma_i = np.sqrt(np.sum(((np.arange(8) - mu_i) ** 2) * np.sum(glcm, axis=1)))
        sigma_j = np.sqrt(np.sum(((np.arange(8) - mu_j) ** 2) * np.sum(glcm, axis=0)))
        
        correlation = 0
        for i in range(8):
            for j in range(8):
                if sigma_i > 0 and sigma_j > 0:
                    correlation += (i - mu_i) * (j - mu_j) * glcm[i, j] / (sigma_i * sigma_j)
        features.append(correlation)
        
        return features
    
    def visualize_ela(self, image, quality=90, scale=10):
        """
        Visualize Error Level Analysis (ELA) to highlight potential tampered regions.
        
        Args:
            image: Input image (numpy array or PIL Image)
            quality: JPEG quality for compression
            scale: Scaling factor for ELA visualization
            
        Returns:
            ELA visualization image
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to PIL Image for JPEG compression
        pil_img = Image.fromarray(image)
        
        # Save with a specific JPEG quality
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load the compressed image
        compressed_img = np.array(Image.open(buffer))
        
        # Calculate the ELA (difference between original and compressed)
        ela = np.abs(image.astype(np.float32) - compressed_img.astype(np.float32))
        
        # Scale for better visualization
        ela_visualization = np.clip(ela * scale, 0, 255).astype(np.uint8)
        
        return ela_visualization
