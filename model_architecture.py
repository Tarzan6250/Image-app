import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    """Attention mechanism to focus on suspicious regions"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map
        attention = self.sigmoid(self.conv(x))
        # Apply attention
        return x * attention.expand_as(x), attention

class ForgeryDetectionNet(nn.Module):
    """Image Forgery Detection Network with attention mechanism"""
    def __init__(self, num_classes=2, pretrained=True, with_segmentation=True):
        super(ForgeryDetectionNet, self).__init__()
        
        # Use a pre-trained ResNet as the backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and FC
        
        # Feature pyramid for multi-scale feature extraction
        self.pyramid_layers = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),  # P5
            nn.Conv2d(1024, 256, kernel_size=1),  # P4
            nn.Conv2d(512, 256, kernel_size=1),   # P3
        ])
        
        # Attention blocks for each pyramid level
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(256),
            AttentionBlock(256),
            AttentionBlock(256)
        ])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # U-Net-style decoder head
        self.with_segmentation = with_segmentation
        if with_segmentation:
            # Decoder conv blocks
            self.dec_conv1 = nn.Sequential(
                nn.Conv2d(256 * 3, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec_conv2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec_conv3 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.seg_final = nn.Conv2d(64, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Get original image size for upsampling
        orig_size = x.size()[2:]
        
        # Extract features from backbone
        features = []
        x = self.backbone[0](x)  # Conv1
        x = self.backbone[1](x)  # BN1
        x = self.backbone[2](x)  # ReLU
        x = self.backbone[3](x)  # MaxPool
        
        x = self.backbone[4](x)  # Layer1
        features.append(x)  # Save for FPN
        
        x = self.backbone[5](x)  # Layer2
        features.append(x)  # Save for FPN
        
        x = self.backbone[6](x)  # Layer3
        features.append(x)  # Save for FPN
        
        x = self.backbone[7](x)  # Layer4
        features.append(x)  # Save for FPN
        
        # Apply feature pyramid
        pyramid_features = []
        attention_maps = []
        
        # Process P5 (from Layer4)
        p5 = self.pyramid_layers[0](features[3])
        p5, att5 = self.attention_blocks[0](p5)
        pyramid_features.append(p5)
        attention_maps.append(att5)
        
        # Process P4 (from Layer3)
        p4 = self.pyramid_layers[1](features[2])
        p4, att4 = self.attention_blocks[1](p4)
        pyramid_features.append(p4)
        attention_maps.append(att4)
        
        # Process P3 (from Layer2)
        p3 = self.pyramid_layers[2](features[1])
        p3, att3 = self.attention_blocks[2](p3)
        pyramid_features.append(p3)
        attention_maps.append(att3)
        
        # Global average pooling on each pyramid level
        pooled_features = [self.gap(feat) for feat in pyramid_features]
        pooled_features = [feat.view(feat.size(0), -1) for feat in pooled_features]
        
        # Concatenate pooled features
        concat_features = torch.cat(pooled_features, dim=1)
        
        # Classification
        classification = self.classifier(concat_features)
        
        if self.with_segmentation:
            # Upsample all pyramid features to the same size
            p3_size = pyramid_features[2].size()[2:]
            p5_up = F.interpolate(pyramid_features[0], size=p3_size, mode='bilinear', align_corners=True)
            p4_up = F.interpolate(pyramid_features[1], size=p3_size, mode='bilinear', align_corners=True)
            
            # Concatenate for segmentation
            seg_features = torch.cat([p5_up, p4_up, pyramid_features[2]], dim=1)
            
            # U-Net-style decode
            x = self.dec_conv1(seg_features)
            x = self.up1(x)
            x = self.dec_conv2(x)
            x = self.up2(x)
            x = self.dec_conv3(x)
            x = self.up3(x)
            segmentation = self.sigmoid(self.seg_final(x))
            
            # Resize segmentation to original input size if needed
            if segmentation.size()[2:] != orig_size:
                segmentation = F.interpolate(segmentation, size=orig_size, mode='bilinear', align_corners=True)
            
            return classification, segmentation, attention_maps
        else:
            return classification, attention_maps

class ForgeryDetectionNetAlternative(nn.Module):
    """Alternative model for 768-dimensional feature input"""
    def __init__(self, num_classes=2, feature_dim=768, with_segmentation=True):
        super(ForgeryDetectionNetAlternative, self).__init__()
        
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # For segmentation if needed
        self.with_segmentation = with_segmentation
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Process features
        features = self.feature_layers(x)
        
        # Classification
        classification = self.classifier(features)
        
        # In this simplified model, we don't actually generate a segmentation mask
        # from the feature vector, but we could add a decoder network if needed
        if self.with_segmentation:
            # Return dummy segmentation and attention for API compatibility
            batch_size = x.size(0)
            # Create a dummy segmentation mask (would be replaced with actual implementation)
            segmentation = torch.zeros((batch_size, 1, 32, 32), device=x.device)
            # Create a dummy attention map
            attention_maps = [torch.zeros((batch_size, 1, 16, 16), device=x.device)]
            
            return classification, segmentation, attention_maps
        else:
            # Create a dummy attention map for API compatibility
            attention_maps = [torch.zeros((x.size(0), 1, 16, 16), device=x.device)]
            return classification, attention_maps
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
