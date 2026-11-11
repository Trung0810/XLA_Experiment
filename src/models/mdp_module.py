"""
Multi-scale Differential Perception (MDP) Module

This module implements the core innovation of M³Diff: multi-scale differential
perception for fine-grained image difference captioning.

Architecture:
1. DifferentialPerception: Channel-wise subtraction for difference features
2. MultiScaleDifferentialPerception: Extract differences at multiple scales
3. CrossAttentionFusion: Fuse difference features with original features
4. MDPModule: Complete module integrating all components

Reference: Phase2.md Section 5 - MDP Module Implementation Plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class DifferentialPerception(nn.Module):
    """
    Basic differential perception unit.
    
    Computes channel-wise difference between two feature maps and projects
    the difference to capture discriminative information.
    
    Args:
        dim: Feature dimension
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        
        # Project difference features
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
    
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a: Features from image A [B, N, C] or [B, C, H, W]
            feat_b: Features from image B [B, N, C] or [B, C, H, W]
            
        Returns:
            diff_feat: Projected difference features [B, N, C] or [B, C, H, W]
        """
        # Channel-wise absolute difference
        diff = torch.abs(feat_a - feat_b)
        
        # Handle both 3D (B, N, C) and 4D (B, C, H, W) tensors
        if diff.dim() == 4:  # [B, C, H, W]
            B, C, H, W = diff.shape
            diff = diff.flatten(2).transpose(1, 2)  # [B, H*W, C]
            diff_proj = self.projection(diff)  # [B, H*W, C]
            diff_proj = diff_proj.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        else:  # [B, N, C]
            diff_proj = self.projection(diff)
        
        return diff_proj


class MultiScaleDifferentialPerception(nn.Module):
    """
    Multi-scale differential perception.
    
    Extracts difference features at multiple scales to capture both fine-grained
    and high-level differences.
    
    Args:
        dim: Feature dimension
        num_scales: Number of scales (default: 3)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(self, dim: int, num_scales: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # Differential perception at each scale
        self.scales = nn.ModuleList([
            DifferentialPerception(dim, dropout) 
            for _ in range(num_scales)
        ])
        
        # Downsampling for multi-scale processing
        self.downsample = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])
        
        # Fusion of multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
    
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a: Features from image A [B, C, H, W]
            feat_b: Features from image B [B, C, H, W]
            
        Returns:
            fused_diff: Multi-scale fused difference features [B, N, C]
        """
        B, C, H, W = feat_a.shape
        
        # Extract differences at multiple scales
        diff_features = []
        
        for scale_idx, (diff_module, downsample) in enumerate(zip(self.scales, self.downsample)):
            # Downsample to current scale
            feat_a_scaled = downsample(feat_a)  # [B, C, H_s, W_s]
            feat_b_scaled = downsample(feat_b)  # [B, C, H_s, W_s]
            
            # Compute difference at this scale
            diff = diff_module(feat_a_scaled, feat_b_scaled)  # [B, C, H_s, W_s]
            
            # Upsample back to original size if downsampled
            if scale_idx > 0:
                diff = F.interpolate(diff, size=(H, W), mode='bilinear', align_corners=False)
            
            diff_features.append(diff)
        
        # Concatenate multi-scale features
        # [B, C*num_scales, H, W]
        multi_scale_diff = torch.cat(diff_features, dim=1)
        
        # Reshape for fusion: [B, H*W, C*num_scales]
        multi_scale_diff = multi_scale_diff.flatten(2).transpose(1, 2)
        
        # Fuse multi-scale features: [B, H*W, C]
        fused_diff = self.fusion(multi_scale_diff)
        
        return fused_diff


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between original and difference features.
    
    Uses multi-head attention where:
    - Query: Original features
    - Key/Value: Difference features
    
    This allows the model to selectively attend to relevant differences.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, original_feat: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original_feat: Original image features [B, N, C]
            diff_feat: Difference features [B, N, C]
            
        Returns:
            fused_feat: Enhanced features [B, N, C]
        """
        # Cross-attention: Q from original, K/V from difference
        attn_out, _ = self.cross_attn(
            query=original_feat,
            key=diff_feat,
            value=diff_feat
        )
        
        # Residual connection + LayerNorm
        feat = self.norm1(original_feat + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(feat)
        
        # Residual connection + LayerNorm
        fused_feat = self.norm2(feat + ffn_out)
        
        return fused_feat


class MDPModule(nn.Module):
    """
    Complete Multi-scale Differential Perception Module.
    
    This is the main MDP module that integrates:
    1. Multi-scale differential perception
    2. Cross-attention fusion for both images
    
    Args:
        dim: Feature dimension (e.g., 1024 for CLIP ViT-L/14)
        num_scales: Number of scales for multi-scale processing (default: 3)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
    
    Usage:
        mdp = MDPModule(dim=1024, num_scales=3, num_heads=8)
        feat_a_enhanced, feat_b_enhanced = mdp(feat_a, feat_b)
    """
    
    def __init__(
        self,
        dim: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        
        # Multi-scale differential perception
        self.multi_scale_diff = MultiScaleDifferentialPerception(
            dim=dim,
            num_scales=num_scales,
            dropout=dropout
        )
        
        # Cross-attention fusion for both images
        self.fusion_a = CrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.fusion_b = CrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of MDP module.
        
        Args:
            feat_a: Features from image A [B, C, H, W]
            feat_b: Features from image B [B, C, H, W]
            
        Returns:
            feat_a_enhanced: Enhanced features for image A [B, N, C]
            feat_b_enhanced: Enhanced features for image B [B, N, C]
            where N = H * W (flattened spatial dimensions)
        """
        B, C, H, W = feat_a.shape
        
        # Extract multi-scale difference features
        diff_feat = self.multi_scale_diff(feat_a, feat_b)  # [B, H*W, C]
        
        # Flatten original features for attention
        feat_a_flat = feat_a.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat_b_flat = feat_b.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Cross-attention fusion: enhance original features with difference info
        feat_a_enhanced = self.fusion_a(feat_a_flat, diff_feat)  # [B, H*W, C]
        feat_b_enhanced = self.fusion_b(feat_b_flat, diff_feat)  # [B, H*W, C]
        
        return feat_a_enhanced, feat_b_enhanced


# Test function
def test_mdp_module():
    """Test MDP module with dummy data"""
    print("Testing MDP Module...")
    print("-" * 50)
    
    # Configuration
    batch_size = 2
    channels = 1024  # CLIP ViT-L/14 output dim
    height = 24  # 336/14 = 24 for CLIP image size 336
    width = 24
    
    # Create dummy features
    feat_a = torch.randn(batch_size, channels, height, width)
    feat_b = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {feat_a.shape}")
    
    # Initialize MDP module
    mdp = MDPModule(dim=channels, num_scales=3, num_heads=8, dropout=0.1)
    
    # Forward pass
    feat_a_enhanced, feat_b_enhanced = mdp(feat_a, feat_b)
    
    print(f"Output shape A: {feat_a_enhanced.shape}")
    print(f"Output shape B: {feat_b_enhanced.shape}")
    print(f"Expected: [batch_size={batch_size}, N={height*width}, C={channels}]")
    
    # Verify output shapes
    assert feat_a_enhanced.shape == (batch_size, height * width, channels)
    assert feat_b_enhanced.shape == (batch_size, height * width, channels)
    
    print("-" * 50)
    print("[PASS] MDP Module test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in mdp.parameters())
    trainable_params = sum(p.numel() for p in mdp.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return mdp


if __name__ == "__main__":
    # Run test
    mdp_module = test_mdp_module()
