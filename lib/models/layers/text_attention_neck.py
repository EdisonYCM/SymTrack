import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextAttentionNeck(nn.Module):
    """
    This module receives text features extracted by the TokenFD backbone,
    performs template-search matching through a cross-attention mechanism,
    and finally generates an attention mask that is fully aligned with the
    spatial size of the ODTrack feature map.
    """
    def __init__(self, text_dim: int, embed_dim: int, num_heads: int, output_size: tuple):
        """
        Args:
            text_dim (int): Dimensionality of the TokenFD features (1024).
            embed_dim (int): Internal embedding dimension of the cross-attention module.
            num_heads (int): Number of heads in multi-head attention.
            output_size (tuple): Spatial size of the final output mask, e.g., (16, 16).
        """
        super().__init__()
        self.output_size = output_size
        
        # 1. Linear projection layer
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # 2. Cross-attention module
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Mask generation head
        # This lightweight CNN maps and downsamples features from the
        # TokenFD space (28x28) to the ODTrack space (16x16).
        # The number of TokenFD patches is 28*28 = 784.
        self.mask_head = nn.Sequential(
            # Input: [B, embed_dim, 28, 28]
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Use a strided convolution for downsampling.
            # This is better than direct interpolation because it is learnable.
            # (28 -> 14)
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Adaptive pooling ensures the exact target output size regardless of the input.
            nn.AdaptiveAvgPool2d(output_size),
            # Final 1x1 convolution to produce a single-channel mask.
            nn.Conv2d(embed_dim // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, z_text, x_text):
        """
        Args:
            z_text (torch.Tensor): TokenFD template features, [B, Nzt, 1024] (Nzt=784)
            x_text (torch.Tensor): TokenFD search features, [B, Nxt, 1024] (Nxt=784)
        """
        # Step A: Project into the embedding space
        z_text_proj = self.text_proj(z_text) # [B, Nzt, 256]
        x_text_proj = self.text_proj(x_text) # [B, Nxt, 256]

        # Step B: Cross-attention matching
        text_enhancement, _ = self.cross_attention(query=x_text_proj, key=z_text_proj, value=z_text_proj)
        x_text_fused = self.norm(x_text_proj + text_enhancement) # [B, Nxt, 256]

        # Step C: Reshape into a 2D grid for the mask head
        bs, hw, c = x_text_fused.shape
        h = w = int(math.sqrt(hw)) # h = w = 28
        x_text_grid = x_text_fused.transpose(1, 2).view(bs, c, h, w)

        # Step D: Generate the final aligned attention map through the mask head
        text_attention_mask = self.mask_head(x_text_grid) # [B, 1, 16, 16]
        
        return text_attention_mask
