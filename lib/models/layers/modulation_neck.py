import torch
import torch.nn as nn

class TargetModulationNeck(nn.Module):
    """
    A lightweight, trainable neck to create a target-aware feature map.
    It uses the aggregated template feature to modulate the search features via
    a simple and efficient depth-wise correlation.
    """
    def __init__(self, in_channels, out_channels=1, zero_init=True):
        super().__init__()
        
        # A simple MLP to process the aggregated template feature
        self.channel_mapper = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True)
        )

        # The depth-wise correlation is implemented as a 1x1 convolution
        # with groups=in_channels, which is very efficient.
        self.modulator = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=True)
        
        # A final layer to create a single-channel modulation mask
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        if zero_init:
            # Initialize to have no effect at the start of training
            # The modulator bias is initialized to 0, so the initial output is just the input
            # The mask_generator bias is initialized so the sigmoid output is ~0.5
            nn.init.constant_(self.modulator.bias, 0)
            nn.init.constant_(self.mask_generator[0].weight, 0)
            nn.init.constant_(self.mask_generator[0].bias, 0)

    def forward(self, search_feat_grid, template_feat_tokens):
        """
        Args:
            search_feat_grid (torch.Tensor): Search features as a 2D grid. Shape: [B, C, H, W].
            template_feat_tokens (torch.Tensor): Template features as a token list. Shape: [B, HW_template, C].
        Returns:
            torch.Tensor: The modulated search feature grid. Shape: [B, C, H, W].
        """
        # 1. Distill the template into a single representative vector
        template_agg = template_feat_tokens.mean(dim=1) # Shape: [B, C]
        
        # 2. Process the template vector to get modulation weights
        modulation_weights = self.channel_mapper(template_agg) # Shape: [B, C]
        
        # 3. Apply the modulation to the search features
        # We use a trick here: we add the processed template vector to the bias of the modulator
        # This is a highly efficient way to perform channel-wise modulation
        modulated_search = self.modulator(search_feat_grid) + modulation_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 4. Create the spatial modulation mask
        modulation_mask = self.mask_generator(modulated_search) # Shape: [B, 1, H, W]
        
        # 5. Apply the mask to the original search features
        refined_search_feat = search_feat_grid * modulation_mask
        
        return refined_search_feat
