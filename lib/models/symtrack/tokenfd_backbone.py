import torch
import torch.nn as nn

from lib.models.vendor.internvision import InternVisionModel

class TokenFDViT(nn.Module):
    """
    Wraps the pretrained TokenFD InternVisionModel as a dedicated
    text feature extractor.
    The weights of this module should be frozen during training.
    """
    def __init__(self, checkpoint_path, torch_dtype=torch.bfloat16):
        super().__init__()
        print(f"Loading TokenFD visual backbone from: {checkpoint_path}")
        self.vision_encoder = InternVisionModel.from_pretrained(
            checkpoint_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype
        )
        self.output_dim = self.vision_encoder.config.hidden_size  # 1024
        # Compute the number of patches based on the model configuration
        self.num_patches = (self.vision_encoder.config.image_size // self.vision_encoder.config.patch_size) ** 2

    def forward(self, images: torch.Tensor):
        """
        Forward pass.
        Args:
            images (torch.Tensor): Input image tensor with shape [B, 3, H, W].
                                   H and W must match the pretrained model input size (448x448).
        Returns:
            torch.Tensor: Returns patch tokens with shape [B, NumPatches, C].
                          We explicitly remove the CLS token because patch tokens
                          are more commonly used for dense prediction in SOT.
        """
        # The visual encoder output is a tuple containing
        # last_hidden_state and pooled_output
        outputs = self.vision_encoder(pixel_values=images)
        
        # The shape of last_hidden_state is [B, NumPatches + 1, C]
        last_hidden_state = outputs[0]
        
        # The first token in the sequence is the CLS token,
        # and the rest are patch tokens
        # We only return patch tokens because they contain rich spatial information
        patch_tokens = last_hidden_state[:, 1:]
        
        assert patch_tokens.shape[1] == self.num_patches, "The number of patch tokens does not match the expected value."
        
        return patch_tokens
