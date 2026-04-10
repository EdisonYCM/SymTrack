import torch
import torch.nn as nn
import os
import sys
import argparse
import importlib
from torchinfo import summary

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.models.symtrack import build_symtrack

class VisualBackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        
        embed_dim = getattr(model.backbone, 'embed_dim', 768)
        
        if hasattr(model, 'query_embed') and model.query_embed is not None:
            self.track_query = model.query_embed.weight.unsqueeze(0)
        elif hasattr(model, 'box_head') and hasattr(model.box_head, 'query_embed') and model.box_head.query_embed is not None:
            self.track_query = model.box_head.query_embed.weight.unsqueeze(0)
        else:
            self.track_query = torch.zeros(1, 1, embed_dim).cuda()
            
    def forward(self, z, x):
        """
        Args:
            z: Template Tensor [B, 3, H, W]
            x: Search Tensor [B, 3, H, W]
        """
        z_input = [z] 
        x_input = x 
        return self.backbone(z_input, x_input, 
                             ce_template_mask=None, 
                             ce_keep_rate=None, 
                             return_last_attn=False,
                             track_query=self.track_query)

def main():
    parser = argparse.ArgumentParser(description='Analyze Model Layers')
    parser.add_argument('--script', type=str, default='symtrack')
    parser.add_argument('--config', type=str, default='baseline_text_scalear') 
    args = parser.parse_args()

    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)

    print(f"Loading model with config: {args.config}...")
    model = build_symtrack(cfg, training=False)
    
    model.float() 
    model.cuda()
    model.eval()

    print("\n" + "="*40)
    print(" [1/3] Analyzing Visual Backbone (ViT-B)")
    print("="*40)
    
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    template = torch.randn(1, 3, z_sz, z_sz).cuda()
    search = torch.randn(1, 3, x_sz, x_sz).cuda()

    backbone_wrapper = VisualBackboneWrapper(model)
    summary(backbone_wrapper, 
            input_data=[template, search], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=3)

    print("\n" + "="*40)
    print(" [2/3] Analyzing PTR Module (Neck)")
    print("="*40)
    
    if hasattr(model, 'neck') and model.neck is not None:
        ptr_module = model.neck
    else:
        print("PTR not in model, creating a temporary one for analysis...")
        from lib.models.layers.modulation_neck import TargetModulationNeck
        ptr_module = TargetModulationNeck(in_channels=768).cuda()
    
    c_dim = 768 
    feat_sz = x_sz // 16
    feat_grid = torch.randn(1, c_dim, feat_sz, feat_sz).cuda()
    feat_tokens = torch.randn(1, (z_sz // 16)**2, c_dim).cuda()

    summary(ptr_module, 
            input_data=[feat_grid, feat_tokens],
            col_names=["output_size", "num_params", "kernel_size"],
            depth=3)

    print("\n" + "="*40)
    print(" [3/3] Analyzing CEC Module (TextAttnNeck)")
    print("="*40)

    if hasattr(model, 'text_attention_neck') and model.text_attention_neck is not None:
        cec_module = model.text_attention_neck
    else:
        print("CEC not in model, creating a temporary one for analysis...")
        from lib.models.layers.text_attention_neck import TextAttentionNeck
        cec_module = TextAttentionNeck(text_dim=1024, embed_dim=256, num_heads=4, output_size=(16,16)).cuda()

    try:
        text_dim = cec_module.text_proj.in_features
    except:
        text_dim = 1024

    num_patches = 784 
    z_text = torch.randn(1, num_patches, text_dim).cuda().float()
    x_text = torch.randn(1, num_patches, text_dim).cuda().float()

    summary(cec_module, 
            input_data=[z_text, x_text],
            col_names=["output_size", "num_params", "kernel_size"],
            depth=3)

if __name__ == "__main__":
    main()