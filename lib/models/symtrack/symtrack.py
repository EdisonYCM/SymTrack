import math
import os
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.symtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.symtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.layers.modulation_neck import TargetModulationNeck

from lib.models.layers.text_attention_neck import TextAttentionNeck
from lib.models.symtrack.tokenfd_backbone import TokenFDViT 
import torch.distributed as dist

class SymTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, generic_backbone, text_backbone, box_head, modulation_neck, text_attention_neck, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = generic_backbone
        self.text_backbone = text_backbone # TokenFD ViT
        self.modulation_neck = modulation_neck # PTR's modulation neck
        self.text_attention_neck = text_attention_neck # CEC's text attention neck
        
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        # self.track_query = None
        self.register_buffer("track_query", torch.zeros(1, token_len, generic_backbone.embed_dim), persistent=False)
        self.track_query_is_valid = False 
        
        self.token_len = token_len

    
    def forward(self, template: torch.Tensor,
              search: torch.Tensor,
              template_tokenfd: List[torch.Tensor], 
                search_tokenfd: List[torch.Tensor],   
              ce_template_mask=None,
              ce_keep_rate=None,
              return_last_attn=False,
              ):
        assert isinstance(search, list), "The type of search is not List"
        assert isinstance(template_tokenfd, list), "The type of template_tokenfd is not List"
        assert isinstance(search_tokenfd, list), "The type of search_tokenfd is not List"

        # template_cat = torch.cat(template, dim=0)
        template_tokenfd_cat = torch.cat(template_tokenfd, dim=0)
        
        out_dict_list = []
        for i in range(len(search)):
            query_to_pass = self.track_query if self.track_query_is_valid else None
            
            # Parallel Feature Extraction with the backbone (ViT)
            x_all_tokens, aux_dict = self.backbone(z=template, x=search[i],
                                                   ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                                   return_last_attn=return_last_attn, 
                                                #    track_query=self.track_query,
                                                    track_query=query_to_pass, 
                                                   token_len=self.token_len)
            
            correct_bs = search_tokenfd[i].shape[0]

            if x_all_tokens.shape[0] != correct_bs:
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    start_idx = rank * correct_bs
                    end_idx = start_idx + correct_bs
                    x_all_tokens = x_all_tokens[start_idx:end_idx]

            # Text Backbone (TokenFD ViT)
            # search_tokenfd_cat = torch.cat([s[i:i+1] for s in search_tokenfd], dim=0) if len(search_tokenfd) > 1 and search_tokenfd[0].shape[0] > 1 else search_tokenfd[i]
            search_img_tokenfd = search_tokenfd[i]
            with torch.no_grad():
                z_text = self.text_backbone(template_tokenfd_cat.to(torch.bfloat16))
                x_text = self.text_backbone(search_img_tokenfd.to(torch.bfloat16))

            # Feature Separation & PTR ModulationNeck 
            feat_last = x_all_tokens[-1] if isinstance(x_all_tokens, list) else x_all_tokens
            x_generic_tokens = feat_last[:, -self.feat_len_s:]
            
            if self.backbone.add_cls_token:
                z_generic_tokens = feat_last[:, self.token_len:-self.feat_len_s]
            else:
                z_generic_tokens = feat_last[:, :-self.feat_len_s]

            bs, hw, c = x_generic_tokens.shape
            h = w = int(math.sqrt(hw))
            x_generic_grid = x_generic_tokens.transpose(1, 2).view(bs, c, h, w)
            x_feat_modulated_grid = self.modulation_neck(x_generic_grid, z_generic_tokens)

            # Text Enhancement with CEC TextAttentionNeck 
            bs, num_patches_x, c_text = x_text.shape
            num_templates = z_text.shape[0] // bs
            
            z_text_reshaped = z_text.view(num_templates, bs, -1, c_text).permute(1, 0, 2, 3).reshape(bs, -1, c_text)

            text_attention_mask = self.text_attention_neck(z_text_reshaped, x_text)


            # Final Fusion 
            final_feat_grid = x_feat_modulated_grid * text_attention_mask

            # Prediction Head 
            enc_opt_refined = final_feat_grid.flatten(2).transpose(1, 2)
            
            if self.backbone.add_cls_token:
                self.track_query = (x_all_tokens[:, :self.token_len].clone()).detach()
                self.track_query_is_valid = True 
            
            att = torch.matmul(enc_opt_refined, x_all_tokens[:, :self.token_len].transpose(1, 2))
            
            opt = (enc_opt_refined.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['text_attention_mask'] = text_attention_mask
            out['backbone_feat'] = x_all_tokens
            out_dict_list.append(out)
            
        return out_dict_list


    def forward_head(self, enc_opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        """
        enc_opt: (B, Nq, C, HW)
        """
        bs, Nq, C, HW = enc_opt.size()
        opt_feat = enc_opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # [B*Nq, C, H, W]

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_symtrack(cfg, training=True):
    print("Building SymTrack (Dual-Backbone with PTR + CEC)...")

    current_dir = os.path.dirname(os.path.abspath(__file__))  
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    
    # Build Text Backbone
    text_backbone = TokenFDViT(checkpoint_path=cfg.MODEL.TEXT_BACKBONE.CHECKPOINT)
    for param in text_backbone.parameters():
        param.requires_grad = False
    text_backbone.eval()
    
    # 3. Instantiate ModulationNeck
    modulation_neck = TargetModulationNeck(in_channels=hidden_dim)

    # 4. Instantiate TextAttentionNeck
    h, w = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE, cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
    text_attention_neck = TextAttentionNeck(
        text_dim=text_backbone.output_dim,
        embed_dim=cfg.MODEL.FUSION.EMBED_DIM,
        num_heads=cfg.MODEL.FUSION.NUM_HEADS,
        output_size=(h, w)
    )

    # --- 开始添加修复代码 ---
    # 将新建的 text_attention_neck 模块的权重和偏置转换为 bfloat16 类型，
    # 以匹配 text_backbone 的输出类型。
    text_attention_neck.to(torch.bfloat16)
    # --- 结束添加 ---
    
    # 5. Build Head
    box_head = build_box_head(cfg, hidden_dim)
    
    # 6. Assemble the final model
    model = SymTrack(
        generic_backbone=backbone,
        text_backbone=text_backbone,
        modulation_neck=modulation_neck,
        text_attention_neck=text_attention_neck,
        box_head=box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )    
    
    if hasattr(backbone, 'num_patches_z'):
         model.num_patches_z = backbone.num_patches_z
    else:
        template_size = cfg.DATA.TEMPLATE.SIZE
        stride = cfg.MODEL.BACKBONE.STRIDE
        template_num = cfg.DATA.TEMPLATE.NUMBER
        model.num_patches_z = (template_size // stride)**2 * template_num

    return model
