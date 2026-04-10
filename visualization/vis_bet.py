import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import pickle
import cv2
import torch
import numpy as np
import math
from tqdm import tqdm
import yaml 
from lib.test.evaluation.environment import env_settings
from lib.test.utils.load_text import load_text
from lib.test.evaluation import get_dataset


def get_plot_color(is_gt):
    return (0, 255, 0) if is_gt else (0, 0, 255) # Green for GT, Red for Prediction


def load_config(tracker_name, tracker_param):
    settings = env_settings()
    config_path = os.path.join(settings.prj_dir, 'experiments', tracker_name, f'{tracker_param}.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default values.")
        return {
            'DATA': {'SEARCH': {'SIZE': 384}, 'TEMPLATE': {'SIZE': 192}}, 
            'MODEL': {'BACKBONE': {'STRIDE': 16, 'ADD_CLS_TOKEN': True}}
        }
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_vit_attention(attn_tensor, cfg):
    if attn_tensor is None or not isinstance(attn_tensor, (torch.Tensor, np.ndarray)):
        return None
    
    if isinstance(attn_tensor, np.ndarray):
        attn_tensor = torch.from_numpy(attn_tensor)

    if attn_tensor.dim() != 4:
        print(f"Warning: Expected 4D attention tensor, but got {attn_tensor.dim()}D. Shape: {attn_tensor.shape}")
        return None

    # (B, Num_Heads, L, L)
    B, H, L, _ = attn_tensor.shape
    
    L_x = 196
    feat_sz = 14

    start_idx = 1
    end_idx = 1 + L_x
    
    if end_idx > L:
        return None

    try:
        attn_cls_to_x = attn_tensor[0, :, 0, start_idx:end_idx] # Shape: (Heads, 196)
        
        attn_2d_flat = attn_cls_to_x.mean(dim=0)
        
        attn_map_2d = attn_2d_flat.reshape(feat_sz, feat_sz)
        
        return attn_map_2d
    except Exception as e:
        print(f"Error processing ViT attention (L={L}, L_x={L_x}, feat_sz={feat_sz}): {e}")
        if 'attn_2d_flat' in locals():
             print(f"Debug: attn_2d_flat shape: {attn_2d_flat.shape}")
        return None


def create_detailed_visualization(image, map_data, search_region_box, map_type, cfg):
    if search_region_box is None:
        return None, None 

    H, W, _ = image.shape
    sr_x, sr_y, sr_w, sr_h = [int(v) for v in search_region_box]

    processed_map_2d = None
    if map_type == 'vit_attention':
        processed_map_2d = process_vit_attention(map_data, cfg) 
    elif map_data is not None and isinstance(map_data, (torch.Tensor, np.ndarray)):
        if map_data.dim() > 2:
            processed_map_2d = map_data.squeeze()
        elif map_data.dim() == 2:
            processed_map_2d = map_data

    blended_full_image = image.copy() 
    
    if processed_map_2d is not None and processed_map_2d.dim() == 2:
        score_map_np = processed_map_2d.cpu().float().numpy()
        
        try:
            heatmap_resized = cv2.resize(score_map_np, (sr_w, sr_h)) 
        except cv2.error as e:
            print(f"Error resizing heatmap to ({sr_w}, {sr_h}): {e}")
            return None, image 
        
        min_val, max_val = heatmap_resized.min(), heatmap_resized.max()
        if max_val - min_val > 1e-6:
            heatmap_norm = ((heatmap_resized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap_resized, dtype=np.uint8)
            
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET) 

        heatmap_overlay_full = np.zeros_like(image, dtype=np.uint8)
        
        clip_x1, clip_y1 = max(0, sr_x), max(0, sr_y)
        clip_x2, clip_y2 = min(W, sr_x + sr_w), min(H, sr_y + sr_h)

        h_x1 = clip_x1 - sr_x
        h_y1 = clip_y1 - sr_y
        h_x2 = clip_x2 - sr_x
        h_y2 = clip_y2 - sr_y

        if h_x1 >= h_x2 or h_y1 >= h_y2:
            return None, image 

        heatmap_overlay_full[clip_y1:clip_y2, clip_x1:clip_x2] = heatmap_color[h_y1:h_y2, h_x1:h_x2]

        blended_full_image = cv2.addWeighted(image, 0.6, heatmap_overlay_full, 0.4, 0)
        
    clip_x1_patch, clip_y1_patch = max(0, sr_x), max(0, sr_y)
    clip_x2_patch, clip_y2_patch = min(W, sr_x + sr_w), min(H, sr_y + sr_h)
    
    if clip_x1_patch >= clip_x2_patch or clip_y1_patch >= clip_y2_patch:
        return None, blended_full_image 
    
    image_patch = image[clip_y1_patch:clip_y2_patch, clip_x1_patch:clip_x2_patch]
    if image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
        return None, blended_full_image 

    blended_patch = blended_full_image[clip_y1_patch:clip_y2_patch, clip_x1_patch:clip_x2_patch]

    if processed_map_2d is not None and processed_map_2d.dim() == 2:
        try:
            heatmap_resized_patch = cv2.resize(score_map_np, (sr_w, sr_h))

            pure_heatmap_patch = heatmap_resized_patch[h_y1:h_y2, h_x1:h_x2]
        except cv2.error as e:
            print(f"Error resizing heatmap to patch size: {e}")
            return None, blended_full_image
            
        min_val, max_val = heatmap_resized_patch.min(), heatmap_resized_patch.max()
        if max_val - min_val > 1e-6:
            heatmap_norm_patch = ((heatmap_resized_patch - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            heatmap_norm_patch = np.zeros_like(heatmap_resized_patch, dtype=np.uint8)
        
        pure_heatmap_patch = cv2.applyColorMap(heatmap_norm_patch, cv2.COLORMAP_JET)
    else:
        pure_heatmap_patch = np.zeros_like(image_patch)
        
    h = blended_patch.shape[0]
    if pure_heatmap_patch.shape[0] != h:
        pure_heatmap_patch = cv2.resize(pure_heatmap_patch, (blended_patch.shape[1], h))
        
    composite_image = cv2.hconcat([blended_patch, pure_heatmap_patch])

    return composite_image, blended_full_image

def visualize_sequence(tracker_name, tracker_param, dataset_name, sequence_name, map_type):
    print(f"Starting robust visualization for map type: {map_type}...")
    settings = env_settings()
    dataset = get_dataset(dataset_name)
    sequence = next((s for s in dataset if s.name == sequence_name), None)
    if sequence is None: raise ValueError(f"Sequence '{sequence_name}' not found")

    cfg = load_config(tracker_name, tracker_param)
    print(f"Loaded config: SEARCH_SIZE={cfg['DATA']['SEARCH']['SIZE']}, TEMPLATE_SIZE={cfg['DATA']['TEMPLATE']['SIZE']}, STRIDE={cfg['MODEL']['BACKBONE']['STRIDE']}")

    base_results_path = os.path.join(settings.results_path, tracker_name, tracker_param, dataset_name, sequence.name)
    pred_path = f"{base_results_path}.txt"
    search_region_path = f"{base_results_path}_search_region.txt"

    if map_type == 'score_map':
        map_dir = os.path.join(base_results_path, "score_maps")
        output_sub_dir = "detailed_score_map"
    elif map_type == 'text_attention':
        map_dir = os.path.join(base_results_path, "text_attention_masks")
        output_sub_dir = "detailed_text_attention"
    elif map_type == 'vit_attention':
        map_dir = os.path.join(base_results_path, "vit_attn_maps")
        output_sub_dir = "detailed_vit_attention" 
    else:
        raise ValueError(f"Unknown map_type: {map_type}")
        
    if not os.path.exists(map_dir):
        print(f"Error: Map directory not found at {map_dir}")
        return
        
    print(f"Loading map data from directory: {map_dir}")

    output_dir_full = os.path.join(settings.prj_dir, "visualization", tracker_name, tracker_param, sequence.name, "full_frame")
    output_dir_detail = os.path.join(settings.prj_dir, "visualization", tracker_name, tracker_param, sequence.name, output_sub_dir)
    output_dir_blend_full = os.path.join(settings.prj_dir, "visualization", tracker_name, tracker_param, sequence.name, f"full_frame_blend_{map_type}")
    os.makedirs(output_dir_full, exist_ok=True)
    os.makedirs(output_dir_detail, exist_ok=True)
    os.makedirs(output_dir_blend_full, exist_ok=True)
    print(f"Saving full frames to: {output_dir_full}")
    print(f"Saving detailed views to: {output_dir_detail}")
    print(f"Saving blended full frames to: {output_dir_blend_full}")
    
    gt_boxes = torch.tensor(sequence.ground_truth_rect)
    pred_boxes = torch.tensor(load_text(str(pred_path), delimiter=('\t', ','), dtype=np.float64))
    search_region_boxes = torch.tensor(load_text(str(search_region_path), delimiter=('\t', ','), dtype=np.float64))

    for frame_idx, frame_path in enumerate(tqdm(sequence.frames, desc=f"Visualizing {sequence.name}")):
        image = cv2.imread(frame_path)
        image_clean_copy = image.copy() 
        
        search_region_box = search_region_boxes[frame_idx - 1] if frame_idx > 0 and frame_idx - 1 < len(search_region_boxes) else None
        
        current_map = None
        map_load_path = os.path.join(map_dir, f"{frame_idx:08d}.pkl")
        
        try:
            with open(map_load_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            current_map_data = None
            if isinstance(loaded_data, dict):
                key_to_visualize = 'attn_last'  # 'attn_last', 'attn_mid', 'attn_first' 
                
                if key_to_visualize in loaded_data:
                    current_map_data = loaded_data[key_to_visualize]
                else:
                    print(f"Warning: Key '{key_to_visualize}' not found in {map_load_path}. Available keys are: {list(loaded_data.keys())}")
            
            elif loaded_data is not None:
                current_map_data = loaded_data 

            if current_map_data is not None:
                if isinstance(current_map_data, torch.Tensor):
                    current_map = current_map_data
                else:
                    current_map = torch.from_numpy(current_map_data) 
            else:
                current_map = None 

        except FileNotFoundError:
            if frame_idx > 0: 
                pass 
            pass
        except Exception as e:
            print(f"Error loading or processing {map_load_path}: {e}")
            pass
        
        detailed_image, blended_full_image = create_detailed_visualization(image_clean_copy, current_map, search_region_box, map_type, cfg) 
        
        if detailed_image is not None:
            cv2.imwrite(os.path.join(output_dir_detail, f"{frame_idx:08d}.jpg"), detailed_image)
        
        if blended_full_image is not None:
            if frame_idx < len(gt_boxes) and gt_boxes[frame_idx].sum() > 0:
                x, y, w, h = gt_boxes[frame_idx]; cv2.rectangle(blended_full_image, (int(x), int(y)), (int(x+w), int(y+h)), get_plot_color(is_gt=True), 2)
            if frame_idx < len(pred_boxes):
                x, y, w, h = pred_boxes[frame_idx]; cv2.rectangle(blended_full_image, (int(x), int(y)), (int(x+w), int(y+h)), get_plot_color(is_gt=False), 2)
            
            cv2.imwrite(os.path.join(output_dir_blend_full, f"{frame_idx:08d}.jpg"), blended_full_image)
        else:
            cv2.imwrite(os.path.join(output_dir_blend_full, f"{frame_idx:08d}.jpg"), image_clean_copy)
        
        if frame_idx < len(gt_boxes) and gt_boxes[frame_idx].sum() > 0:
            x, y, w, h = gt_boxes[frame_idx]; cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), get_plot_color(is_gt=True), 2)
        
        if frame_idx < len(pred_boxes):
            x, y, w, h = pred_boxes[frame_idx]; cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), get_plot_color(is_gt=False), 2)
        
        if search_region_box is not None:
            sr_x, sr_y, sr_w, sr_h = [int(v) for v in search_region_box]
            cv2.rectangle(image, (sr_x, sr_y), (sr_x + sr_w, sr_y + sr_h), (255, 255, 255), 1)

        cv2.imwrite(os.path.join(output_dir_full, f"{frame_idx:08d}.jpg"), image)
    
    print("Visualization complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize score map heatmaps for a tracking sequence.')
    parser.add_argument('tracker_name', type=str, help='Name of the tracker (e.g., symtrack).')
    parser.add_argument('tracker_param', type=str, help='Name of the tracker parameter file (e.g., baseline_text_scalear).')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset (e.g., lasot, got10k_test).')
    parser.add_argument('sequence_name', type=str, help='Name of the sequence to visualize (e.g., car-1, person-5).')
    
    parser.add_argument('--map_type', type=str, default='score_map', 
                        choices=['score_map', 'text_attention', 'vit_attention'], 
                        help='Which map to visualize.')

    args = parser.parse_args()
    
    visualize_sequence(args.tracker_name, args.tracker_param, args.dataset_name, args.sequence_name, args.map_type)
    
    #  python visualization/vis_bet.py symtrack baseline_text_scalear artvideo_sot xx --map_type vit_attention
