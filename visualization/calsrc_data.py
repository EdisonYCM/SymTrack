import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import torch
import numpy as np
from tqdm import tqdm
import collections

from lib.test.evaluation.environment import env_settings
from lib.test.utils.load_text import load_text
from lib.test.evaluation import get_dataset

def calculate_area(box):
    w = max(0.0, box[2])
    h = max(0.0, box[3])
    return w * h

def calculate_intersection_area(box_a, box_b):
    box_a_x1, box_a_y1, box_a_x2, box_a_y2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    box_b_x1, box_b_y1, box_b_x2, box_b_y2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    inter_x1 = max(box_a_x1, box_b_x1)
    inter_y1 = max(box_a_y1, box_b_y1)
    inter_x2 = min(box_a_x2, box_b_x2)
    inter_y2 = min(box_a_y2, box_b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h

def calculate_src_for_dataset(tracker_name, dataset_name, param_list):
    print(f"Starting Search Region Coverage (SRC) calculation for the entire dataset...")
    print(f"Tracker: {tracker_name}")
    print(f"Dataset: {dataset_name}")
    
    settings = env_settings()
    dataset = get_dataset(dataset_name)

    all_src_scores = collections.OrderedDict((param, []) for param in param_list)
    
    for sequence in tqdm(dataset, desc=f"Processing Sequences in {dataset_name}"):
        
        gt_boxes = torch.tensor(sequence.ground_truth_rect)
        num_frames = len(sequence.frames)

        for param_name in param_list:
            base_results_path = os.path.join(settings.results_path, tracker_name, param_name, dataset_name, sequence.name)
            search_region_path = f"{base_results_path}_search_region.txt"

            if not os.path.exists(search_region_path):
                # print(f"Warning: File not found for sequence '{sequence.name}' with param '{param_name}'. Skipping.")
                continue
            
            try:
                search_region_boxes = torch.tensor(load_text(str(search_region_path), delimiter=('\t', ','), dtype=np.float64))
            except Exception as e:
                print(f"Error loading {search_region_path}: {e}")
                continue

            for frame_idx in range(1, num_frames):
                sr_idx = frame_idx - 1
                if sr_idx >= len(search_region_boxes):
                    break 

                gt_box = gt_boxes[frame_idx]
                search_box = search_region_boxes[sr_idx]
                
                if gt_box.sum() <= 0:
                    continue

                area_gt = calculate_area(gt_box)
                if area_gt <= 1e-6:
                    continue

                area_intersection = calculate_intersection_area(search_box, gt_box)
                src = area_intersection / area_gt
                
                all_src_scores[param_name].append(src)

    print("\n" + "="*50)
    print(f"Final SRC Results for {tracker_name} on {dataset_name}")
    print("="*50)
    
    for param_name, scores in all_src_scores.items():
        if scores:
            mean_src = np.mean(scores)
            total_frames = len(scores)
            print(f"{param_name:<30} | Mean SRC: {mean_src:.4f} (over {total_frames} frames)")
        else:
            print(f"{param_name:<30} | FAILED (No valid frames or files found for this param across the dataset)")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and Compare Search Region Coverage (SRC) for multiple tracker_params across an entire dataset.')
    
    parser.add_argument('tracker_name', type=str, help="Name of the tracker (e.g., 'symtrack').")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset to evaluate (e.g., 'lasot', 'got10k_test').")
    
    args = parser.parse_args()
    
    tracker_param_list = [
        'baseline_text_scalear',
    ]

    calculate_src_for_dataset(
        args.tracker_name, 
        args.dataset_name,
        tracker_param_list
    )
    # python visualization/calsrc_data.py symtrack artvideo_sot