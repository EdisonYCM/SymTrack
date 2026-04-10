import math
import numpy as np
import torch
import torch.nn.functional as F
from lib.models.symtrack import build_symtrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import cv2

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1 # Can be tuned

    def predict(self):
        # [4, 1] 
        return self.kf.predict()

    def correct(self, measurement):
        measurement_col = np.array([[measurement[0]], [measurement[1]]], np.float32)
        return self.kf.correct(measurement_col)

    def init(self, measurement):
        self.kf.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)




class SymTrack(BaseTracker):
    def __init__(self, params):
        super(SymTrack, self).__init__(params)
        network = build_symtrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                import os
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_patch_arr = None
        self.memory_frames = []
        self.memory_masks = []

        self.use_gated_ms = self.cfg.TEST.get('USE_GATED_MS', False)
        self.uncertainty_threshold = self.cfg.TEST.get('UNCERTAINTY_THRESHOLD', 0.98)
        self.scale_factors = self.cfg.TEST.get('SCALE_FACTORS', [1.0])
        if self.use_gated_ms:
            print(f"Gated Multi-scale enabled. Factors={self.scale_factors}, Threshold={self.uncertainty_threshold}")
        self.use_kalman = self.cfg.TEST.get('USE_KALMAN', False)
        if self.use_kalman:
            self.kalman_filter = KalmanFilter()
            self.kalman_alpha = self.cfg.TEST.get('KALMAN_ALPHA', 0.5) # Fusion weight
            print(f"Kalman Filter enabled. Fusion alpha={self.kalman_alpha}")
            
        

    def initialize(self, image, info: dict):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'], self.params.template_factor,
            output_sz=self.params.template_size
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.memory_frames = [template.tensors]
        self.template_tokenfd_list = []
        for i in range(z_patch_arr.shape[0]):
            template_img_np = z_patch_arr[i]
            if template_img_np.ndim == 2:
                template_img_np = cv2.cvtColor(template_img_np, cv2.COLOR_GRAY2BGR)
            z_patch_arr_tokenfd, _, _ = sample_target(
                template_img_np,
                [0, 0, template_img_np.shape[1], template_img_np.shape[0]], # BBox in patch is the patch itself
                1.0, # Factor is 1.0
                output_sz=self.cfg.DATA.TOKENFD_SIZE
            )

            if z_amask_arr.ndim > 2:
                single_amask_np = z_amask_arr[i]
            else:
                single_amask_np = z_amask_arr
            
            template_tokenfd = self.preprocessor.process(
                z_patch_arr_tokenfd, single_amask_np,
            )
            self.template_tokenfd_list.append(template_tokenfd.tensors)

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

        self.state = info['init_bbox']
        self.frame_id = 0

        if self.use_kalman:
            cx, cy = self.state[0] + 0.5*self.state[2], self.state[1] + 0.5*self.state[3]
            self.kalman_filter.init(np.array([cx, cy], np.float32))

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save,
                    "text_attention_mask": None,
                    "vit_attn_map": None
                }

        return {
                "text_attention_mask": None,
                "vit_attn_map": None
            }

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        prev_state = self.state

        # Motion Prediction with Kalman Filter (If Enabled)
        if self.use_kalman:
            motion_pred = self.kalman_filter.predict()
            motion_center = (motion_pred[0,0], motion_pred[1,0])
            # Use motion prediction to guide the search area center
            search_center_state = [motion_center[0] - 0.5*self.state[2], motion_center[1] - 0.5*self.state[3], self.state[2], self.state[3]]
        else:
            search_center_state = self.state

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, search_center_state, self.params.search_factor, output_sz=self.params.search_size)
        best_resize_factor = resize_factor
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # Get the search patch for the TokenFD backbone
        x_patch_arr_tokenfd, _, _ = sample_target(image, self.state, self.params.search_factor,
                                                output_sz=self.cfg.DATA.TOKENFD_SIZE) # Use the new size
        search_tokenfd = self.preprocessor.process(x_patch_arr_tokenfd, x_amask_arr) # Re-use amask
        
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()

        template_tokenfd_list = self.template_tokenfd_list
        
        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_list, 
                search=[search.tensors], 
                template_tokenfd=template_tokenfd_list,      
                search_tokenfd=[search_tokenfd.tensors],    
                ce_template_mask=box_mask_z
            )

        out_dict = out_dict[0]

        response_map = out_dict['score_map'] * self.output_window
        peak_score = torch.max(response_map).item()
        score_entropy = -torch.sum(response_map * torch.log(response_map + 1e-6)).item()
        
        is_uncertain = (peak_score < self.uncertainty_threshold)

        # If uncertain and Gated Multi-Scale is ON, run other scales
        if is_uncertain and self.use_gated_ms and len(self.scale_factors) > 1:
            best_out_dict = out_dict
            best_resize_factor = resize_factor
            
            other_scales = [s for s in self.scale_factors if s != 1.0]
            for scale in other_scales:
                x_patch_arr_s, resize_factor_s, x_amask_arr_s = sample_target(image, search_center_state, self.params.search_factor * scale, output_sz=self.params.search_size)
                search_s = self.preprocessor.process(x_patch_arr_s, x_amask_arr_s)
                # with torch.no_grad():
                #     out_dict_s = self.network.forward(template=template_list, search=[search_s.tensors], ce_template_mask=box_mask_z)[0]
                x_patch_arr_tokenfd_s, _, _ = sample_target(image, search_center_state, self.params.search_factor * scale, output_sz=self.cfg.DATA.TOKENFD_SIZE)
                search_tokenfd_s = self.preprocessor.process(x_patch_arr_tokenfd_s, x_amask_arr_s)
                
                with torch.no_grad():
                    out_dict_s = self.network.forward(
                        template=template_list, 
                        search=[search_s.tensors],
                        template_tokenfd=template_tokenfd_list,
                        search_tokenfd=[search_tokenfd_s.tensors],
                        ce_template_mask=box_mask_z
                    )[0]
                
                response_map_s = out_dict_s['score_map'] * self.output_window
                peak_score_s = torch.max(response_map_s).item()
                if peak_score_s > peak_score:
                    peak_score = peak_score_s
                    best_out_dict = out_dict_s
                    best_resize_factor = resize_factor_s
            out_dict = best_out_dict
            resize_factor = best_resize_factor

        response = out_dict['score_map'] * self.output_window
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_box = (pred_boxes.view(-1, 4).mean(dim=0) * self.params.search_size / resize_factor).tolist()
        visual_pred_box = self.map_box_back(pred_box, resize_factor, search_center_state)
        
        # Fusion with Kalman Filter (If Enabled)
        if self.use_kalman:
            visual_center = np.array([visual_pred_box[0] + 0.5*visual_pred_box[2], visual_pred_box[1] + 0.5*visual_pred_box[3]], np.float32)
            
            # Perform Kalman update
            self.kalman_filter.correct(visual_center)
            
            # Fuse the visual center and the Kalman-updated center
            kf_center = (self.kalman_filter.kf.statePost[0,0], self.kalman_filter.kf.statePost[1,0])
            fused_center_x = self.kalman_alpha * kf_center[0] + (1 - self.kalman_alpha) * visual_center[0]
            fused_center_y = self.kalman_alpha * kf_center[1] + (1 - self.kalman_alpha) * visual_center[1]

            # Final box uses the fused center but visual size
            self.state = [fused_center_x - 0.5*visual_pred_box[2], fused_center_y - 0.5*visual_pred_box[3], visual_pred_box[2], visual_pred_box[3]]
        else:
            self.state = visual_pred_box
            
        self.state = clip_box(self.state, H, W, margin=10)
        
        # Calculate the search region on the original image using the state from the PREVIOUS frame
        cx_prev, cy_prev = prev_state[0] + 0.5 * prev_state[2], prev_state[1] + 0.5 * prev_state[3]
        search_area_size = self.params.search_size / best_resize_factor
        sr_x = cx_prev - 0.5 * search_area_size
        sr_y = cy_prev - 0.5 * search_area_size
        search_region_bbox = [sr_x, sr_y, search_area_size, search_area_size]

        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                  output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
        self.memory_frames.append(frame)

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))

        return {
                "target_bbox": self.state,
                "score_map": out_dict['score_map'],
                "search_region_bbox": search_region_bbox,
                "size_map": out_dict['size_map'],
                "offset_map": out_dict['offset_map'],
                'text_attention_mask': out_dict.get('text_attention_mask'),
                "vit_attn_map": out_dict.get('attn')
                }

    def select_memory_frames(self):
        num_segments = int(self.cfg.TEST.TEMPLATE_NUMBER)
        if len(self.memory_frames) <= num_segments:
            indexes = np.arange(len(self.memory_frames))
        else:
            cur_frame_idx = self.frame_id
            if num_segments > 1:
                dur = max(cur_frame_idx // num_segments, 1)
                indexes = np.concatenate([np.array([0]),
                                          np.array(list(range(num_segments - 1))) * dur + dur // 2 + 1])
            else:
                indexes = np.array([0])
            indexes = np.unique(indexes).astype(int)
            indexes = np.clip(indexes, 0, len(self.memory_frames) - 1)

        select_frames = [self.memory_frames[i].cuda() for i in indexes]
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            select_masks_list = [self.memory_masks[i].cuda() for i in indexes]
            return select_frames, torch.cat(select_masks_list, dim=1)
        else:
            return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float, search_center_state: list):
        cx_prev, cy_prev = search_center_state[0] + 0.5 * search_center_state[2], search_center_state[1] + 0.5 * search_center_state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def get_tracker_class():
    return SymTrack
