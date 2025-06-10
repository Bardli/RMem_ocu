# Cell 7: Video Generation with Mask Overlays
import cv2 # Ensure cv2 is imported
import numpy as np # Ensure numpy is imported
import os # Ensure os is imported

# `example_sequence_data` was populated in Cell 5 with:
# [{'seq_name': name, 'frames': [list of np_frames_chw_norm (C,H,W)],
#   'pred_masks': [list of np_masks_hw_pred (H,W)],
#   'gt_masks': [list of np_masks_hw_gt (H,W)]}, ...]

# Define a color palette (BGR for OpenCV)
palette = np.array([
    [0, 0, 0],      # 0: Background (Black)
    [0, 0, 255],    # 1: Object 1 (Red)
    [0, 255, 0],    # 2: Object 2 (Green)
    [255, 0, 0],    # 3: Object 3 (Blue)
    [0, 255, 255],  # 4: Object 4 (Yellow)
    [255, 0, 255],  # 5: Object 5 (Magenta)
    [255, 255, 0],  # 6: Object 6 (Cyan)
    [128, 0, 0],    # 7: Maroon
    [0, 128, 0],    # 8: Dark Green
    [0, 0, 128],    # 9: Navy
    [128, 128, 0],  # 10: Olive
    [128, 0, 128],  # 11: Purple
    [0, 128, 128],  # 12: Teal
], dtype=np.uint8)

def overlay_mask_on_frame(frame_chw_norm, mask_hw, alpha=0.5, palette=palette):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    frame_chw = (frame_chw_norm.astype(np.float32) * std + mean) * 255.0
    frame_chw = np.clip(frame_chw, 0, 255).astype(np.uint8)

    frame_hwc_rgb = frame_chw.transpose(1, 2, 0)
    frame_hwc_rgb_contiguous = np.ascontiguousarray(frame_hwc_rgb)
    frame_hwc_bgr = cv2.cvtColor(frame_hwc_rgb_contiguous, cv2.COLOR_RGB2BGR)

    color_overlay = np.zeros_like(frame_hwc_bgr, dtype=np.uint8)
    unique_obj_ids = np.unique(mask_hw)

    for obj_id_val in unique_obj_ids:
        obj_id = int(obj_id_val)
        if obj_id == 0:
            continue

        color_idx = obj_id % len(palette)
        if color_idx == 0 and obj_id != 0 :
             color_idx = 1 + (obj_id // len(palette)) % (len(palette)-1) if len(palette) > 1 else 1

        color = palette[color_idx if color_idx < len(palette) else 1]
        color_overlay[mask_hw == obj_id] = color

    frame_hwc_bgr_contiguous = np.ascontiguousarray(frame_hwc_bgr)
    color_overlay_contiguous = np.ascontiguousarray(color_overlay)

    try:
        overlaid_frame = cv2.addWeighted(frame_hwc_bgr_contiguous, 1.0, color_overlay_contiguous, alpha, 0)
    except cv2.error as e:
        print(f"cv2.addWeighted error: {e}. Using original frame.")
        return frame_hwc_bgr_contiguous

    return overlaid_frame

# Ensure OUTPUT_VIDEO_DIR and MAX_EXAMPLE_SEQUENCES_FOR_VIDEO are defined (e.g., from Cell 1)
if 'OUTPUT_VIDEO_DIR' not in locals(): OUTPUT_VIDEO_DIR = './evaluation_videos_default'
if 'MAX_EXAMPLE_SEQUENCES_FOR_VIDEO' not in locals(): MAX_EXAMPLE_SEQUENCES_FOR_VIDEO = 1
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)


print(f"Generating videos for up to {MAX_EXAMPLE_SEQUENCES_FOR_VIDEO} example sequences...")
if 'example_sequence_data' not in locals() or not example_sequence_data:
    print("Warning: `example_sequence_data` is not defined or empty. Skipping video generation.")
else:
    for i, seq_data in enumerate(example_sequence_data):
        if i >= MAX_EXAMPLE_SEQUENCES_FOR_VIDEO:
            print(f"Reached max number of example videos ({MAX_EXAMPLE_SEQUENCES_FOR_VIDEO}).")
            break

        seq_name = seq_data['seq_name']
        safe_seq_name = "".join(c if c.isalnum() else "_" for c in seq_name)
        video_filename_pred = os.path.join(OUTPUT_VIDEO_DIR, f"{safe_seq_name}_pred_overlay.mp4")
        video_filename_gt = os.path.join(OUTPUT_VIDEO_DIR, f"{safe_seq_name}_gt_overlay.mp4")

        first_frame_chw_norm = seq_data['frames'][0].astype(np.float32)
        mean_disp = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std_disp = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        temp_frame_chw_disp = (first_frame_chw_norm * std_disp + mean_disp) * 255.0
        height, width = temp_frame_chw_disp.shape[1], temp_frame_chw_disp.shape[2]

        video_writers = {}
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writers['pred'] = cv2.VideoWriter(video_filename_pred, fourcc, 10.0, (width, height))
            video_writers['gt'] = cv2.VideoWriter(video_filename_gt, fourcc, 10.0, (width, height))

            if not video_writers['pred'].isOpened(): raise IOError(f"Cannot open video writer for {video_filename_pred}")
            if not video_writers['gt'].isOpened(): raise IOError(f"Cannot open video writer for {video_filename_gt}")

            for frame_idx in range(len(seq_data['frames'])):
                frame_chw_norm = seq_data['frames'][frame_idx]
                pred_mask_hw = seq_data['pred_masks'][frame_idx]
                gt_mask_hw = seq_data['gt_masks'][frame_idx]

                overlaid_frame_pred = overlay_mask_on_frame(frame_chw_norm, pred_mask_hw, alpha=0.6)
                overlaid_frame_gt = overlay_mask_on_frame(frame_chw_norm, gt_mask_hw, alpha=0.6)

                video_writers['pred'].write(overlaid_frame_pred)
                video_writers['gt'].write(overlaid_frame_gt)

            print(f"Saved prediction video: {video_filename_pred}")
            print(f"Saved ground truth video: {video_filename_gt}")

        except Exception as e:
            print(f"Error creating video for sequence {seq_name}: {e}")
        finally:
            if 'pred' in video_writers and video_writers['pred'].isOpened(): video_writers['pred'].release()
            if 'gt' in video_writers and video_writers['gt'].isOpened(): video_writers['gt'].release()
print("Video generation finished.")
