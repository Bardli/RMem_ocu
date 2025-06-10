# Cell 6: Evaluation Metric Calculation
# Uses functions from the project's 'evaluation' module.
import importlib # For potential re-import if needed
import numpy as np # Ensure numpy is imported
import os # Ensure os is imported
import sys # Ensure sys is imported

# We need to import the metric calculation utilities.
# Adjust path if necessary to import from evaluation.source
# This assumes the notebook is run from 'aot_plus/' or its parent where 'evaluation' is a sibling or child.
metric_utils_imported = False
try:
    current_proj_root = project_root # Defined in Cell 1
    eval_source_path_option1 = os.path.join(current_proj_root, 'evaluation', 'source')
    eval_source_path_option2 = os.path.join(current_proj_root, 'aot_plus', 'evaluation', 'source')
    eval_source_path_option3 = os.path.join(current_proj_root, '..', 'evaluation', 'source')

    # Determine the correct path to add for 'evaluation.source'
    # This logic assumes 'evaluation' is a sibling to 'aot_plus', or 'aot_plus' is the project root.
    # If 'aot_plus' is the CWD:
    if os.path.basename(os.getcwd()) == 'aot_plus':
        # Check if 'evaluation' is a sibling to 'aot_plus'
        if os.path.isdir(os.path.join(os.getcwd(), '..', 'evaluation', 'source')):
            if os.path.join(os.getcwd(), '..') not in sys.path:
                 sys.path.insert(0, os.path.join(os.getcwd(), '..'))
        # Check if 'evaluation' is a subdirectory of 'aot_plus' (less common for this structure)
        elif os.path.isdir(os.path.join(os.getcwd(), 'evaluation', 'source')):
            if os.getcwd() not in sys.path: # Should already be there
                 sys.path.insert(0, os.getcwd())
    # If CWD is parent of 'aot_plus' (project_root)
    elif os.path.isdir(os.path.join(os.getcwd(), 'evaluation', 'source')):
         if os.getcwd() not in sys.path: # Should already be there
            sys.path.insert(0, os.getcwd())

    from evaluation.source.metrics import db_eval_iou
    from evaluation.source import utils as eval_utils
    print("Successfully imported metric utilities from evaluation.source")
    metric_utils_imported = True
except ImportError as e:
    print(f"Error importing metric utilities: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Please ensure 'evaluation.source' is in sys.path and has an __init__.py if needed.")
    print("Falling back to dummy metric functions.")
    def db_eval_iou(gt_mask_obj, pred_mask_obj, *args, **kwargs):
        print("Dummy db_eval_iou: called because import failed.")
        if gt_mask_obj.ndim == 3: return np.zeros(gt_mask_obj.shape[0])
        elif gt_mask_obj.ndim == 4: return np.zeros((gt_mask_obj.shape[0], gt_mask_obj.shape[1]))
        return np.array([0.0])

    class DummyEvalUtils:
        def db_statistics(self, per_frame_iou_obj):
            print("Dummy db_statistics: called because import failed.")
            return np.array([0.0]), np.array([0.0]), np.array([0.0]) # M, R, D
    eval_utils = DummyEvalUtils()

all_j_means = []
all_j_recalls = []
all_j_decays = []

if not metric_utils_imported:
    print("Skipping metric calculation as utilities could not be imported.")
else:
    print(f"Calculating metrics for {len(all_pred_masks_for_metrics)} sequences...")
    for seq_idx in range(len(all_pred_masks_for_metrics)):
        pred_masks_seq_np = all_pred_masks_for_metrics[seq_idx]
        gt_masks_seq_np = all_gt_masks_for_metrics[seq_idx]

        seq_name_for_log = example_sequence_data[seq_idx]['seq_name'] if seq_idx < len(example_sequence_data) else f"UnknownSeq{seq_idx}"

        if pred_masks_seq_np.shape != gt_masks_seq_np.shape:
            print(f"WARNING: Shape mismatch for sequence {seq_name_for_log}. GT: {gt_masks_seq_np.shape}, Pred: {pred_masks_seq_np.shape}. Skipping.")
            continue

        if pred_masks_seq_np.ndim == 2:
            pred_masks_seq_np = np.expand_dims(pred_masks_seq_np, axis=0)
        if gt_masks_seq_np.ndim == 2:
            gt_masks_seq_np = np.expand_dims(gt_masks_seq_np, axis=0)

        if pred_masks_seq_np.size == 0 or gt_masks_seq_np.size == 0 :
            print(f"WARNING: Empty masks for sequence {seq_name_for_log}. Skipping.")
            continue

        unique_gt_obj_ids = np.unique(gt_masks_seq_np)
        active_obj_ids = sorted([obj_id for obj_id in unique_gt_obj_ids if obj_id > 0])

        if not active_obj_ids:
            print(f"Sequence {seq_name_for_log}: No foreground objects in GT. If predictions also show no objects, IoU is 1 for background. If predictions show objects, IoU is 0.")
            # Handle case where GT is all background. db_eval_iou might need specific handling or interpretation.
            # For now, if there are no active objects, we might consider this sequence as having perfect background prediction if pred is also all background.
            # This specific scenario's desired metric outcome needs clarification. Let's assume for now we only score on active_obj_ids.
            if np.sum(pred_masks_seq_np > 0) == 0: # Pred is all background
                 all_j_means.append(1.0) # Perfect background match
                 all_j_recalls.append(1.0)
                 all_j_decays.append(0.0)
            else: # Pred has foreground, GT does not
                 all_j_means.append(0.0)
                 all_j_recalls.append(0.0)
                 all_j_decays.append(1.0) # Max decay
            continue

        gt_obj_masks_list = []
        pred_obj_masks_list = []
        for obj_id in active_obj_ids:
            gt_obj_masks_list.append((gt_masks_seq_np == obj_id).astype(np.uint8))
            pred_obj_masks_list.append((pred_masks_seq_np == obj_id).astype(np.uint8))

        if not gt_obj_masks_list: # Should not happen if active_obj_ids is not empty
            continue

        gt_for_eval = np.stack(gt_obj_masks_list, axis=0)
        pred_for_eval = np.stack(pred_obj_masks_list, axis=0)

        try:
            j_metrics_res_seq = db_eval_iou(gt_for_eval, pred_for_eval, None)
        except Exception as e:
            print(f"Error in db_eval_iou for sequence {seq_name_for_log}: {e}. Skipping sequence.")
            continue

        if j_metrics_res_seq is not None and j_metrics_res_seq.size > 0:
            for obj_idx_in_eval in range(j_metrics_res_seq.shape[0]):
                obj_j_scores = j_metrics_res_seq[obj_idx_in_eval]
                if obj_j_scores.size > 0:
                    stats = eval_utils.db_statistics(obj_j_scores)
                    jm, jr_values, jd = stats[0], stats[1], stats[2]
                    jr = jr_values[0] if isinstance(jr_values, (list, np.ndarray)) and len(jr_values)>0 else jr_values
                    all_j_means.append(jm)
                    all_j_recalls.append(jr)
                    all_j_decays.append(jd)

if all_j_means:
    overall_j_mean = np.nanmean(all_j_means)
    overall_j_recall = np.nanmean(all_j_recalls)
    overall_j_decay = np.nanmean(all_j_decays)
    print("\n--- Overall Evaluation Metrics (averaged over objects & sequences) ---")
    print(f"Jaccard Mean (IoU): {overall_j_mean:.4f}")
    print(f"Jaccard Recall (Proportion of frames with IoU > 0.5 for detected objects): {overall_j_recall:.4f}")
    print(f"Jaccard Decay: {overall_j_decay:.4f}")
else:
    print("No sequences/objects were successfully evaluated to calculate overall metrics.")
