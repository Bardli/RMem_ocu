import os
import sys
import torch
import shutil

def setup_env_and_run_test():
    # Determine project root and aot_plus directory robustly
    script_path_abs = os.path.abspath(__file__)
    project_root = os.path.dirname(script_path_abs)

    aot_plus_actual_dir = os.path.join(project_root, "aot_plus")

    if not os.path.isdir(aot_plus_actual_dir):
        print(f"Error: aot_plus directory not found or not a directory at {aot_plus_actual_dir}")
        sys.exit(1)

    # 1. Set CWD for the script's main logic
    os.chdir(aot_plus_actual_dir)
    print(f"Changed CWD to: {os.getcwd()}")

    # 2. Paths and Imports
    if "." not in sys.path:
        sys.path.insert(0, ".")

    from networks.managers.trainer import Trainer
    from tools.get_config import get_config

    # 3. Create Dummy Data for ExtractedFramesTrain
    dummy_data_dir = os.path.join(project_root, "extracted_frames")
    os.makedirs(dummy_data_dir, exist_ok=True)
    print(f"Ensuring dummy data directory for ExtractedFramesTrain exists: {dummy_data_dir}")

    dummy_img_path = os.path.join(dummy_data_dir, "test_frame.jpg")
    dummy_json_path = os.path.join(dummy_data_dir, "test_frame.json")

    with open(dummy_img_path, "w") as f:
        f.write("dummy image content")
    # print(f"Created dummy image: {dummy_img_path}") # Reduce noise

    json_content = {
        "imagePath": "test_frame.jpg",
        "imageHeight": 10,
        "imageWidth": 10,
        "shapes": [{"label": "obj1", "points": [[0,0], [0,5], [5,5], [5,0]]}]
    }
    with open(dummy_json_path, "w") as f:
        import json
        json.dump(json_content, f)
    # print(f"Created dummy JSON: {dummy_json_path}") # Reduce noise

    # 4. Basic Configuration
    exp_name = "test_set_epoch_conditional"
    model_name_str = "r50_aotl"
    stage_str = "default"
    dataset_name_key = "EXTRACTED_FRAMES"
    enable_amp = False

    # 7. CPU Adaptation
    if not torch.cuda.is_available():
        print("CUDA not available. Trainer rank will be -1 for CPU operations.")
        current_rank_for_trainer = -1
    else:
        print("CUDA is available. Trainer rank will be 0.")
        current_rank_for_trainer = 0

    # 5. Load Main Configuration
    # print(f"Loading config: stage='{stage_str}', exp_name='{exp_name}', model='{model_name_str}'") # Reduce noise
    cfg = get_config(stage=stage_str, exp_name=exp_name, model=model_name_str)
    if cfg is None:
        print("Failed to load config.")
        sys.exit(1)

    # 6. Apply Test-Specific CFG Overrides
    cfg.DATASETS = [dataset_name_key]
    cfg.TRAIN_GPUS = 1
    cfg.DIST_ENABLE = False  # CRITICAL: This makes train_sampler in Trainer = None

    cfg.PRETRAIN_MODEL = ""
    cfg.MODEL_ENCODER_PRETRAIN = ""
    cfg.PRETRAIN = False

    cfg.TRAIN_TOTAL_STEPS = 1 # Try with 1 step
    cfg.TRAIN_TBLOG = False
    cfg.TRAIN_IMG_LOG = False # Disable image logging too

    cfg.DATA_WORKERS = 0
    cfg.TRAIN_BATCH_SIZE = 1

    # Configure logging paths
    test_log_dir_root = os.path.join(project_root, "test_logs_set_epoch_executor")
    cfg.DIR_ROOT = test_log_dir_root

    cfg.DIR_RESULT = os.path.join(cfg.DIR_ROOT, cfg.EXP_NAME, cfg.STAGE_NAME)
    cfg.DIR_LOG = os.path.join(cfg.DIR_RESULT, 'log')
    cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ckpt')
    cfg.DIR_EMA_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
    cfg.DIR_TB_LOG = os.path.join(cfg.DIR_RESULT, 'log', 'tensorboard')
    cfg.DIR_IMG_LOG = os.path.join(cfg.DIR_RESULT, 'log', 'img')

    for dir_path in [cfg.DIR_RESULT, cfg.DIR_LOG, cfg.DIR_CKPT, cfg.DIR_EMA_CKPT, cfg.DIR_TB_LOG, cfg.DIR_IMG_LOG]:
        os.makedirs(dir_path, exist_ok=True)

    # print(f"Log directory: {cfg.DIR_LOG}") # Reduce noise
    # print(f"DIST_ENABLE set to: {cfg.DIST_ENABLE}") # Reduce noise
    # print(f"DATA_WORKERS set to: {cfg.DATA_WORKERS}") # Reduce noise
    # print(f"TRAIN_BATCH_SIZE set to: {cfg.TRAIN_BATCH_SIZE}") # Reduce noise

    # 8. Instantiate and Run Trainer
    trainer = None
    try:
        # print("Attempting to initialize Trainer...") # Reduce noise
        trainer = Trainer(rank=current_rank_for_trainer, cfg=cfg, enable_amp=enable_amp)
        print("Trainer initialized successfully.")

        print("Calling sequential_training (for 1 step)...")
        trainer.sequential_training()
        print("sequential_training call completed without AttributeError related to set_epoch.")

    except AttributeError as ae:
        print(f"AttributeError encountered: {ae}")
        if 'NoneType' in str(ae) and 'set_epoch' in str(ae):
            print("Test FAILED: The specific AttributeError for train_sampler.set_epoch was raised.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up dummy data
        if os.path.exists(dummy_img_path): os.remove(dummy_img_path)
        if os.path.exists(dummy_json_path): os.remove(dummy_json_path)
        try:
            if os.path.exists(dummy_data_dir) and not os.listdir(dummy_data_dir):
                os.rmdir(dummy_data_dir)
                # print(f"Cleaned up dummy data directory: {dummy_data_dir}") # Reduce noise
        except OSError as e:
            print(f"Note: Could not remove dummy data directory {dummy_data_dir}: {e}")

        os.chdir(project_root)

if __name__ == "__main__":
    setup_env_and_run_test()
