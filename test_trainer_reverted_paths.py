import os
import sys
import torch
import shutil

def setup_env_and_run_test():
    # Determine project root and aot_plus directory robustly
    script_path_abs = os.path.abspath(__file__) # e.g. /app/test_trainer_reverted_paths.py
    project_root = os.path.dirname(script_path_abs) # e.g. /app

    aot_plus_actual_dir = os.path.join(project_root, "aot_plus") # e.g. /app/aot_plus

    if not os.path.isdir(aot_plus_actual_dir):
        print(f"Error: aot_plus directory not found or not a directory at {aot_plus_actual_dir}")
        sys.exit(1)

    # 1. Set CWD for the script's main logic
    os.chdir(aot_plus_actual_dir)
    print(f"Changed CWD to: {os.getcwd()}")

    # 2. Paths and Imports
    if "." not in sys.path: # Ensure aot_plus modules can be imported
        sys.path.insert(0, ".")

    from networks.managers.trainer import Trainer
    from tools.get_config import get_config

    # 3. Basic Configuration
    exp_name = "finetune_extracted_notebook_test_reverted"
    model_name_str = "r50_aotl"
    stage_str = "default"
    dataset_name_key = "EXTRACTED_FRAMES" # Key in cfg.DATASETS and cfg.DATASET_CONFIGS

    intended_gpu_id = 0
    enable_amp = True

    if not torch.cuda.is_available():
        print("CUDA not available. Forcing CPU mode (rank will be -1 for Trainer).")
        actual_gpu_id_for_trainer_rank = -1
    else:
        actual_gpu_id_for_trainer_rank = intended_gpu_id
        print(f"CUDA is available. Intended GPU ID for Trainer rank: {intended_gpu_id}")

    # 4. Load Main Configuration
    print(f"Loading config: stage='{stage_str}', exp_name='{exp_name}', model='{model_name_str}'")
    cfg = get_config(stage=stage_str, exp_name=exp_name, model=model_name_str)
    if cfg is None:
        print("Failed to load config.")
        sys.exit(1)

    # 5. Apply Notebook/Test Specific CFG Overrides
    cfg.DATASETS = [dataset_name_key]

    if dataset_name_key not in cfg.DATASET_CONFIGS:
        print(f"Error: {dataset_name_key} not found in cfg.DATASET_CONFIGS from default config.")
        sys.exit(1)

    cfg.TRAIN_GPUS = 1
    cfg.DIST_ENABLE = False
    cfg.PRETRAIN_MODEL = ""
    cfg.PRETRAIN = True
    cfg.PRETRAIN_FULL = True
    cfg.TRAIN_TOTAL_STEPS = 10

    # Logging directories relative to project_root (/app)
    test_log_dir_root = os.path.join(project_root, "test_logs_reverted_executor")
    cfg.DIR_ROOT = test_log_dir_root # Override base directory for results

    # init_dir() in DefaultEngineConfig constructs paths relative to its DIR_ROOT.
    # Call it again if we changed DIR_ROOT, or manually set all required DIR_* paths.
    # cfg.init_dir() # If this method exists and is safe to call multiple times or after overrides.
    # For safety, set them manually based on the new DIR_ROOT.
    # Ensure EXP_NAME and STAGE_NAME are part of the path construction as in default.py
    cfg.DIR_RESULT = os.path.join(cfg.DIR_ROOT, cfg.EXP_NAME, cfg.STAGE_NAME)
    cfg.DIR_LOG = os.path.join(cfg.DIR_RESULT, 'log')
    cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ckpt')
    # ... and any other DIR_* paths the Trainer might use from cfg, like DIR_EMA_CKPT, DIR_TB_LOG etc.
    cfg.DIR_EMA_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
    cfg.DIR_TB_LOG = os.path.join(cfg.DIR_RESULT, 'log', 'tensorboard')
    cfg.DIR_IMG_LOG = os.path.join(cfg.DIR_RESULT, 'log', 'img')
    cfg.DIR_EVALUATION = os.path.join(cfg.DIR_RESULT, 'eval')
    cfg.DIR_TEST = os.path.join(cfg.DIR_RESULT, 'test')


    print(f"Log directory set to: {cfg.DIR_LOG}")
    os.makedirs(cfg.DIR_LOG, exist_ok=True)
    os.makedirs(cfg.DIR_CKPT, exist_ok=True)
    os.makedirs(cfg.DIR_EMA_CKPT, exist_ok=True)
    os.makedirs(cfg.DIR_TB_LOG, exist_ok=True)
    os.makedirs(cfg.DIR_IMG_LOG, exist_ok=True)
    os.makedirs(cfg.DIR_EVALUATION, exist_ok=True)
    os.makedirs(cfg.DIR_TEST, exist_ok=True)


    # Create dummy data in project_root/extracted_frames/ (e.g., /app/extracted_frames/)
    # This is where '../extracted_frames/' from CWD=/app/aot_plus will point.
    dummy_data_abs_dir = os.path.join(project_root, "extracted_frames")
    os.makedirs(dummy_data_abs_dir, exist_ok=True)
    print(f"Ensuring dummy data directory exists: {dummy_data_abs_dir}")

    sample_img_path = os.path.join(dummy_data_abs_dir, "reverted_sample1.jpg")
    sample_json_path = os.path.join(dummy_data_abs_dir, "reverted_sample1.json")

    if not os.path.exists(sample_img_path):
        with open(sample_img_path, "w") as f: f.write("dummy image data")
        print(f"Created dummy file: {sample_img_path}")
    if not os.path.exists(sample_json_path):
        with open(sample_json_path, "w") as f: f.write('{"imageHeight": 100, "imageWidth": 100, "shapes": [{"label": "obj1", "points": [[10,10],[20,20],[10,20]]}]}')
        print(f"Created dummy file: {sample_json_path}")

    print(f"Trainer will be initialized with rank: {actual_gpu_id_for_trainer_rank}")

    try:
        print("Attempting to initialize Trainer...")
        # rank parameter for Trainer is used as self.gpu = rank + cfg.DIST_START_GPU.
        # Patched code relies on self.gpu == -1 and torch.cuda.is_available() being false for CPU mode.
        trainer = Trainer(rank=actual_gpu_id_for_trainer_rank, cfg=cfg, enable_amp=enable_amp)
        print("Trainer initialized successfully! Dataset should be loaded.")

        if trainer.train_loader and hasattr(trainer.train_loader, 'dataset'):
            dataset_len = len(trainer.train_loader.dataset)
            print(f"Length of training dataset: {dataset_len}")
            if dataset_len == 0:
                print("Error: Training dataset is empty!")
                sys.exit(1)
        else:
            print("Error: trainer.train_loader or trainer.train_loader.dataset is None or not found.")
            sys.exit(1)

    except ValueError as ve:
        print(f"ValueError during Trainer initialization or dataset loading: {ve}")
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
        if os.path.exists(sample_img_path): os.remove(sample_img_path)
        if os.path.exists(sample_json_path): os.remove(sample_json_path)
        try:
            if os.path.exists(dummy_data_abs_dir) and not os.listdir(dummy_data_abs_dir):
                os.rmdir(dummy_data_abs_dir)
                print(f"Cleaned up dummy data directory: {dummy_data_abs_dir}")
        except OSError as e:
            print(f"Note: Could not remove dummy data directory {dummy_data_abs_dir}: {e}")

        # Clean up test log directory structure
        # if os.path.exists(test_log_dir_root):
        #     print(f"Cleaning up test log directory: {test_log_dir_root}")
        #     shutil.rmtree(test_log_dir_root) # Use with caution

        # Restore original CWD (though script exits, good practice if it were a function)
        os.chdir(project_root)
        # print(f"Restored CWD to: {os.getcwd()}") # This won't print as sys.exit might be called

if __name__ == "__main__":
    setup_env_and_run_test()
