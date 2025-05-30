import os
import sys
import torch

# Ensure aot_plus is in the Python path
# Assuming the script is run from the 'aot_plus' directory itself,
# the current directory (.) should be in sys.path by default.
# If not, or if running from elsewhere, this might be needed:
# module_path = os.path.abspath(os.path.join('.')) # or 'aot_plus' if script is outside
# if module_path not in sys.path:
#     sys.path.append(module_path)

from networks.managers.trainer import Trainer
from tools.get_config import get_config

def run_test():
    print(f"Current working directory: {os.getcwd()}")

    # 1. Define basic configuration parameters
    exp_name = "aotplus_test"
    model_name_str = "r50_aotl" # Changed from "aotplus" to a valid model config
    stage_str = "pre_ytb"    # Example, adjust if needed
    dataset_name = "extracted_frames_train" # Key dataset to test
    gpu_id = -1 # Changed to -1 to attempt CPU execution

    # 2. Load the configuration
    # Args typically passed via command line
    # class Args:
    #     pass
    # args = Args()
    # args.config = f"configs/{model_name_str}/{stage_str}.yaml" # This was for a different get_config
    # args.opts = []

    # Add a dummy command-line argument for dataset if get_config expects it
    # This depends on how get_config is structured. If it parses sys.argv directly,
    # we might need to simulate that. For now, let's assume direct opts are enough.
    # sys.argv = ['train.py', f'--config=configs/{model_name_str}/{stage_str}.yaml'] # Example if needed

    print(f"Loading config for stage: {stage_str}, experiment: {exp_name}, model: {model_name_str}")
    cfg = get_config(stage=stage_str, exp_name=exp_name, model=model_name_str)

    if cfg is None:
        print("Failed to load config.")
        return

    # 3. Apply notebook-specific settings to the cfg object
    cfg.EXP_NAME = exp_name
    cfg.MODEL_NAME = model_name_str
    cfg.STAGE_NAME = stage_str # Ensure this matches expected config structure

    # Dataset configuration - critical for the test
    cfg.DATASETS = {
        dataset_name: { # dataset_name is "extracted_frames_train"
            "DATASET": "ExtractedFramesTrain", # This value is used as a key for DATASET_CONFIGS
            "TYPE": "train",
            "VID_PER_GPU": 1, # Keep low for testing
            "WORKERS_PER_GPU": 1, # Keep low for testing
            "SEQ_LEN": 1, # As per ExtractedFramesTrain typical setup
            "MAX_OBJ_NUM": 10,
            "ROOT_DIR": "extracted_frames/", # Corrected path
            "JSON_ANNO_DIR": "extracted_frames/", # Corrected path
            "SPLIT": "train", # Assuming 'train' split exists or is handled
            "REPEAT_FACTOR": 1.0,
            "REPEAT_TIMES": 1,
            "MIN_AREA_RATIO": 0.001,
            "MAX_AREA_RATIO": 0.3,
            "MIN_HW_RATIO": 0.1,
            "MAX_HW_RATIO": 10.0,
            "SAMPLE_STRATEGY": "continuous",
            "ENABLE_RANDOM_CROP": True,
            "ENABLE_RANDOM_FLIP": True,
            "ENABLE_RANDOM_ROTATION": True,
            "ENABLE_RANDOM_RESIZE": True,
            "ENABLE_MULTI_SCALE": True,
            "RESIZE_SIZE_RANGE": [256, 512],
            "CROP_SIZE": [473, 473], # Example, ensure it's valid
            "MEAN": [0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],
            "IGNORE_VALUE": 255,
            "PRETRAIN_TRANSFORM_PREAUG": True,
            "TRANSFORM_PREAUG_PROB": 0.5,
            "TRANSFORM_PREAUG_FIX_RATIO": True,
            "TRANSFORM_RANDOM_CROP_RESIZE_RANGE": [0.8, 1.2],
            "TRANSFORM_RANDOM_CROP_RESIZE_RATIO_RANGE": [0.9, 1.1],
            "TRANSFORM_RANDOM_ROTATION_RANGE": [-10, 10],
            "TRANSFORM_RANDOM_FLIP_PROB": 0.5
        }
    }
    # cfg.DATASETS_TRAIN_NAME = dataset_name # "extracted_frames_train" # Incorrect: used by loader is DATASETS_TRAIN_NAMES (plural)
    cfg.DATASETS_TRAIN_NAMES = [dataset_name] # Corrected to be a list
    cfg.DATASETS_EVAL_NAMES = [] # Explicitly set eval names to empty list
    cfg.DATASETS_TRAIN_REPEAT = [1.0]
    cfg.DATASETS_TRAIN_VID_PER_GPU = [1]
    cfg.DATASETS_TRAIN_SEQ_LEN = [1]
    cfg.DATASETS_TRAIN_MAX_OBJ_NUM = [10]

    # This section defines specific configurations for dataset types.
    # The key here must match the key used in cfg.DATASETS (e.g., "extracted_frames_train").
    cfg.DATASET_CONFIGS = {
        dataset_name: { # Key is now "extracted_frames_train"
            "TYPE": "ExtractedFramesTrain", # This specifies the class name to instantiate
            "CONFIG": {
                "COMMON": {
                    "DATA_ANNO_DIR": "extracted_frames/",
                    "DATA_IMG_DIR": "extracted_frames/",
                    "DYNAMIC_MERGE": False,
                    "ENABLE_PREV_FRAME": False,
                    "MAX_OBJ_NUM": 10, # Should match cfg.DATASETS_TRAIN_MAX_OBJ_NUM[0]
                    "OUTPUT_SIZE": [473, 473], # Should match CROP_SIZE
                    "REPEAT_TIME": 1,
                    "SEQ_LEN": 1 # Should match cfg.DATASETS_TRAIN_SEQ_LEN[0]
                },
                "TRAIN": { # Specific to "train" TYPE
                    "IGNORE_IN_MERGE": False,
                    "IGNORE_THRESH": 0.0, # Example, might need adjustment
                    "MERGE_PROB": 0.0,
                    "RAND_GAP": 1, # Not very relevant for single frames
                    "RAND_REVERSE": False,
                    "RGB": True,
                    "USE_MERGE": False
                }
            }
            # "TYPE" was moved up to be a direct child of dataset_name key
        }
    }

    cfg.TRAIN_GPUS = 1 # Test with 1 GPU
    cfg.DIST_ENABLE = False # Disable distributed training for this test
    cfg.TRAIN_BN_MOMENTUM = 0.01 # Example value
    cfg.TRAIN_BN_TYPE = "pytorch"

    # Pretrained model path - can be empty if not loading weights
    cfg.PRETRAIN_MODEL = "" # Not essential for dataset loading test

    # 4. Initialize essential directories
    # cfg.DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Should be aot_plus parent
    # The get_config now sets DIR_ROOT to /app/aot_plus via configs.default.py
    # We need to ensure DIR_EXP etc. are correctly formed based on that.
    # DefaultEngineConfig in configs.default sets DIR_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
    # which if __file__ is configs/default.py, means DIR_ROOT is aot_plus parent.
    # Let's ensure DIR_ROOT is /app for consistency with how experiments might be structured.
    # The config loaded from pre_ytb.yaml (via EngineConfig) will set these.
    # We might need to override some of them if the defaults are not suitable for the test.

    # These will be set by the loaded config (e.g. from default.py -> EngineConfig)
    # We are overriding some here for the test if needed, or ensuring they exist.
    if not hasattr(cfg, 'DIR_ROOT'):
        cfg.DIR_ROOT = os.path.abspath(os.path.join(os.getcwd())) # Should be /app/aot_plus

    cfg.DIR_EXP = os.path.join(cfg.DIR_ROOT, 'experiments_test', cfg.EXP_NAME) # Use a different experiments root for test
    cfg.DIR_LOG = os.path.join(cfg.DIR_EXP, 'log')
    cfg.DIR_CKPT = os.path.join(cfg.DIR_EXP, 'ckpt')
    cfg.DIR_EMA_CKPT = os.path.join(cfg.DIR_EXP, 'ema_ckpt')
    cfg.DIR_TB = os.path.join(cfg.DIR_EXP, 'tb')


    if not os.path.exists(cfg.DIR_LOG):
        os.makedirs(cfg.DIR_LOG)
    if not os.path.exists(cfg.DIR_CKPT):
        os.makedirs(cfg.DIR_CKPT)

    # Create a dummy extracted_frames directory and a sample file for the dataset loader
    # to prevent FileNotFoundError if it tries to list files immediately.
    # The ExtractedFramesTrain loader uses image_root and label_root from its own __init__,
    # which are set to "extracted_frames/". So these files need to be in /app/aot_plus/extracted_frames/

    # Ensure CWD is /app/aot_plus for these relative paths to work as expected by the dataset loader
    # The script already tries to cd to aot_plus at the beginning.
    # dummy_data_dir = os.path.join(os.getcwd(), "extracted_frames")
    # The paths in ExtractedFramesTrain are relative to the project root (aot_plus)
    dummy_data_dir = os.path.join(cfg.DIR_ROOT, "extracted_frames")
    if os.path.basename(cfg.DIR_ROOT) != "aot_plus": # If DIR_ROOT was somehow not aot_plus
        dummy_data_dir = os.path.join(os.getcwd(), "extracted_frames")


    if not os.path.exists(dummy_data_dir):
        os.makedirs(dummy_data_dir)
        print(f"Created dummy data directory: {dummy_data_dir}")

    # Create a dummy image and json file
    # The dataset loader might try to access os.path.join(self.image_root, seqname, img_filename)
    # seqname is filename without extension. So, files should be directly in extracted_frames.
    sample_img_path = os.path.join(dummy_data_dir, "sample1.jpg")
    sample_json_path = os.path.join(dummy_data_dir, "sample1.json")

    if not os.path.exists(sample_img_path):
        with open(sample_img_path, "w") as f:
            f.write("dummy image data")
        print(f"Created dummy file: {sample_img_path}")
    if not os.path.exists(sample_json_path):
        with open(sample_json_path, "w") as f:
            f.write('{"imageHeight": 100, "imageWidth": 100, "shapes": [{"label": "obj1", "points": [[10,10],[20,20],[10,20]]}]}')
        print(f"Created dummy file: {sample_json_path}")


    print("Configuration prepared. Attempting to initialize Trainer...")
    try:
        # 5. Attempt to instantiate the Trainer
        # Assuming rank 0 for a single process test
        trainer = Trainer(rank=gpu_id, cfg=cfg, enable_amp=True)
        print("Trainer initialized successfully!")
        # Optionally, try to prepare dataset explicitly if Trainer init doesn't do it thoroughly enough
        # print("Attempting to prepare dataset...")
        # trainer.prepare_dataset()
        # print("Dataset prepared successfully!")

    except ValueError as ve:
        print(f"ValueError during Trainer initialization: {ve}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during Trainer initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set current working directory to 'aot_plus' if the script is inside it
    # This is important because configs and other paths are relative to 'aot_plus'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "aot_plus":
        os.chdir(script_dir)
    elif os.path.basename(os.getcwd()) != "aot_plus":
        # Attempt to change to aot_plus if script is in a subdir of aot_plus parent
        # or if aot_plus is a subdir of CWD
        if os.path.exists("aot_plus"):
            os.chdir("aot_plus")
        elif os.path.exists(os.path.join("..", "aot_plus")): # If script is in like aot_plus/tools
             os.chdir(os.path.join("..")) # This makes CWD aot_plus if script is in aot_plus/test_trainer_init.py
        else:
            print("Error: Script not in 'aot_plus' directory or 'aot_plus' not found in parent. Please run from 'aot_plus'.")
            sys.exit(1)

    run_test()
