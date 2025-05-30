import torch
from torch.utils.data import DataLoader

# Assuming image_transforms and video_transforms are modules in the same directory or installed
# For simplicity, direct imports from .train_datasets and .eval_datasets are used.
# If these modules define transform classes, they would be imported here or passed in cfg.
from .train_datasets import (ExtractedFramesTrain, StaticTrain, VOSTrain,
                             DAVIS2017_Train, YOUTUBEVOS_Train, TEST,
                             VOST_Train, VISOR_Train)
from .eval_datasets import (VOSTest, DAVIS_Test, YOUTUBEVOS_Test) # Corrected DAVIS2017_Test to DAVIS_Test, removed VisorTest

# It's common to have transform modules.
# from . import image_transforms
# from . import video_transforms

def build_train_dataset(cfg, transforms=None): # Renamed and added transforms argument
    """
    Builds the training dataset based on the configuration.
    Returns a torch.utils.data.Dataset object.
    """
    datasets = []
    if not hasattr(cfg, 'DATASETS') or not cfg.DATASETS:
        raise ValueError("cfg.DATASETS must be defined and not empty.")

    if not hasattr(cfg, 'DATASET_CONFIGS') or not cfg.DATASET_CONFIGS:
        raise ValueError("cfg.DATASET_CONFIGS must be defined and not empty.")

    # The 'transforms' argument is now passed in from the caller (Trainer.prepare_dataset)
    # transform = None # This line is removed

    for dataset_name in cfg.DATASETS:
        if dataset_name not in cfg.DATASET_CONFIGS:
            raise ValueError(f"Configuration for dataset {dataset_name} not found in cfg.DATASET_CONFIGS.")
        
        dataset_config_entry = cfg.DATASET_CONFIGS[dataset_name]
        dataset_type = dataset_config_entry.get("TYPE")
        specific_config = dataset_config_entry.get("CONFIG", {})
        common_config = specific_config.get("COMMON", {})
        train_config = specific_config.get("TRAIN", {})

        if dataset_type == "ExtractedFramesTrain":
            # Map config values to ExtractedFramesTrain constructor arguments
            # Constructor: transform=None, rgb=True, repeat_time=1, seq_len=1, max_obj_n=10, ignore_thresh=1.0
            ds = ExtractedFramesTrain(
                # root=common_config.get("DATA_IMG_DIR", cfg.DIR_EXTRACTED_FRAMES), # Assuming DIR_EXTRACTED_FRAMES is defined in cfg
                transform=transforms, # Use the passed-in transforms
                rgb=train_config.get("RGB", True),
                repeat_time=common_config.get("REPEAT_TIME", 1),
                seq_len=common_config.get("SEQ_LEN", 1),
                max_obj_n=common_config.get("MAX_OBJ_NUM", 10),
                ignore_thresh=train_config.get("IGNORE_THRESH", 1.0) # or 0.0 as per default.py
            )
            datasets.append(ds)
        elif dataset_type == "StaticTrain":
            # Constructor: root, output_size, seq_len=5, max_obj_n=10, dynamic_merge=True, merge_prob=1.0
            # This needs root and output_size from common_config or global cfg
            ds = StaticTrain(
                root=cfg.DIR_STATIC, # Assuming global DIR_STATIC from cfg
                output_size=common_config.get("OUTPUT_SIZE", cfg.DATA_RANDOMCROP), # output_size is specific to StaticTrain's needs
                seq_len=common_config.get("SEQ_LEN", 5),
                max_obj_n=common_config.get("MAX_OBJ_NUM", 10),
                # dynamic_merge and merge_prob could come from train_config or common_config
                dynamic_merge=train_config.get("DYNAMIC_MERGE", common_config.get("DYNAMIC_MERGE", True)),
                merge_prob=train_config.get("MERGE_PROB", 1.0)
                # Note: StaticTrain itself applies transforms internally, so passing `transforms` here might be redundant
                # or require StaticTrain to be adapted. For now, we assume StaticTrain handles its own transforms
                # or the passed `transforms` is compatible / None for it.
                # If StaticTrain is meant to use the global `transforms`, its constructor/logic might need adjustment.
            )
            datasets.append(ds)
        elif dataset_type == "YOUTUBEVOS_Train":
            # Constructor: root='./datasets/YTB', year=2019, transform=None, rgb=True, rand_gap=3, 
            #              seq_len=3, rand_reverse=True, dynamic_merge=True, 
            #              enable_prev_frame=False, max_obj_n=10, merge_prob=0.3
            ds = YOUTUBEVOS_Train(
                root=cfg.DIR_YTB, # Assuming global DIR_YTB
                year=train_config.get("YEAR", 2019), # Example: could be in train_config
                transform=transforms, # Use the passed-in transforms
                rgb=train_config.get("RGB", True),
                rand_gap=train_config.get("RAND_GAP", cfg.DATA_RANDOM_GAP_YTB),
                seq_len=common_config.get("SEQ_LEN", 3),
                rand_reverse=train_config.get("RAND_REVERSE", cfg.DATA_RANDOM_REVERSE_SEQ),
                dynamic_merge=common_config.get("DYNAMIC_MERGE", True),
                enable_prev_frame=common_config.get("ENABLE_PREV_FRAME", False),
                max_obj_n=common_config.get("MAX_OBJ_NUM", 10),
                merge_prob=train_config.get("MERGE_PROB", cfg.DATA_DYNAMIC_MERGE_PROB)
            )
            datasets.append(ds)
        # Add other dataset types here (DAVIS2017_Train, VOST_Train, VISOR_Train)
        # For brevity, only ExtractedFramesTrain, StaticTrain, and YOUTUBEVOS_Train are sketched out.
        else:
            print(f"Warning: Dataset type '{dataset_type}' for '{dataset_name}' is not yet supported in build_train_dataloader. Skipping.")
            # Alternatively, raise NotImplementedError(f"Dataset type {dataset_type} not supported.")

    if not datasets:
        raise ValueError("No datasets were loaded. Check cfg.DATASETS and their configurations.")

    # If multiple datasets, typically ConcatDataset is used.
    # For now, if len(datasets) > 1, we might need to handle it.
    # Let's assume for now that cfg.DATASETS will usually contain one dataset for training,
    # or that the Trainer handles a list of datasets.
    # If only one dataset, use it directly.
    final_dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    
    return final_dataset # Return the dataset object, not the DataLoader

def build_eval_dataloader(cfg, dataset_name, split='val'):
    """
    Builds an evaluation dataloader based on the configuration.
    """
    # cfg.DATASETS might not be the primary list for eval, dataset_name is passed directly.
    # However, DATASET_CONFIGS is essential.
    if not hasattr(cfg, 'DATASET_CONFIGS') or not cfg.DATASET_CONFIGS:
        raise ValueError("cfg.DATASET_CONFIGS must be defined and not empty for build_eval_dataloader.")

    if dataset_name not in cfg.DATASET_CONFIGS:
        raise ValueError(f"Configuration for dataset '{dataset_name}' not found in cfg.DATASET_CONFIGS for eval.")

    # Define eval_transform, potentially from cfg or a default simple transform for eval
    # Placeholder: build this from cfg if needed (e.g., ToTensor, Normalize)
    # This would be similar to how transforms are handled in build_train_dataloader,
    # but typically without data augmentation.
    eval_transform = None 

    dataset_config_entry = cfg.DATASET_CONFIGS[dataset_name]
    dataset_type = dataset_config_entry.get("TYPE")
    specific_config = dataset_config_entry.get("CONFIG", {})
    common_config = specific_config.get("COMMON", {})
    # eval_config = specific_config.get("EVAL", {}) # Optional: if "EVAL" specific params exist

    dataset = None
    if dataset_type == "ExtractedFramesTrain": # Using the train class for eval
        dataset = ExtractedFramesTrain(
            transform=eval_transform, # Use the passed-in eval_transform
            rgb=common_config.get("RGB", True), # from common config
            seq_len=1, # Override for eval: typically process single frames
            max_obj_n=common_config.get("MAX_OBJ_NUM", 10), # from common config
            # repeat_time is not relevant for eval
            # ignore_thresh might be relevant if eval script handles it, otherwise default
            ignore_thresh=common_config.get("IGNORE_THRESH", 0.0) # Using common_config for consistency
        )
    elif dataset_type == "VOSTest":
        # VOSTest is more complex as it's usually instantiated per sequence by the eval script.
        # The eval script (e.g., tools/eval.py) would need to provide sequence-specific info.
        # This function might only return the class and its fixed params, or need more context.
        # For now, this is a placeholder for how it *could* be structured if called directly with global cfg.
        print(f"Warning: Instantiation for {dataset_type} in build_eval_dataloader is complex and typically handled per-sequence by an evaluation script.")
        # Example (highly simplified, likely incorrect for direct use without eval script logic):
        # dataset = VOSTest(
        #     image_root=common_config.get("DATA_IMG_DIR", cfg.DIR_VOST),
        #     label_root=common_config.get("DATA_ANNO_DIR", cfg.DIR_VOST), # Annotations might be different for eval
        #     transform=eval_transform,
        #     rgb=common_config.get("RGB", True),
        #     # Other VOSTest specific parameters like 'seq_name', 'images', 'labels'
        #     # would need to be passed or handled by the calling evaluation script.
        # )
        raise NotImplementedError(f"Full instantiation for {dataset_type} within build_eval_dataloader is complex. Evaluation scripts usually handle per-sequence setup.")
    elif dataset_type == "DAVIS_Test": # Corrected name
        print(f"Warning: Instantiation for {dataset_type} in build_eval_dataloader is complex.")
        raise NotImplementedError(f"Full instantiation for {dataset_type} within build_eval_dataloader is complex. Evaluation scripts usually handle per-sequence setup.")
    elif dataset_type == "YOUTUBEVOS_Test":
        print(f"Warning: Instantiation for {dataset_type} in build_eval_dataloader is complex.")
        raise NotImplementedError(f"Full instantiation for {dataset_type} within build_eval_dataloader is complex. Evaluation scripts usually handle per-sequence setup.")
    # Add other evaluation dataset types as needed
    else:
        raise NotImplementedError(f"Evaluation for dataset type '{dataset_type}' (from dataset '{dataset_name}') not implemented in build_eval_dataloader.")

    if dataset is None: # Should be caught by NotImplementedError or successful instantiation
        raise ValueError(f"Failed to create dataset for '{dataset_name}' with type '{dataset_type}'. This should not happen.")

    eval_loader = DataLoader(
        dataset,
        batch_size=getattr(cfg, "TEST_BATCH_SIZE", 1), 
        shuffle=False, # Never shuffle for evaluation
        num_workers=getattr(cfg, "TEST_WORKERS", getattr(cfg, "DATA_WORKERS", 0)), # Use TEST_WORKERS or fallback
        pin_memory=True,
        drop_last=False
    )
    return eval_loader

# Example of how transforms could be built (conceptual)
# def build_transforms(cfg, is_train=True):
#     if is_train:
#         # common_transforms = ...
#         # aug_transforms = ...
#         # final_transforms = ...
#         # return Compose(final_transforms)
#         pass
#     else:
#         # return Compose([...])
#         pass
