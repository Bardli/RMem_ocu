import torch
from torch.utils.data import DataLoader

# Assuming image_transforms and video_transforms are modules in the same directory or installed
# For simplicity, direct imports from .train_datasets and .eval_datasets are used.
# If these modules define transform classes, they would be imported here or passed in cfg.
from .train_datasets import (ExtractedFramesTrain, StaticTrain, VOSTrain,
                             DAVIS2017_Train, YOUTUBEVOS_Train, TEST,
                             VOST_Train, VISOR_Train)
from .eval_datasets import (VOSTest, DAVIS2017_Test, YOUTUBEVOS_Test,
                            VisorTest)

# It's common to have transform modules.
# from . import image_transforms
# from . import video_transforms

def build_train_dataloader(cfg):
    """
    Builds the training dataloader based on the configuration.
    """
    datasets = []
    if not hasattr(cfg, 'DATASETS') or not cfg.DATASETS:
        raise ValueError("cfg.DATASETS must be defined and not empty.")

    if not hasattr(cfg, 'DATASET_CONFIGS') or not cfg.DATASET_CONFIGS:
        raise ValueError("cfg.DATASET_CONFIGS must be defined and not empty.")

    # TODO: Implement transforms pipeline based on cfg
    # For now, transform is None. In a real scenario, this would be built
    # using cfg.DATA_RANDOMCROP, cfg.DATA_RANDOMFLIP, etc.
    # and specific transform classes from image_transforms or video_transforms.
    transform = None 

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
                transform=transform, # This should be a composed transform object
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
                output_size=common_config.get("OUTPUT_SIZE", cfg.DATA_RANDOMCROP),
                seq_len=common_config.get("SEQ_LEN", 5),
                max_obj_n=common_config.get("MAX_OBJ_NUM", 10),
                # dynamic_merge and merge_prob could come from train_config or common_config
                dynamic_merge=train_config.get("DYNAMIC_MERGE", common_config.get("DYNAMIC_MERGE", True)),
                merge_prob=train_config.get("MERGE_PROB", 1.0)
            )
            datasets.append(ds)
        elif dataset_type == "YOUTUBEVOS_Train":
            # Constructor: root='./datasets/YTB', year=2019, transform=None, rgb=True, rand_gap=3, 
            #              seq_len=3, rand_reverse=True, dynamic_merge=True, 
            #              enable_prev_frame=False, max_obj_n=10, merge_prob=0.3
            ds = YOUTUBEVOS_Train(
                root=cfg.DIR_YTB, # Assuming global DIR_YTB
                year=train_config.get("YEAR", 2019), # Example: could be in train_config
                transform=transform,
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
    
    # Common DataLoader parameters
    # These should ideally come from cfg (e.g., cfg.TRAIN_BATCH_SIZE, cfg.DATA_WORKERS)
    batch_size = getattr(cfg, "TRAIN_BATCH_SIZE", 1) 
    num_workers = getattr(cfg, "DATA_WORKERS", 0)
    shuffle = True # Typically true for training

    train_loader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Common optimization
        drop_last=True # Usually true for training
    )
    
    return train_loader

def build_eval_dataloader(cfg, dataset_name, split='val'):
    """
    Builds an evaluation dataloader.
    Dataset specific configurations should be handled.
    """
    if not hasattr(cfg, 'DATASET_CONFIGS') or dataset_name not in cfg.DATASET_CONFIGS:
        # Fallback or error if specific config not found
        print(f"Warning: No specific config for {dataset_name} in DATASET_CONFIGS for eval. Using global settings.")
        # This part needs more robust handling for eval datasets.
        # For example, VOSTest might take root, split, etc.
        # eval_transform = ... (usually simpler than train_transform)
        eval_transform = None 
        
        if dataset_name == 'VOSTest': # Example
            ds = VOSTest(root=cfg.DIR_VOST, transform=eval_transform, split=split)
        elif dataset_name == 'DAVIS2017_Test':
            ds = DAVIS2017_Test(root=cfg.DIR_DAVIS, transform=eval_transform, split=split, year=2017)
        # Add other eval dataset types
        else:
            raise NotImplementedError(f"Evaluation for dataset {dataset_name} not implemented.")
        
        eval_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=getattr(cfg, "TEST_WORKERS", 0))
        return eval_loader

    # Preferred path: Using DATASET_CONFIGS if available for eval datasets too
    dataset_config_entry = cfg.DATASET_CONFIGS[dataset_name]
    dataset_type = dataset_config_entry.get("TYPE") # e.g. "VOSTest"
    # ... (similar logic to build_train_dataloader for instantiating eval datasets) ...
    
    # Placeholder for now, proper implementation would mirror train but for eval types.
    print(f"Building eval dataloader for {dataset_name} type {dataset_type} (not fully implemented via DATASET_CONFIGS yet).")
    # Fallback to the simpler implementation above for now.
    return build_eval_dataloader(cfg, dataset_name, split) # Recursive call to the simpler logic for now.


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
