#!/bin/bash

# Experiment name (can be customized)
exp="finetune_extracted"

# GPU configuration (adjust as needed)
gpu_num="1" # Assuming single GPU for fine-tuning, can be increased
devices="0" # Assuming GPU ID 0, adjust if different

# Model - choose the model you want to fine-tune
# Examples (uncomment the one you want to use):
# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl" # Defaulting to r50_aotl, change if needed
# model="r50_deaotl"
# model="swinb_aotl"

# Stage - this should correspond to how you want to load weights or configure the model
# For fine-tuning, you might use a stage that loads pretrained weights.
# If 'EXTRACTED_FRAMES' config in default.py implies a stage, use that.
# Otherwise, you might need a generic stage or one that matches a pretrained model.
# Let's assume a stage name like 'pretrain' or use the model name itself if appropriate.
# For now, let's use a placeholder 'finetune_stage'. You may need to adjust this based on
# how pretrained models are loaded in this framework.
# A common practice is to have a stage for pretraining and another for fine-tuning.
# If no specific fine-tuning stage exists, you might need to use a pretraining stage
# and provide a path to your pretrained model weights via --pretrained_path.
stage="default" # Using 'default' stage, assuming it's generic enough or loads base weights.
                # Or, if you have a specific pretrained model, use its stage and specify PRETRAIN_MODEL in config or via args.

# Dataset name
dataset_name="EXTRACTED_FRAMES"

# Path to pretrained model (IMPORTANT FOR FINE-TUNING)
# Replace this with the actual path to your .pth pretrained model file.
# If the 'stage' above handles loading a base model, this might not be strictly needed,
# but for explicit fine-tuning, it's good practice.
pretrained_model_path="" # e.g., "./pretrain_models/r50_aotl.pth"

echo "Starting training for experiment: ${exp}"
echo "Model: ${model}"
echo "Stage: ${stage}"
echo "Dataset: ${dataset_name}"
echo "GPU_NUM: ${gpu_num}"
echo "DEVICES: ${devices}"
if [ ! -z "${pretrained_model_path}" ]; then
    echo "Pretrained model: ${pretrained_model_path}"
fi

# Construct the training command
COMMAND="CUDA_VISIBLE_DEVICES=${devices} python tools/train.py \
    --exp_name ${exp} \
    --stage ${stage} \
    --model ${model} \
    --datasets ${dataset_name} \
    --gpu_num ${gpu_num} \
    --batch_size 2 \ # Adjust batch size based on your GPU memory
    --total_step 10000 \ # Adjust total steps for fine-tuning
    --amp \ # Enable Automatic Mixed Precision if your hardware supports it
    --fix_random" # For reproducibility

if [ ! -z "${pretrained_model_path}" ]; then
    COMMAND="${COMMAND} \
    --pretrained_path ${pretrained_model_path}"
fi

# Optional: specify log directory
# COMMAND="${COMMAND} \
#    --log ./logs_extracted_frames"

echo ""
echo "Running command:"
echo "${COMMAND}"
echo ""

# Execute the command
${COMMAND}

echo "Training finished for experiment: ${exp}"
