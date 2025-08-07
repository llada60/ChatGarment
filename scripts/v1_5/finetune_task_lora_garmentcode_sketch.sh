#!/bin/bash

# export LD_LIBRARY_PATH=/is/software/nvidia/cuda-12.1/lib64
# export PATH=$PATH:/is/software/nvidia/cuda-12.1/bin
# export CUDA_HOME=/is/software/nvidia/cuda-12.1

# export CPATH=/is/software/nvidia/cudnn-8.4.1-cu11.6/include
# export C_INCLUDE_PATH=/is/software/nvidia/cudnn-8.4.1-cu11.6/include
# export LIBRARY_PATH=/is/software/nvidia/cudnn-8.4.1-cu11.6/lib64
# export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH

export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
# export TCNN_CUDA_ARCHITECTURES=80

deepspeed llava/train/sketch/train_mem_garmentcode_outfit.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/ids/liliu/data/llava/llava-v1.5-7b \
    --version v1 \
    --data_path ./ \
    --data_path_eval /home/ids/liliu/data/ChatGarment/evaluations/garment_edit_eva.json \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

