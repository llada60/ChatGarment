#!/bin/bash
#SBATCH --job-name=0eval          # Name of your job
#SBATCH --output=eval_logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=eval_logs/%x_%j.err             # Error file
#SBATCH --partition=L40S              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

export CUDA_HOME=/usr/local/cuda-12.1 \
PATH=$CUDA_HOME/bin:$PATH:/bin:/usr/bin \
LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_PORT=50000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chatgarment

./scripts/v1_5/evaluate_garment_v2_sketch_2step.sh /home/ids/liliu/data/ChatGarment/evaluations/close_eva_imgs/sketch /home/ids/liliu/projects/ChatGarment/runs/chatgarment_pre_trained/ckpt_model_epoch0_07-27_12_37/model_fp32.pt
python run_garmentcode_sim.py --all_paths_json /home/ids/liliu/projects/ChatGarment/runs/ckpt_model_epoch0_07-27_12_37/close_eva_imgs_img_recon
cd ContourCraft-CG

python evaluation_scripts/close_evaluate.py --method llava --path /home/ids/liliu/projects/ChatGarment/runs/ckpt_model_epoch0_07-27_12_37/close_eva_imgs_img_recon
python evaluation_scripts/close_evaluate.py --method llava --path /home/ids/liliu/projects/ChatGarment/runs/ckpt_model_epoch0_07-27_12_37/close_eva_imgs_img_recon  --is_fscore 1