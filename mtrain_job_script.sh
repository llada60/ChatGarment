#!/bin/bash
#SBATCH --job-name=try_finetune          # Name of your job
#SBATCH --output=logs/mtrain_logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/mtrain_logs/%x_%j.err             # Error file
#SBATCH --partition=A40              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH:/bin:/usr/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multigarment

# Execute the Python scjobjobript with specific arguments
./scripts/v1_5/finetune_task_lora_garmentcode_multi_sketch.sh