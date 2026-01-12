#!/bin/bash
#SBATCH --job-name=extract_pythia
#SBATCH --account=rrg-msh
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/anasiri/links/scratch/slurm_logs/%x_%j.out
#SBATCH --error=/home/anasiri/links/scratch/slurm_logs/%x_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo ""

# HuggingFace cache settings
export HF_HOME=/home/anasiri/links/scratch/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Unbuffered Python output for real-time monitoring
export PYTHONUNBUFFERED=1

# Load required modules
module load gentoo/2023 gcc arrow/21.0.0

# Activate conda environment
source /home/anasiri/jupyter_py3/bin/activate

# Change to project directory
cd /project/rrg-msh/anasiri/MonoLoss_git/sae

# Run feature extraction
python extract_features_text.py \
    --hf_dataset /home/anasiri/links/scratch/hf_datasets/monology_pile_unc \
    --output_dir /home/anasiri/links/scratch/MonoLoss_features \
    --model_type pythia \
    --model_name EleutherAI/pythia-410m-deduped \
    --layers 11 \
    --site post \
    --pool mean \
    --batch_size 16

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $?"
