#!/bin/bash
#SBATCH --job-name=gamadhani_raga
#SBATCH --output=logs/gamadhani_raga_%j.out
#SBATCH --error=logs/gamadhani_raga_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Adjust these based on your HPC system
# Common partitions: gpu, a100, v100, etc.
# Increase time for full training (48-72 hours typical)

# Set up environment
module load python/3.10
module load cuda/11.8  # Adjust based on your HPC
module load cudnn/8.6

# Activate virtual environment
source $HOME/gamadhani_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p logs
mkdir -p $SCRATCH/checkpoints/gamadhani_raga

# Training configuration
DB_PATH="YOUR_DATASET_PATH"  # CHANGE THIS to your dataset path
CONFIG="configs/transformer_pitch_raga_config.gin"
BATCH_SIZE=64
NUM_WORKERS=8
VAL_EVERY=5
CHECKPOINT_EVERY=10
MAX_EPOCHS=500
PATIENCE=30

# Optional: Specify number of ragas if known
# NUM_RAGAS=64

# Run training
python gamadhani/scripts/train_transformer_raga.py \
    --config=$CONFIG \
    --db_path=$DB_PATH \
    --batch_size=$BATCH_SIZE \
    --workers=$NUM_WORKERS \
    --val_every=$VAL_EVERY \
    --checkpoint_model_every=$CHECKPOINT_EVERY \
    --max_epochs=$MAX_EPOCHS \
    --patience=$PATIENCE \
    --gpu=0 \
    --group="raga-conditioning-exp1" \
    --log_to_wandb=True \
    --wandb_notes="Raga-conditioned GaMaDHaNi training with FiLM conditioning"

# Optional overrides
# Add to command above:
# --num_ragas=$NUM_RAGAS \
# --override="NUM_RAGAS=64" \
# --override="BATCH_SIZE=32"

echo "Training complete!"
