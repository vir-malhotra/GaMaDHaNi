# Quick Start: Raga-Conditioned GaMaDHaNi

## What Was Implemented

✅ **Hierarchical raga conditioning** using FiLM (Feature-wise Linear Modulation)
✅ **Raga embedding layer** - learnable representations for each raga
✅ **Modified transformer architecture** - conditions generation on raga IDs
✅ **Training pipeline** - extracts raga from dataset, trains conditioned model
✅ **Generation script** - generate music for specific ragas
✅ **HPC-ready** - SLURM script for cluster training

## Files Created/Modified

### New Files:
- `gamadhani/src/model_transformer_raga.py` - Raga-conditioned transformer
- `gamadhani/scripts/train_transformer_raga.py` - Training script
- `configs/transformer_pitch_raga_config.gin` - Configuration
- `scripts/train_raga_hpc.sh` - SLURM submission script
- `generate_raga.py` - Generation with raga conditioning
- `RAGA_CONDITIONING_README.md` - Full documentation

### Modified Files:
- `gamadhani/src/task_functions.py` - Now passes raga IDs from dataset

## Quick Start Steps

### 1. Verify Your Dataset Has Raga Labels

```python
from gamadhani.src.dataset import SequenceDataset

dataset = SequenceDataset(db_path="path/to/train")
sample = dataset[0]

# Check if raga exists
if 'raga' in sample:
    print(f"✓ Dataset has raga labels! Raga ID: {sample['raga']}")
else:
    print("✗ Dataset missing raga labels - cannot use raga conditioning")
```

### 2. Test Locally (Optional)

```bash
# Small test run to verify everything works
python gamadhani/scripts/train_transformer_raga.py \
    --config=configs/transformer_pitch_raga_config.gin \
    --db_path=/path/to/your/dataset \
    --batch_size=16 \
    --max_epochs=5 \
    --debug=True \
    --log_to_wandb=False
```

### 3. Train on HPC

#### Edit the SLURM script:
```bash
nano scripts/train_raga_hpc.sh

# Update this line:
DB_PATH="/path/to/your/dataset"  # Change to your actual path
```

#### Submit job:
```bash
chmod +x scripts/train_raga_hpc.sh
mkdir -p logs
sbatch scripts/train_raga_hpc.sh
```

#### Monitor:
```bash
# Check job
squeue -u $USER

# Watch progress
tail -f logs/gamadhani_raga_*.out

# Checkpoints saved to:
# $SCRATCH/checkpoints/gamadhani_raga/raga-conditioning-exp1/<job_id>/models/
```

### 4. Generate with Trained Model

```bash
# Replace with your checkpoint path
CHECKPOINT="/path/to/checkpoint/models/best.ckpt"
AUDIO_MODEL="/path/to/audio/model"

# Generate Raga Yaman (ID 0)
python generate_raga.py \
    --pitch_model_type=transformer \
    --pitch_run=$CHECKPOINT \
    --audio_run=$AUDIO_MODEL \
    --number_of_samples=5 \
    --raga_id=0 \
    --raga_name="Yaman" \
    --seq_len=1200 \
    --temperature=0.95 \
    --outfolder=./outputs/yaman

# Generate multiple ragas
python generate_raga.py \
    --pitch_model_type=transformer \
    --pitch_run=$CHECKPOINT \
    --audio_run=$AUDIO_MODEL \
    --number_of_samples=3 \
    --raga_ids=0,5,10 \
    --seq_len=1200 \
    --outfolder=./outputs/comparison
```

## Architecture Overview

```
Input Tokens → Token Embedding
                    ↓
        Transformer Layer 1
                    ↓
        [Raga Embedding] ← Raga ID
                    ↓
        FiLM Modulation (γ * x + β)
                    ↓
        Transformer Layer 2
                    ↓
                  ...
                    ↓
        FiLM Modulation
                    ↓
        Output Layer → Next Token
```

**Key Innovation**: FiLM applies learned scale (γ) and shift (β) based on raga, modulating the transformer's representations to capture raga-specific patterns.

## Expected Results

After training (300-500 epochs):
- **Different ragas produce different melodic patterns**
- **Pitch distributions vary by raga**
- **Raga-specific phrases emerge**
- **More authentic Hindustani classical music generation**

## Common Issues

### "No raga in dataset"
→ Your dataset doesn't have raga labels in `global_conditions`. You'll need raga-labeled data.

### "All ragas sound the same"
→ Need more training epochs (100+) or more data per raga.

### "CUDA out of memory"
→ Reduce batch size in SLURM script: `BATCH_SIZE=32` or `BATCH_SIZE=16`

### "Where do I find audio model?"
→ Use original GaMaDHaNi audio model (download from Hugging Face or use existing)

## Next Steps

1. **Train the model** on your HPC cluster
2. **Identify raga mappings** (which ID = which raga name)
3. **Generate samples** for different ragas
4. **Evaluate** with expert musicians
5. **Compare** conditioned vs unconditioned generation

## Key Hyperparameters

| Parameter | What it Does | Default | When to Change |
|-----------|--------------|---------|----------------|
| `NUM_RAGAS` | Number of ragas | 64 (auto) | If you know exact count |
| `RAGA_DIM` | Embedding size | 128 | Increase for complex ragas |
| `BATCH_SIZE` | Training batch | 64 | Reduce if OOM errors |
| `LR` | Learning rate | 1e-3 | Standard for transformers |
| `temperature` | Generation randomness | 0.95 | Higher = more creative |

## Full Documentation

See [RAGA_CONDITIONING_README.md](RAGA_CONDITIONING_README.md) for:
- Detailed architecture explanation
- Complete API reference
- Advanced configuration
- Troubleshooting guide
- Validation procedures

## Questions?

1. Check [RAGA_CONDITIONING_README.md](RAGA_CONDITIONING_README.md)
2. Verify dataset has raga labels
3. Check SLURM output logs
4. Review wandb training curves (if enabled)

---

**Ready to train?** → Edit `scripts/train_raga_hpc.sh` and run `sbatch scripts/train_raga_hpc.sh`
