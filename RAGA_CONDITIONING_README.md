# Raga-Conditioned GaMaDHaNi Training & Generation

This document describes how to train and use raga-conditioned models for GaMaDHaNi.

## Overview

The raga-conditioned implementation uses **FiLM (Feature-wise Linear Modulation)** to hierarchically condition the transformer model on raga information. This allows the model to learn raga-specific melodic patterns and generate authentic raga-based compositions.

## Architecture

### Key Components

1. **RagaEmbedding Layer** ([model_transformer_raga.py:68](gamadhani/src/model_transformer_raga.py:68))
   - Learnable embeddings for each raga
   - Dimension: 128 (configurable)
   - Initialized randomly, learned during training

2. **FiLM Conditioning** ([model_transformer_raga.py:21](gamadhani/src/model_transformer_raga.py:21))
   - Applies scale (γ) and shift (β) transformations
   - Modulates transformer features based on raga
   - Formula: `output = γ * features + β`

3. **Modified Task Pipeline** ([task_functions.py:54](gamadhani/src/task_functions.py:54))
   - Passes raga IDs from dataset to model
   - Extracts from protobuf `global_conditions.raga`

## Requirements

### Dataset Requirements

Your LMDB dataset **must** have raga labels in the protobuf structure:

```protobuf
message GlobalConditions {
    optional float tonic = 1;
    optional int32 raga = 2;      // Required for raga conditioning
    optional int32 singer = 3;
}
```

Check if your dataset has raga labels:
```python
from gamadhani.src.dataset import SequenceDataset

dataset = SequenceDataset(db_path="path/to/train")
sample = dataset[0]
print("Raga ID:", sample.get('raga', 'NOT FOUND'))
```

### Environment Setup

```bash
# Python 3.10+
conda create -n gamadhani_raga python=3.10
conda activate gamadhani_raga

# Install dependencies
pip install -r requirements.txt

# Additional for HPC
module load cuda/11.8
module load cudnn/8.6
```

## Training

### 1. Local Training (for testing)

```bash
python gamadhani/scripts/train_transformer_raga.py \
    --config=configs/transformer_pitch_raga_config.gin \
    --db_path=/path/to/dataset \
    --batch_size=32 \
    --workers=4 \
    --gpu=0 \
    --max_epochs=50 \
    --debug=True \
    --log_to_wandb=False
```

### 2. HPC Training (full training)

#### Step 1: Prepare the SLURM script

Edit [scripts/train_raga_hpc.sh](scripts/train_raga_hpc.sh):

```bash
# Update these variables:
DB_PATH="/path/to/your/dataset"  # Your LMDB dataset path
BATCH_SIZE=64                     # Adjust based on GPU memory
NUM_WORKERS=8                     # Number of CPU workers
MAX_EPOCHS=500                    # Full training: 300-500 epochs
```

#### Step 2: Submit job

```bash
# Make script executable
chmod +x scripts/train_raga_hpc.sh

# Create logs directory
mkdir -p logs

# Submit to SLURM
sbatch scripts/train_raga_hpc.sh
```

#### Step 3: Monitor training

```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f logs/gamadhani_raga_*.out

# Watch wandb (if enabled)
# Visit: https://wandb.ai/your-username/gamadhani-raga-conditioning
```

### 3. Configuration Options

#### Override config parameters:

```bash
python gamadhani/scripts/train_transformer_raga.py \
    --config=configs/transformer_pitch_raga_config.gin \
    --db_path=/path/to/dataset \
    --override="NUM_RAGAS=72" \
    --override="RAGA_DIM=256" \
    --override="LR=5e-4" \
    --batch_size=64
```

#### Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_RAGAS` | 64 | Number of unique ragas in dataset (auto-inferred if not set) |
| `RAGA_DIM` | 128 | Raga embedding dimension |
| `MODEL_DIM` | 512 | Transformer hidden dimension |
| `NUM_LAYERS` | 8 | Number of transformer layers |
| `BATCH_SIZE` | 64 | Training batch size |
| `LR` | 1e-3 | Learning rate |
| `SEQ_LEN` | 1200 | Sequence length (tokens) |

### 4. Expected Training Time

| Setup | Hardware | Approximate Time |
|-------|----------|------------------|
| Full training | 1x A100 (40GB) | 24-48 hours |
| Full training | 1x V100 (32GB) | 48-72 hours |
| Test run (50 epochs) | 1x GPU | 2-4 hours |

## Generation

### 1. Unconditioned Generation (no raga)

```bash
python generate_raga.py \
    --pitch_model_type=transformer \
    --pitch_run=/path/to/checkpoint \
    --audio_run=/path/to/audio_model \
    --number_of_samples=5 \
    --seq_len=1200 \
    --temperature=0.95 \
    --outfolder=./outputs
```

### 2. Conditioned Generation (with raga)

#### Single raga for all samples:

```bash
python generate_raga.py \
    --pitch_model_type=transformer \
    --pitch_run=/path/to/checkpoint \
    --audio_run=/path/to/audio_model \
    --number_of_samples=5 \
    --raga_id=15 \
    --raga_name="Yaman" \
    --seq_len=1200 \
    --temperature=0.95 \
    --outfolder=./outputs/yaman
```

#### Different ragas for each sample:

```bash
python generate_raga.py \
    --pitch_model_type=transformer \
    --pitch_run=/path/to/checkpoint \
    --audio_run=/path/to/audio_model \
    --number_of_samples=3 \
    --raga_ids=15,22,8 \
    --seq_len=1200 \
    --outfolder=./outputs/mixed_ragas
```

### 3. Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--temperature` | Sampling randomness (0.5-1.5) | 0.9-1.0 for natural |
| `--seq_len` | Output length in tokens | 1200 (≈12 seconds) |
| `--raga_id` | Raga ID from dataset | 0 to NUM_RAGAS-1 |

## Raga Mapping

After training, you'll need to identify which raga ID corresponds to which raga name. Create a mapping file:

```python
# create_raga_mapping.py
from gamadhani.src.dataset import SequenceDataset
from collections import Counter

dataset = SequenceDataset(db_path="path/to/train")
raga_ids = []

for i in range(len(dataset)):
    sample = dataset[i]
    if 'raga' in sample and sample['raga'] >= 0:
        raga_ids.append(int(sample['raga']))

raga_counts = Counter(raga_ids)
print("Raga ID distribution:")
for raga_id, count in sorted(raga_counts.items()):
    print(f"  Raga {raga_id}: {count} samples")
```

If your dataset has metadata mapping raga IDs to names, create:

```json
// raga_mapping.json
{
  "0": "Yaman",
  "1": "Bhairav",
  "2": "Malkauns",
  ...
}
```

## Validation

### 1. Check if raga conditioning is working:

```python
import torch
from gamadhani.src.model_transformer_raga import XTransformerPriorRaga

# Load checkpoint
model = XTransformerPriorRaga.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Generate with different ragas
raga_0 = model.sample_fn(batch_size=1, seq_len=100, raga_ids=torch.tensor([0]))
raga_15 = model.sample_fn(batch_size=1, seq_len=100, raga_ids=torch.tensor([15]))

# Outputs should be different
print("Raga 0 mean:", raga_0.float().mean().item())
print("Raga 15 mean:", raga_15.float().mean().item())
```

### 2. Compare with baseline:

Generate samples with and without conditioning, compare:
- Pitch distribution statistics
- Interval patterns
- Expert musician evaluation

## Troubleshooting

### Issue: "Raga not found in dataset"

**Solution:** Check dataset has raga labels:
```python
sample = dataset[0]
print(sample.keys())  # Should include 'raga'
```

### Issue: "All ragas produce similar output"

**Possible causes:**
1. Model hasn't trained long enough (need 100+ epochs)
2. Dataset has imbalanced raga distribution
3. Raga conditioning strength too weak

**Solutions:**
- Train longer
- Check raga distribution in dataset
- Increase `RAGA_DIM` in config

### Issue: "CUDA out of memory"

**Solutions:**
- Reduce `batch_size` (try 32, 16, 8)
- Reduce `MODEL_DIM` (try 256)
- Use gradient accumulation:
  ```python
  # In training script
  trainer = pl.Trainer(..., accumulate_grad_batches=2)
  ```

### Issue: "Model generates silence"

**Possible causes:**
- Invalid raga ID (outside 0 to NUM_RAGAS-1)
- Model not properly loaded

**Solutions:**
- Check raga ID range
- Verify checkpoint loads correctly

## File Structure

```
GaMaDHaNi/
├── gamadhani/
│   ├── src/
│   │   ├── model_transformer_raga.py       # Raga-conditioned model
│   │   ├── task_functions.py               # Modified to pass raga
│   │   └── ...
│   ├── scripts/
│   │   └── train_transformer_raga.py       # Training script
│   └── ...
├── configs/
│   └── transformer_pitch_raga_config.gin   # Config file
├── scripts/
│   └── train_raga_hpc.sh                   # SLURM script
├── generate_raga.py                         # Generation with raga
└── RAGA_CONDITIONING_README.md             # This file
```

## Citation

If you use this raga-conditioned extension, please cite both the original GaMaDHaNi paper and acknowledge the conditioning implementation:

```bibtex
@article{shikarpur2024hierarchical,
  title={Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music},
  author={Shikarpur, Nithya and Dendukuri, Krishna Maneesha and Wu, Yusong and Caillon, Antoine and Huang, Cheng-Zhi Anna},
  journal={arXiv preprint arXiv:2408.12658},
  year={2024}
}
```

## Support

For issues specific to raga conditioning:
1. Check this README
2. Validate dataset has raga labels
3. Check wandb logs for training progress
4. Open an issue with training logs and config

## References

- Original GaMaDHaNi: https://github.com/snnithya/GaMaDHaNi
- FiLM: https://arxiv.org/abs/1709.07871
- Raga conditioning inspired by: Ross et al. (2017) Learning Embeddings of Raga
