# Training Setup Guide

## Quick Start: Create CSV and Setup Validation

### Step 1: Create CSV with Validation Split

```bash
cd /ccn2/u/wanhee/VideoX-Fun/scripts/wan2.2_fun

python create_csv_from_videos.py \
    /scratch/m000063/data/bvd2/handpicked/ \
    --val_split 0.2
```

**Creates:**
- `../../datasets/handpicked.csv` - Training data (80%)
- `../../datasets/handpicked_val.csv` - Validation reference (20%)
- `../../datasets/handpicked_val_videos.txt` - List of validation video paths

### Step 2: Generate Validation Config

```bash
python generate_validation_config.py \
    ../../datasets/handpicked_val_videos.txt \
    --num_samples 5
```

**Output:**
```bash
# Validation settings
export VALIDATION_STEPS=1000
export VALIDATION_PROMPTS=(
    ""
    ""
    ""
    ""
    ""
)
export VALIDATION_PATHS=(
    "/scratch/m000063/data/bvd2/handpicked/video_001.mp4"
    "/scratch/m000063/data/bvd2/handpicked/video_015.mp4"
    ...
)
```

### Step 3: Update Training Script

Edit `train_control_lora_psi.sh`:

**Line 13** - Update training CSV:
```bash
export DATA_META="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/handpicked.csv"
```

**Lines 44-53** - Replace validation section with output from Step 2

**Line 16** - Update PSI models directory:
```bash
export PSI_MODELS_DIR="/path/to/your/psi/models"
```

### Step 4: Run Training

```bash
./train_control_lora_psi.sh
```

## Notes

- Empty prompts (`""`) work for control-based training without text descriptions
- CSV auto-names based on directory name (handpicked → handpicked.csv)
- Recommended: 3-5 validation samples, 10-20% validation split

