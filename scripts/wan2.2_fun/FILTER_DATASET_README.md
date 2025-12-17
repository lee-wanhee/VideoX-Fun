# Video Dataset Frame Count Filter

## Problem

When training with `VIDEO_SAMPLE_N_FRAMES=81` and `VIDEO_SAMPLE_STRIDE=2`, videos need **at least 161 frames** to provide the full 81-frame sequence.

Videos with fewer frames cause problems:
- ❌ Each batch truncates to the shortest video in that batch
- ❌ Different batches have different frame counts → variable tensor shapes
- ❌ Variable tensor shapes prevent GPU optimization (no cuDNN autotuning, no kernel fusion)
- ❌ Can hurt training speed by 20-30% and cause unstable gradients
- ❌ Different GPUs in distributed training get different sequence lengths

## Solution

Filter your dataset CSV to only include videos that meet the minimum frame requirement.

## Quick Start

### Option 1: Use the Convenient Shell Script

1. **Edit the script** to set your paths:
   ```bash
   vim filter_dataset.sh
   ```
   
   Update these lines:
   ```bash
   export INPUT_CSV="/path/to/your/dataset.csv"
   export OUTPUT_CSV="/path/to/your/dataset_filtered.csv"
   export DATA_ROOT="/path/to/videos"  # Or leave empty if CSV has absolute paths
   ```

2. **Run the filter:**
   ```bash
   ./filter_dataset.sh
   ```

3. **Update your training script** to use the filtered CSV:
   ```bash
   export DATA_META="/path/to/your/dataset_filtered.csv"
   ```

### Option 2: Run Python Script Directly

```bash
python filter_videos_by_frame_count.py \
    --input_csv /path/to/dataset.csv \
    --output_csv /path/to/dataset_filtered.csv \
    --data_root /path/to/videos \
    --min_frames 161 \
    --sample_stride 2 \
    --target_n_frames 81 \
    --num_workers 16 \
    --report_file filtering_report.txt
```

## Parameters

- `--input_csv`: Path to your original dataset CSV
- `--output_csv`: Path where filtered CSV will be saved
- `--data_root`: Root directory for videos (if CSV has relative paths)
- `--min_frames`: Minimum frames required (default: 161)
- `--sample_stride`: Frame sampling stride (default: 2)
- `--target_n_frames`: Target frames to sample (default: 81)
- `--num_workers`: Parallel workers for faster processing (default: 8)
- `--report_file`: Optional detailed report file

## How It Calculates Minimum Frames

```
min_frames = (target_n_frames - 1) × sample_stride + 1
           = (81 - 1) × 2 + 1
           = 161 frames
```

## Example Output

```
============================================================================
Video Dataset Frame Count Filter
============================================================================
Total videos in input CSV: 10000

Processing videos...
100%|████████████████████████████████| 10000/10000 [05:23<00:00, 30.89it/s]

============================================================================
SUMMARY
============================================================================
Total videos:     10000
Kept:             8543 (85.4%)
Rejected:         1457 (14.6%)
============================================================================

Kept Videos - Frame Statistics:
  Original frames: min=161, max=1500, avg=453.2
  Sampled frames:  min=81, max=81, avg=81.0
  ✓ All videos will sample to 81 frames (GOOD for GPU performance!)
```

## Benefits After Filtering

✅ **Consistent tensor shapes** across all batches  
✅ **GPU optimizations enabled** (cuDNN autotuning, kernel fusion, Flash Attention)  
✅ **20-30% faster training** due to optimized GPU kernels  
✅ **More stable gradients** in distributed training  
✅ **Better memory utilization** (no fragmentation)  

## Different Frame Requirements?

If you change your training settings, recalculate the minimum:

| Config | Min Frames Needed |
|--------|-------------------|
| 81 frames, stride 2 | 161 frames |
| 81 frames, stride 3 | 241 frames |
| 65 frames, stride 2 | 129 frames |
| 49 frames, stride 2 | 97 frames |
| 17 frames, stride 2 | 33 frames |

Formula: `(n_frames - 1) × stride + 1`

## Troubleshooting

**"Too many videos rejected!"**
- Consider reducing `VIDEO_SAMPLE_N_FRAMES` in your training config
- Or increase `VIDEO_SAMPLE_STRIDE` (but this skips more frames)
- Or collect longer videos for your dataset

**"Script is slow"**
- Increase `--num_workers` (default is 8, try 16 or 32)
- Make sure you have `decord` installed (faster than cv2)

**"File not found errors"**
- Check that `DATA_ROOT` is set correctly
- Verify video file paths in your CSV match actual files

## CSV Format

Expected CSV format (with header):
```csv
file_path,control_file_path,text,type
/path/to/video1.mp4,/path/to/video1.mp4,,video
/path/to/video2.mp4,/path/to/video2.mp4,optional text,video
```

The script preserves all columns and only filters rows.

