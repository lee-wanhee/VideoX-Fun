#!/bin/bash

# ============================================================================
# Filter Dataset CSV by Frame Count
# ============================================================================
# This script filters your video dataset CSV to only include videos with
# sufficient frames for training, matching your training configuration.
# ============================================================================

# Settings from train_control_lora_psi.sh
export VIDEO_SAMPLE_N_FRAMES=81
export VIDEO_SAMPLE_STRIDE=2

# Calculate minimum frames needed: (n_frames - 1) * stride + 1
export MIN_FRAMES=$((($VIDEO_SAMPLE_N_FRAMES - 1) * $VIDEO_SAMPLE_STRIDE + 1))

# Input/Output paths - UPDATE THESE FOR YOUR DATASET
export INPUT_CSV="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/handpicked.csv"
export OUTPUT_CSV="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/handpicked_filtered_${VIDEO_SAMPLE_N_FRAMES}frames.csv"
export DATA_ROOT=""  # Leave empty if CSV has absolute paths, otherwise set to your data directory

# Optional: save detailed report
export REPORT_FILE="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/filtering_report_${VIDEO_SAMPLE_N_FRAMES}frames.txt"

# Number of parallel workers (adjust based on your CPU cores)
export NUM_WORKERS=16

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo "============================================================================"
echo "Video Dataset Frame Count Filter"
echo "============================================================================"
echo "Configuration:"
echo "  Target frames:     $VIDEO_SAMPLE_N_FRAMES"
echo "  Sample stride:     $VIDEO_SAMPLE_STRIDE"
echo "  Min frames needed: $MIN_FRAMES"
echo ""
echo "Paths:"
echo "  Input CSV:         $INPUT_CSV"
echo "  Output CSV:        $OUTPUT_CSV"
echo "  Data root:         ${DATA_ROOT:-'(using absolute paths)'}"
echo "  Report file:       $REPORT_FILE"
echo ""
echo "Performance:"
echo "  Workers:           $NUM_WORKERS"
echo "============================================================================"
echo ""

# Check if input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    echo "Please update the INPUT_CSV variable in this script."
    exit 1
fi

# Check if data root exists (if specified)
if [ -n "$DATA_ROOT" ] && [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root directory not found: $DATA_ROOT"
    echo "Please update the DATA_ROOT variable in this script."
    exit 1
fi

# ============================================================================
# Run Filtering
# ============================================================================
echo "Starting filtering process..."
echo ""

python /scratch/m000063/users/wanhee/VideoX-Fun/scripts/wan2.2_fun/filter_videos_by_frame_count.py \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --min_frames $MIN_FRAMES \
    --sample_stride $VIDEO_SAMPLE_STRIDE \
    --target_n_frames $VIDEO_SAMPLE_N_FRAMES \
    --num_workers $NUM_WORKERS \
    --report_file "$REPORT_FILE" \
    ${DATA_ROOT:+--data_root "$DATA_ROOT"}

# ============================================================================
# Post-Filtering Instructions
# ============================================================================
echo ""
echo "============================================================================"
echo "Next Steps"
echo "============================================================================"
echo ""
echo "1. Review the filtering results above"
echo "2. Check the detailed report: $REPORT_FILE"
echo "3. Update your training script to use the filtered CSV:"
echo ""
echo "   In train_control_lora_psi.sh, change:"
echo "   export DATA_META=\"$OUTPUT_CSV\""
echo ""
echo "This will ensure all videos in your training batches have consistent"
echo "frame counts, leading to better GPU performance and more stable training."
echo "============================================================================"

