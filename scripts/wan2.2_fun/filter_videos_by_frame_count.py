#!/usr/bin/env python3
"""
Filter video dataset CSV to only include videos with sufficient frames.

Usage:
    python filter_videos_by_frame_count.py \
        --input_csv /path/to/input.csv \
        --output_csv /path/to/output_filtered.csv \
        --min_frames 161 \
        --sample_stride 2 \
        --data_root /path/to/video/directory
"""

import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("WARNING: decord not available, falling back to cv2")
    import cv2


def get_video_frame_count(video_path):
    """Get the total number of frames in a video."""
    if not os.path.exists(video_path):
        return -1, f"File not found: {video_path}"
    
    try:
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            frame_count = len(vr)
            return frame_count, None
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return -1, f"Cannot open video: {video_path}"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count, None
    except Exception as e:
        return -1, f"Error reading {video_path}: {str(e)}"


def process_video_row(row, data_root, min_frames, sample_stride, target_n_frames):
    """Process a single CSV row and determine if video meets requirements."""
    file_path = row['file_path']
    
    # Handle absolute vs relative paths
    if data_root and not os.path.isabs(file_path):
        video_path = os.path.join(data_root, file_path)
    else:
        video_path = file_path
    
    frame_count, error = get_video_frame_count(video_path)
    
    if error:
        return {
            'keep': False,
            'row': row,
            'frame_count': frame_count,
            'reason': error
        }
    
    # Calculate how many frames we can actually sample
    max_possible_frames = (frame_count - 1) // sample_stride + 1
    actual_n_frames = min(target_n_frames, max_possible_frames)
    
    # Determine if video is acceptable
    if frame_count >= min_frames:
        keep = True
        reason = f"OK ({frame_count} frames -> {actual_n_frames} sampled frames)"
    else:
        keep = False
        reason = f"Too short ({frame_count} frames, need {min_frames})"
    
    return {
        'keep': keep,
        'row': row,
        'frame_count': frame_count,
        'actual_n_frames': actual_n_frames,
        'reason': reason
    }


def filter_dataset_csv(
    input_csv,
    output_csv,
    min_frames=161,
    sample_stride=2,
    target_n_frames=81,
    data_root=None,
    num_workers=8,
    report_file=None
):
    """
    Filter a video dataset CSV to only include videos with sufficient frames.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output filtered CSV file
        min_frames: Minimum number of frames required in original video
        sample_stride: Frame sampling stride
        target_n_frames: Target number of frames to sample
        data_root: Root directory for video files (if paths in CSV are relative)
        num_workers: Number of parallel workers for video processing
        report_file: Optional path to save detailed report
    """
    print("=" * 80)
    print("Video Dataset Frame Count Filter")
    print("=" * 80)
    print(f"Input CSV:      {input_csv}")
    print(f"Output CSV:     {output_csv}")
    print(f"Data Root:      {data_root if data_root else '(using absolute paths)'}")
    print(f"Min Frames:     {min_frames}")
    print(f"Sample Stride:  {sample_stride}")
    print(f"Target Frames:  {target_n_frames}")
    print(f"Workers:        {num_workers}")
    print("=" * 80)
    
    # Read input CSV
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"\nTotal videos in input CSV: {len(rows)}")
    
    # Process videos in parallel
    print("\nProcessing videos...")
    process_fn = partial(
        process_video_row,
        data_root=data_root,
        min_frames=min_frames,
        sample_stride=sample_stride,
        target_n_frames=target_n_frames
    )
    
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, rows),
                total=len(rows),
                desc="Checking videos"
            ))
    else:
        results = [process_fn(row) for row in tqdm(rows, desc="Checking videos")]
    
    # Separate kept and rejected videos
    kept_videos = [r for r in results if r['keep']]
    rejected_videos = [r for r in results if not r['keep']]
    
    # Write filtered CSV
    print(f"\nWriting filtered CSV to: {output_csv}")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in kept_videos:
            writer.writerow(result['row'])
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos:     {len(rows)}")
    print(f"Kept:             {len(kept_videos)} ({len(kept_videos)/len(rows)*100:.1f}%)")
    print(f"Rejected:         {len(rejected_videos)} ({len(rejected_videos)/len(rows)*100:.1f}%)")
    print("=" * 80)
    
    # Frame count statistics for kept videos
    if kept_videos:
        frame_counts = [r['frame_count'] for r in kept_videos]
        sampled_counts = [r['actual_n_frames'] for r in kept_videos]
        print("\nKept Videos - Frame Statistics:")
        print(f"  Original frames: min={min(frame_counts)}, max={max(frame_counts)}, avg={sum(frame_counts)/len(frame_counts):.1f}")
        print(f"  Sampled frames:  min={min(sampled_counts)}, max={max(sampled_counts)}, avg={sum(sampled_counts)/len(sampled_counts):.1f}")
        all_same_length = len(set(sampled_counts)) == 1
        if all_same_length:
            print(f"  ✓ All videos will sample to {sampled_counts[0]} frames (GOOD for GPU performance!)")
        else:
            print(f"  ⚠ Videos will sample to {len(set(sampled_counts))} different frame counts (may hurt GPU performance)")
            print(f"    Frame counts: {sorted(set(sampled_counts))}")
    
    # Save detailed report if requested
    if report_file:
        print(f"\nWriting detailed report to: {report_file}")
        with open(report_file, 'w') as f:
            f.write("VIDEO DATASET FILTERING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Input CSV:      {input_csv}\n")
            f.write(f"Output CSV:     {output_csv}\n")
            f.write(f"Min Frames:     {min_frames}\n")
            f.write(f"Sample Stride:  {sample_stride}\n")
            f.write(f"Target Frames:  {target_n_frames}\n\n")
            
            f.write(f"Total videos:   {len(rows)}\n")
            f.write(f"Kept:           {len(kept_videos)}\n")
            f.write(f"Rejected:       {len(rejected_videos)}\n\n")
            
            if rejected_videos:
                f.write("\nREJECTED VIDEOS:\n")
                f.write("-" * 80 + "\n")
                for r in rejected_videos:
                    f.write(f"{r['row']['file_path']}\n")
                    f.write(f"  Reason: {r['reason']}\n\n")
    
    # Show sample of rejected videos
    if rejected_videos:
        print("\nSample of rejected videos (first 10):")
        for i, r in enumerate(rejected_videos[:10]):
            print(f"  {i+1}. {r['row']['file_path']}")
            print(f"     {r['reason']}")
    
    print("\n" + "=" * 80)
    print(f"✓ Filtered dataset saved to: {output_csv}")
    if report_file:
        print(f"✓ Detailed report saved to: {report_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter video dataset CSV by frame count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter with default settings (161 min frames, stride=2, target=81 frames)
  python filter_videos_by_frame_count.py \\
      --input_csv dataset.csv \\
      --output_csv dataset_filtered.csv \\
      --data_root /path/to/videos
  
  # Custom frame requirements
  python filter_videos_by_frame_count.py \\
      --input_csv dataset.csv \\
      --output_csv dataset_filtered.csv \\
      --min_frames 200 \\
      --sample_stride 3 \\
      --target_n_frames 65 \\
      --num_workers 16
        """
    )
    
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to output filtered CSV file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for video files (if CSV has relative paths)')
    parser.add_argument('--min_frames', type=int, default=161,
                        help='Minimum number of frames in original video (default: 161)')
    parser.add_argument('--sample_stride', type=int, default=2,
                        help='Frame sampling stride (default: 2)')
    parser.add_argument('--target_n_frames', type=int, default=81,
                        help='Target number of frames to sample (default: 81)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--report_file', type=str, default=None,
                        help='Path to save detailed filtering report (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        print(f"ERROR: Input CSV not found: {args.input_csv}")
        return 1
    
    if args.data_root and not os.path.exists(args.data_root):
        print(f"ERROR: Data root directory not found: {args.data_root}")
        return 1
    
    # Calculate minimum frames needed
    calculated_min = (args.target_n_frames - 1) * args.sample_stride + 1
    if args.min_frames < calculated_min:
        print(f"WARNING: min_frames ({args.min_frames}) is less than required ({calculated_min})")
        print(f"         for {args.target_n_frames} frames with stride {args.sample_stride}")
        print(f"         Setting min_frames to {calculated_min}")
        args.min_frames = calculated_min
    
    # Run filtering
    filter_dataset_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        min_frames=args.min_frames,
        sample_stride=args.sample_stride,
        target_n_frames=args.target_n_frames,
        data_root=args.data_root,
        num_workers=args.num_workers,
        report_file=args.report_file
    )
    
    return 0


if __name__ == '__main__':
    exit(main())

