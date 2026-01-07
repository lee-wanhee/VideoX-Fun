#!/usr/bin/env python3
"""
Simple Dataset Creator for VideoX-Fun

Creates a filtered CSV metadata file from a directory of videos.
Automatically filters out videos that are too short for training.

Usage:
    python create_dataset.py /path/to/videos
    python create_dataset.py /path/to/videos --n_frames 81 --stride 2
    python create_dataset.py /path/to/videos -o my_dataset.csv
"""

import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def get_frame_count(video_path):
    """Get the total number of frames in a video."""
    # Import inside function to work properly with multiprocessing spawn
    try:
        from decord import VideoReader, cpu
        decord_available = True
    except ImportError:
        decord_available = False
    
    video_path = str(video_path)  # Ensure string path
    if not os.path.exists(video_path):
        return -1
    try:
        if decord_available:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            return len(vr)
        else:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return -1
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
    except Exception as e:
        return -1


def process_video(args):
    """Check if a video has enough frames."""
    video_path, min_frames = args
    frame_count = get_frame_count(video_path)
    return str(video_path), frame_count, frame_count >= min_frames


def create_dataset(
    video_dir,
    output_csv=None,
    n_frames=81,
    stride=2,
    text="a video",
    workers=16,
    extensions=('.mp4', '.avi', '.mov', '.mkv'),
):
    """
    Create a filtered dataset CSV from a directory of videos.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Output CSV path (auto-generated if None)
        n_frames: Number of frames to sample during training
        stride: Frame sampling stride
        text: Text description for all videos
        workers: Number of parallel workers
        extensions: Video file extensions to include
    """
    video_dir = Path(video_dir).resolve()
    
    # Calculate minimum frames needed
    min_frames = (n_frames - 1) * stride + 1
    
    # Auto-generate output path
    if output_csv is None:
        # Use directory name, or parent if dir is "video"
        dir_name = video_dir.name
        if dir_name in ('video', 'videos'):
            dir_name = video_dir.parent.name
        
        repo_root = Path(__file__).resolve().parent.parent.parent
        datasets_dir = repo_root / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        output_csv = datasets_dir / f"{dir_name}_{n_frames}f.csv"
    else:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"""
================================================================================
VideoX-Fun Dataset Creator
================================================================================
Video directory:  {video_dir}
Output CSV:       {output_csv}
Target frames:    {n_frames}
Sample stride:    {stride}
Min frames:       {min_frames} (required in source video)
Text description: "{text}"
Workers:          {workers}
================================================================================
""")
    
    # Find all videos (recursively search subdirectories)
    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.rglob(f'*{ext}'))
        video_files.extend(video_dir.rglob(f'*{ext.upper()}'))
    video_files = sorted(set(video_files))
    
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print(f"ERROR: No videos found in {video_dir}")
        print(f"Supported formats: {extensions}")
        return None
    
    # Check frame counts in parallel
    print(f"\nChecking frame counts (min required: {min_frames})...")
    
    # Prepare args as tuples for multiprocessing
    video_args = [(str(v), min_frames) for v in video_files]
    
    if workers > 1:
        # Use spawn context for compatibility with decord
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(process_video, video_args),
                total=len(video_files),
                desc="Scanning videos"
            ))
    else:
        results = [process_video(args) for args in tqdm(video_args, desc="Scanning videos")]
    
    # Filter results
    kept = [(path, count) for path, count, ok in results if ok]
    rejected = [(path, count) for path, count, ok in results if not ok]
    
    # Write CSV
    print(f"\nWriting {len(kept)} videos to CSV...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'control_file_path', 'text', 'type'])
        writer.writeheader()
        for video_path, _ in kept:
            writer.writerow({
                'file_path': str(video_path),
                'control_file_path': str(video_path),
                'text': text,
                'type': 'video'
            })
    
    # Print summary
    pct_kept = len(kept) / len(video_files) * 100 if video_files else 0
    
    print(f"""
================================================================================
DONE!
================================================================================
Total videos:   {len(video_files)}
Kept:           {len(kept)} ({pct_kept:.1f}%)
Rejected:       {len(rejected)} (too short)
--------------------------------------------------------------------------------
Output CSV:     {output_csv}
================================================================================

Use this in your training script:
    export DATA_META="{output_csv}"
    export VIDEO_SAMPLE_N_FRAMES={n_frames}
    export VIDEO_SAMPLE_STRIDE={stride}
""")
    
    # Show rejected sample if any
    if rejected and len(rejected) <= 10:
        print("Rejected videos (too short):")
        for path, count in rejected:
            print(f"  {path.name}: {count} frames (need {min_frames})")
    elif rejected:
        print(f"Sample of rejected videos (showing 5 of {len(rejected)}):")
        for path, count in rejected[:5]:
            print(f"  {path.name}: {count} frames (need {min_frames})")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Create filtered dataset CSV from video directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-generates output name)
    python create_dataset.py /path/to/videos
    
    # Custom output path
    python create_dataset.py /path/to/videos -o my_dataset.csv
    
    # Different frame settings
    python create_dataset.py /path/to/videos --n_frames 49 --stride 3
    
    # Custom text description
    python create_dataset.py /path/to/videos --text "a person dancing"
"""
    )
    
    parser.add_argument('video_dir', help='Directory containing video files')
    parser.add_argument('-o', '--output', dest='output_csv', default=None,
                        help='Output CSV path (auto-generated if not specified)')
    parser.add_argument('--n_frames', type=int, default=81,
                        help='Number of frames to sample (default: 81)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Frame sampling stride (default: 2)')
    parser.add_argument('--text', default='a video',
                        help='Text description for all videos (default: "a video")')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.video_dir):
        print(f"ERROR: Directory not found: {args.video_dir}")
        return 1
    
    create_dataset(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        n_frames=args.n_frames,
        stride=args.stride,
        text=args.text,
        workers=args.workers,
    )
    
    return 0


if __name__ == '__main__':
    exit(main())

