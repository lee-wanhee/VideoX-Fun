#!/usr/bin/env python3
"""
Create a CSV metadata file from a directory of videos.
This is useful when you have videos but no text descriptions.
Supports automatic train/validation split.
"""

import os
import csv
import argparse
import random
from pathlib import Path


def create_csv_from_video_dir(
    video_dir, 
    output_csv, 
    control_dir=None,
    text_value="",
    video_extensions=['.mp4', '.avi', '.mov', '.mkv'],
    use_relative_paths=True,
    val_split=0.0,
    val_output_csv=None,
    val_list_txt=None,
    seed=42
):
    """
    Create a CSV file from a directory of videos.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Path to output CSV file (for training data)
        control_dir: Directory containing control videos (if None, uses same as video_dir)
        text_value: Text description for all videos (default: empty string)
        video_extensions: List of video file extensions to include
        use_relative_paths: If True, use relative paths in CSV; if False, use absolute paths
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        val_output_csv: Path to validation CSV file (if None, auto-generated from output_csv)
        val_list_txt: Path to validation video list txt file (for training script)
        seed: Random seed for reproducible splits
    """
    video_dir = Path(video_dir)
    
    # If control_dir not specified, use same as video_dir
    if control_dir is None:
        control_dir = video_dir
    else:
        control_dir = Path(control_dir)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(sorted(video_dir.glob(f'*{ext}')))
    
    print(f"Found {len(video_files)} video files in {video_dir}")
    
    if len(video_files) == 0:
        print(f"Warning: No video files found with extensions {video_extensions}")
        return
    
    # Split into train and validation if requested
    if val_split > 0:
        random.seed(seed)
        random.shuffle(video_files)
        
        n_val = max(1, int(len(video_files) * val_split))  # At least 1 validation sample
        val_files = video_files[:n_val]
        train_files = video_files[n_val:]
        
        print(f"\nSplit data (seed={seed}):")
        print(f"  Training:   {len(train_files)} videos")
        print(f"  Validation: {len(val_files)} videos")
        
        # Auto-generate validation CSV path if not provided
        if val_output_csv is None:
            output_path = Path(output_csv)
            val_output_csv = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
        
        # Auto-generate validation list txt if not provided
        if val_list_txt is None:
            output_path = Path(output_csv)
            val_list_txt = output_path.parent / f"{output_path.stem}_val_videos.txt"
    else:
        train_files = video_files
        val_files = []
    
    # Helper function to write CSV
    def write_csv(video_list, csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['file_path', 'control_file_path', 'text', 'type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for video_file in video_list:
                # Get absolute or relative path
                if use_relative_paths:
                    video_path = video_file.name  # Just filename
                    control_path = video_file.name  # Same file for control
                else:
                    video_path = str(video_file.absolute())
                    control_path = str((control_dir / video_file.name).absolute())
                
                writer.writerow({
                    'file_path': video_path,
                    'control_file_path': control_path,
                    'text': text_value,
                    'type': 'video'
                })
    
    # Write training CSV
    write_csv(train_files, output_csv)
    print(f"\n✓ Created training CSV: {output_csv}")
    print(f"  - Total entries: {len(train_files)}")
    print(f"  - Path type: {'relative' if use_relative_paths else 'absolute'}")
    print(f"  - Text field: {'(empty)' if text_value == '' else repr(text_value)}")
    
    # Write validation CSV if we have validation data
    if val_files:
        write_csv(val_files, val_output_csv)
        print(f"\n✓ Created validation CSV: {val_output_csv}")
        print(f"  - Total entries: {len(val_files)}")
        
        # Write validation video list for training script
        with open(val_list_txt, 'w') as f:
            for video_file in val_files:
                if use_relative_paths:
                    video_path = str(video_dir / video_file.name)
                else:
                    video_path = str(video_file.absolute())
                f.write(f"{video_path}\n")
        
        print(f"\n✓ Created validation video list: {val_list_txt}")
        print(f"  - Total entries: {len(val_files)}")
        print(f"  - Use these paths in your training script's --validation_paths")
    
    print(f"\nFirst few training entries:")
    
    # Show first few entries
    with open(output_csv, 'r') as f:
        lines = f.readlines()
        for line in lines[:6]:  # Header + 5 entries
            print(f"  {line.rstrip()}")
    
    if len(lines) > 6:
        print(f"  ... and {len(lines) - 6} more entries")
    
    if val_files:
        print(f"\nFirst few validation videos:")
        with open(val_list_txt, 'r') as f:
            lines = f.readlines()
            for line in lines[:5]:
                print(f"  {line.rstrip()}")
        if len(lines) > 5:
            print(f"  ... and {len(lines) - 5} more")
    
    return train_files, val_files


def main():
    # Get the repo root directory (3 levels up from this script)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_datasets_dir = repo_root / "datasets"
    
    parser = argparse.ArgumentParser(
        description='Create CSV metadata file from video directory with optional train/val split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-name CSV based on directory name (handpicked -> handpicked.csv)
  python create_csv_from_videos.py /scratch/m000063/data/bvd2/handpicked/
  
  # With validation split (20% for validation)
  python create_csv_from_videos.py /scratch/m000063/data/bvd2/handpicked/ --val_split 0.2
  
  # Specify custom output name
  python create_csv_from_videos.py /path/to/videos my_dataset.csv
  
  # With relative paths (for when data and CSV are in same parent dir)
  python create_csv_from_videos.py /path/to/videos metadata.csv --relative
  
  # Use separate control video directory
  python create_csv_from_videos.py /path/to/videos metadata.csv --control_dir /path/to/controls
  
  # Add generic text to all videos
  python create_csv_from_videos.py /path/to/videos metadata.csv --text "a video"
        """
    )
    
    parser.add_argument(
        'video_dir',
        type=str,
        nargs='?',
        default='/ccn2/dataset/bvd2/processed/kinetics700/',
        help='Directory containing video files (default: /ccn2/dataset/bvd2/processed/kinetics700/)'
    )
    parser.add_argument(
        'output_csv',
        type=str,
        nargs='?',
        default=None,
        help='Path to output CSV file (default: auto-generated from directory name)'
    )
    parser.add_argument(
        '--control_dir',
        type=str,
        default=None,
        help='Directory containing control videos (default: same as video_dir)'
    )
    parser.add_argument(
        '--text',
        type=str,
        default='',
        help='Text description for all videos (default: empty string)'
    )
    parser.add_argument(
        '--relative',
        action='store_true',
        help='Use relative paths instead of absolute paths (default: absolute paths)'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv'],
        help='Video file extensions to include (default: .mp4 .avi .mov .mkv)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.0,
        help='Fraction of videos to use for validation (0.0 to 1.0, default: 0.0 = no split)'
    )
    parser.add_argument(
        '--val_csv',
        type=str,
        default=None,
        help='Path to validation CSV file (default: auto-generated)'
    )
    parser.add_argument(
        '--val_list',
        type=str,
        default=None,
        help='Path to validation video list txt file (default: auto-generated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible train/val split (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output CSV name if not provided
    if args.output_csv is None:
        video_dir_path = Path(args.video_dir)
        # Get the directory name (e.g., "handpicked" from "/path/to/handpicked/")
        dir_name = video_dir_path.name if video_dir_path.name else video_dir_path.parent.name
        args.output_csv = str(default_datasets_dir / f"{dir_name}.csv")
        print(f"Auto-generated output name: {args.output_csv}")
    
    # Create datasets directory if it doesn't exist
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_csv_from_video_dir(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        control_dir=args.control_dir,
        text_value=args.text,
        video_extensions=args.extensions,
        use_relative_paths=args.relative,
        val_split=args.val_split,
        val_output_csv=args.val_csv,
        val_list_txt=args.val_list,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

