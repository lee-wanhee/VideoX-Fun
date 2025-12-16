#!/usr/bin/env python3
"""
Create a CSV metadata file from a directory of videos.
This is useful when you have videos but no text descriptions.
"""

import os
import csv
import argparse
from pathlib import Path


def create_csv_from_video_dir(
    video_dir, 
    output_csv, 
    control_dir=None,
    text_value="",
    video_extensions=['.mp4', '.avi', '.mov', '.mkv'],
    use_relative_paths=True
):
    """
    Create a CSV file from a directory of videos.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Path to output CSV file
        control_dir: Directory containing control videos (if None, uses same as video_dir)
        text_value: Text description for all videos (default: empty string)
        video_extensions: List of video file extensions to include
        use_relative_paths: If True, use relative paths in CSV; if False, use absolute paths (default: False)
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
    
    # Create CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'control_file_path', 'text', 'type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for video_file in video_files:
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
    
    print(f"✓ Created CSV file: {output_csv}")
    print(f"  - Total entries: {len(video_files)}")
    print(f"  - Path type: {'relative' if use_relative_paths else 'absolute'}")
    print(f"  - Text field: {'(empty)' if text_value == '' else repr(text_value)}")
    print(f"\nFirst few entries:")
    
    # Show first few entries
    with open(output_csv, 'r') as f:
        lines = f.readlines()
        for line in lines[:6]:  # Header + 5 entries
            print(f"  {line.rstrip()}")
    
    if len(lines) > 6:
        print(f"  ... and {len(lines) - 6} more entries")


def main():
    # Get the repo root directory (3 levels up from this script)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_datasets_dir = repo_root / "datasets"
    
    parser = argparse.ArgumentParser(
        description='Create CSV metadata file from video directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create CSV with defaults (kinetics700)
  python create_csv_from_videos.py
  
  # Create CSV with relative paths
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
        default=str(default_datasets_dir / 'kinetics700.csv'),
        help=f'Path to output CSV file (default: {{repo}}/datasets/kinetics700.csv)'
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
    
    args = parser.parse_args()
    
    # Create datasets directory if it doesn't exist
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_csv_from_video_dir(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        control_dir=args.control_dir,
        text_value=args.text,
        video_extensions=args.extensions,
        use_relative_paths=args.relative
    )


if __name__ == '__main__':
    main()

