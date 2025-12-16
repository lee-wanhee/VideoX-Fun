#!/usr/bin/env python3
"""
Generate validation configuration for training script from validation video list.

This reads a validation video list (created by create_csv_from_videos.py)
and generates shell script configuration that can be pasted into training scripts.
"""

import argparse
from pathlib import Path


def generate_validation_config(val_list_txt, prompts=None, num_samples=None):
    """
    Generate validation configuration for training script.
    
    Args:
        val_list_txt: Path to validation video list txt file
        prompts: List of prompts to use (if None, generates generic prompts)
        num_samples: Number of validation samples to use (if None, uses all)
    
    Returns:
        Shell script configuration string
    """
    # Read validation video paths
    with open(val_list_txt, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    # Limit number of samples if requested
    if num_samples is not None:
        video_paths = video_paths[:num_samples]
    
    n_videos = len(video_paths)
    
    # Generate prompts if not provided
    if prompts is None:
        # Use empty prompts for control-based training (no text descriptions)
        prompts = [""] * n_videos
    elif len(prompts) < n_videos:
        # Repeat prompts if we have more videos than prompts
        prompts = (prompts * ((n_videos // len(prompts)) + 1))[:n_videos]
    elif len(prompts) > n_videos:
        # Trim prompts if we have more prompts than videos
        prompts = prompts[:n_videos]
    
    # Generate shell script configuration
    config_lines = [
        "# Validation settings",
        "export VALIDATION_STEPS=1000",
        "export VALIDATION_PROMPTS=(",
    ]
    
    for prompt in prompts:
        config_lines.append(f'    "{prompt}"')
    
    config_lines.append(")")
    config_lines.append("export VALIDATION_PATHS=(")
    
    for video_path in video_paths:
        config_lines.append(f'    "{video_path}"')
    
    config_lines.append(")")
    
    return "\n".join(config_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate validation configuration for training script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config from validation list
  python generate_validation_config.py ../../datasets/handpicked_val_videos.txt
  
  # Use only first 5 videos
  python generate_validation_config.py ../../datasets/handpicked_val_videos.txt --num_samples 5
  
  # Provide custom prompts
  python generate_validation_config.py ../../datasets/handpicked_val_videos.txt \\
      --prompts "a person walking" "a person dancing" "a person jumping"
  
  # Save to file
  python generate_validation_config.py ../../datasets/handpicked_val_videos.txt > validation_config.sh
        """
    )
    
    parser.add_argument(
        'val_list_txt',
        type=str,
        help='Path to validation video list txt file'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        default=None,
        help='Custom prompts for validation (default: generic prompts)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of validation samples to use (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    # Generate configuration
    config = generate_validation_config(
        args.val_list_txt,
        prompts=args.prompts,
        num_samples=args.num_samples
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(config)
        print(f"✓ Validation configuration written to: {args.output}")
        print("\nYou can source this file in your training script:")
        print(f"  source {args.output}")
    else:
        print(config)
        print("\n# Copy the above lines into your training script")


if __name__ == '__main__':
    main()

