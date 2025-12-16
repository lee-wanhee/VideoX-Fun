#!/usr/bin/env python3
"""
Test script for custom control extractor integration.

This script tests that:
1. The DummyControlFeatureExtractor works correctly
2. Input/output shapes are correct
3. The integration with training pipeline is functional
"""

import sys
import torch
import numpy as np

# Add project paths
import os
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path), 
    os.path.dirname(os.path.dirname(current_file_path)), 
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# Import the dummy control extractor from the training script
print("Importing DummyControlFeatureExtractor from train_control_lora.py...")

# We need to extract just the class, not run the whole script
exec(open('train_control_lora.py').read().split('def filter_kwargs')[0])

def test_dummy_control_extractor():
    """Test the DummyControlFeatureExtractor class."""
    print("\n" + "="*70)
    print("Testing DummyControlFeatureExtractor")
    print("="*70)
    
    # Configuration matching VAE defaults
    latent_channels = 16
    temporal_compression = 4
    spatial_compression = 8
    
    # Create model
    print(f"\nCreating DummyControlFeatureExtractor...")
    print(f"  - latent_channels: {latent_channels}")
    print(f"  - temporal_compression: {temporal_compression}")
    print(f"  - spatial_compression: {spatial_compression}")
    
    model = DummyControlFeatureExtractor(
        latent_channels=latent_channels,
        temporal_compression=temporal_compression,
        spatial_compression=spatial_compression,
    )
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = 17
    height, width = 512, 512
    
    print(f"\nCreating test input:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - num_frames: {num_frames}")
    print(f"  - resolution: {height}x{width}")
    
    # Input should be (B, F, C, H, W) in range [-1, 1]
    control_input = torch.randn(batch_size, num_frames, 3, height, width)
    control_input = control_input * 2 - 1  # Normalize to [-1, 1]
    
    print(f"\nInput shape: {control_input.shape}")
    print(f"Input range: [{control_input.min():.2f}, {control_input.max():.2f}]")
    
    # Run model
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = model(control_input)
    
    # Check output
    expected_shape = (
        batch_size, 
        latent_channels, 
        num_frames, 
        height // spatial_compression, 
        width // spatial_compression
    )
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Verify shape
    assert output.shape == expected_shape, \
        f"Shape mismatch! Got {output.shape}, expected {expected_shape}"
    
    # Verify dtype
    assert output.dtype == control_input.dtype, \
        f"Dtype mismatch! Got {output.dtype}, expected {control_input.dtype}"
    
    print("\n✓ All tests passed!")
    
    return True


def test_different_resolutions():
    """Test with different input resolutions."""
    print("\n" + "="*70)
    print("Testing Different Resolutions")
    print("="*70)
    
    model = DummyControlFeatureExtractor(
        latent_channels=16,
        spatial_compression=8,
    )
    model.eval()
    
    test_cases = [
        (256, 256, 9),    # Small, short
        (384, 384, 17),   # Medium, standard
        (512, 512, 17),   # Large, standard
        (768, 432, 13),   # Wide, medium
    ]
    
    for height, width, num_frames in test_cases:
        print(f"\nTesting {height}x{width}, {num_frames} frames...")
        
        control_input = torch.randn(1, num_frames, 3, height, width)
        control_input = (control_input - 0.5) * 2  # [-1, 1]
        
        with torch.no_grad():
            output = model(control_input)
        
        expected_shape = (1, 16, num_frames, height // 8, width // 8)
        assert output.shape == expected_shape, \
            f"Failed for {height}x{width}x{num_frames}! Got {output.shape}, expected {expected_shape}"
        
        print(f"  ✓ Output shape: {output.shape}")
    
    print("\n✓ All resolution tests passed!")
    return True


def test_from_pretrained():
    """Test the from_pretrained class method."""
    print("\n" + "="*70)
    print("Testing from_pretrained Method")
    print("="*70)
    
    # Test with non-existent path (should return randomly initialized model)
    print("\nTesting with non-existent checkpoint path...")
    model = DummyControlFeatureExtractor.from_pretrained(
        "/non/existent/path.pth",
        latent_channels=16,
        spatial_compression=8,
    )
    
    print("  ✓ Successfully created model with random initialization")
    
    # Test basic forward pass
    test_input = torch.randn(1, 9, 3, 256, 256) * 2 - 1
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  ✓ Forward pass successful: {output.shape}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# Custom Control Extractor Integration Tests")
    print("#"*70)
    
    try:
        # Run tests
        test_dummy_control_extractor()
        test_different_resolutions()
        test_from_pretrained()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nNext steps:")
        print("1. Implement your actual control extraction logic in custom_control_extractor.py")
        print("2. Test it with: python custom_control_extractor.py")
        print("3. Run training with --use_custom_control_extractor flag")
        print("\nSee README_CUSTOM_CONTROL.md for more details.")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED! ✗")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

