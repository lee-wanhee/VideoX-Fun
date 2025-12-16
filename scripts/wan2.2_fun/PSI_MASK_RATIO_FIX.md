# PSI Control Extractor - Mask Ratio Fix

## Problem

The `psi_control_extractor.py` was failing with this error:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [0, 5, -1] 
because the unspecified dimension size -1 can be any value and is ambiguous
```

This occurred at line 3774 in `psi_predictor.py`:
```python
pos_after_seq_idx_tokens = pos_after_seq.reshape(n_pred_seq, n_tokens_per_patch, -1)
```

The error happened because `n_pred_seq=0`, meaning PSI found **no sequences to predict**.

## Root Cause

The issue was in how masks were created for the two input frames:

### Before (Broken):
```python
# Both frames used the SAME mask_ratio (default 0.0)
unmask_indices_rgb = self.create_mask_indices(
    self.mask_ratio,  # 0.0 = fully visible
    2,  # 2 frames
    seed=self.seed + b
)
```

With the prompt `"rgb0,rgb1->rgb1"`:
- **rgb0** (conditioning frame): Fully visible ✓
- **rgb1** (prediction frame): Fully visible ✗ **PROBLEM!**

When the prediction target `rgb1` is fully visible (unmasked), PSI thinks there's nothing to predict because it already has all the information. This results in `n_pred_seq=0`.

### After (Fixed):
```python
# Frame 0 (conditioning): fully visible
unmask_indices_frame0 = self.create_mask_indices(
    mask_ratio=self.mask_ratio_cond,  # Default: 0.0 (fully visible)
    num_frames=1,
    seed=self.seed + b
)

# Frame 1 (prediction target): heavily masked
unmask_indices_frame1 = self.create_mask_indices(
    mask_ratio=self.mask_ratio_pred,  # Default: 0.9 (90% masked)
    num_frames=1, 
    seed=self.seed + b + 1
)

unmask_indices_rgb = unmask_indices_frame0 + unmask_indices_frame1
```

Now:
- **rgb0**: Fully visible (0% masked) - provides conditioning context
- **rgb1**: 90% masked - PSI knows it needs to predict/extract features for this frame

## Changes Made

1. **Added new parameters** to `__init__`:
   - `mask_ratio_cond: float = 0.0` - Mask ratio for conditioning frame
   - `mask_ratio_pred: float = 0.9` - Mask ratio for prediction frame
   - Deprecated old `mask_ratio` parameter

2. **Updated mask creation** in `forward()`:
   - Create separate masks for conditioning and prediction frames
   - Added explanatory comments about why this is necessary

3. **Added documentation** explaining the importance of masking the prediction frame

## Why This Matters

In PSI's parallel feature extraction:
- **Conditioning frames** should be mostly/fully visible to provide context
- **Prediction frames** must be partially masked so PSI knows to extract features for them
- Without masking the prediction target, PSI's prompt parser finds no work to do

## Comparison with Working Test

The working `parallel_feature_test.py` had this correct all along:

```python
# Frame 0: Fully visible
unmask_indices_frame0 = create_mask_indices(mask_ratio=0.0, seed=seed)

# Frame 1: 90% masked ← KEY DIFFERENCE
unmask_indices_frame1 = create_mask_indices(mask_ratio=0.9, seed=seed + 1)
```

## Configuration

You can now control masking independently:

```python
extractor = PSIControlFeatureExtractor(
    model_name=...,
    quantizer_name=...,
    mask_ratio_cond=0.0,  # Conditioning frame visibility
    mask_ratio_pred=0.9,  # Prediction frame masking (higher = more masked)
    ...
)
```

**Recommended settings:**
- `mask_ratio_cond=0.0` (fully visible conditioning)
- `mask_ratio_pred=0.9` (90% masked prediction target)

## Testing

The fix has been applied. Test with:

```bash
python scripts/wan2.2_fun/psi_control_extractor.py
```

This should now work without the reshape error.

