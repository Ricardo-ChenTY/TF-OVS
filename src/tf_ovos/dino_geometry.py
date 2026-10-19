"""Match SAM masks to the DINO processor's exact nearest-neighbor geometry."""
from __future__ import annotations
import numpy as np
from PIL import Image


def mask_grid_source_indices(shape, processor, grid_wh):
    """Apply RGB resize/crop once to encoded source coordinates.

    Nearest-neighbor transforms preserve coordinate IDs. Zero is reserved for
    processor padding, so pixels outside the image never select a mask token.
    """
    h, w = shape
    if h * w >= 2**24:
        raise ValueError('24-bit geometry map supports images smaller than 2**24 pixels')
    ids = np.arange(1, h * w + 1, dtype=np.uint32).reshape(h, w)
    rgb = np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=-1).astype(np.uint8)
    values = processor(images=Image.fromarray(rgb), return_tensors='np', do_rescale=False,
                       do_normalize=False, resample=Image.Resampling.NEAREST)['pixel_values'][0]
    spatial = Image.fromarray(values.transpose(1, 2, 0).astype(np.uint8))
    a = np.asarray(spatial.resize(grid_wh, Image.Resampling.NEAREST)).astype(np.int64)
    return a[..., 0] + (a[..., 1] << 8) + (a[..., 2] << 16) - 1


def aligned_mask_grid(mask, processor, grid_wh, source_indices=None):
    """Resize and center-crop labels as RGB, without normalization/rescaling."""
    if source_indices is None:
        source_indices = mask_grid_source_indices(mask.shape, processor, grid_wh)
    result = np.zeros(source_indices.shape, dtype=bool)
    valid = source_indices >= 0
    result[valid] = np.asarray(mask, dtype=bool).reshape(-1)[source_indices[valid]]
    return result
