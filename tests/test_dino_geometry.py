import numpy as np
from PIL import Image
import pytest

transformers = pytest.importorskip("transformers", reason="DINO geometry tests require the optional transformers dependency")
BitImageProcessor = transformers.BitImageProcessor
from tf_ovos.dino_geometry import aligned_mask_grid


def test_outside_center_crop_cannot_select_tokens():
    processor = BitImageProcessor(size={'shortest_edge': 256}, crop_size={'height': 224, 'width': 224})
    mask = np.zeros((480, 640), dtype=bool)
    mask[:, :64] = True
    assert not aligned_mask_grid(mask, processor, (16, 16)).any()
    assert np.asarray(Image.fromarray(mask).resize((16, 16), Image.Resampling.NEAREST)).sum() == 32


def test_geometry_tracks_rgb_processor_on_asymmetric_inputs():
    for shape in [(480, 640), (640, 480), (256, 256)]:
        processor = BitImageProcessor(size={'shortest_edge': 256}, crop_size={'height': 224, 'width': 224})
        mask = np.zeros(shape, dtype=np.uint8)
        mask[shape[0] // 3:2 * shape[0] // 3, shape[1] // 4:shape[1] // 2] = 255
        rgb = Image.fromarray(np.repeat(mask[:, :, None], 3, axis=2))
        transformed = processor(images=rgb, return_tensors='np', do_rescale=False, do_normalize=False, resample=Image.Resampling.NEAREST)['pixel_values'][0, 0]
        expected = np.asarray(Image.fromarray(transformed.astype(np.uint8)).resize((16, 16), Image.Resampling.NEAREST)) > 0
        np.testing.assert_array_equal(aligned_mask_grid(mask > 0, processor, (16, 16)), expected)


def test_identity_geometry_preserves_spatial_location():
    processor = BitImageProcessor(do_resize=False, do_center_crop=False)
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:7, 11:15] = True
    np.testing.assert_array_equal(aligned_mask_grid(mask, processor, (16, 16)), mask)
