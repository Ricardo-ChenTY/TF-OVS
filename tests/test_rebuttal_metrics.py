import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tf_ovos.metrics import accumulate_confusion, evaluate_semantic, load_label_map, dense_region_statistics
from tf_ovos.result_sources import read_unique_diagnostics


@pytest.mark.parametrize("bad", [-1, 255, 65535])
def test_undeclared_ids_cannot_become_correct_predictions(bad):
    with pytest.raises(ValueError, match="Unexpected prediction"):
        accumulate_confusion(np.array([[bad]]), np.array([[2]]), 3)


def test_distractors_and_rejection_count_as_fn_without_new_eval_classes():
    result = evaluate_semantic([np.array([[0, 0, 1, 1]])],
                               [np.array([[0, 2, 1, -1]])], 2,
                               num_candidates=3, rejection_label=-1)
    assert result.per_class_iou == [0.5, 0.5]
    assert result.miou == result.pixel_accuracy == result.mean_class_accuracy == 0.5
    # Candidate 2 remains a separate spatial region and can be localized-wrong.
    stats = dense_region_statistics(np.array([[2, 2]]), np.array([[0, 0]]), 2, num_candidates=3)
    assert stats.counts(0.5) == (1, 0)


def test_all_rejected_keeps_gt_support_and_undefined_conditional_rate():
    result = evaluate_semantic([np.array([[0, 1]])], [np.array([[-1, -1]])], 2, rejection_label=-1)
    assert result.per_class_iou == [0, 0]
    assert result.gt_class_regions == 2
    assert result.ziou_gt_supported == result.never_predicted_class_rate == 1
    assert result.mcmr_05 is None
    assert result.coverage_at_05 == 0


def test_best_iou_matching_and_ignore_pixels():
    gt = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 255, 255]])
    pred = np.array([[1, 1, 2, 2, 1, 1, 1, 1, 2, 2]])
    stats = dense_region_statistics(pred, gt, 3)
    assert stats.best_ids.tolist() == [2, 1]
    assert stats.best_ious.tolist() == pytest.approx([0.5, 4/6])
    assert stats.counts(0.5) == (2, 1)


def test_palette_16bit_and_rgb(tmp_path):
    p = tmp_path / 'palette.png'
    image = Image.new('P', (1, 1), 1)
    palette = [0] * 768
    palette[3] = 255  # PIL convert(L) would turn this index into luminance 76.
    image.putpalette(palette)
    image.save(p)
    assert load_label_map(str(p))[0, 0] == 1
    p = tmp_path / 'large.png'
    Image.fromarray(np.array([[458, 846, 65535]], dtype=np.uint16)).save(p)
    assert load_label_map(str(p)).tolist() == [[458, 846, 65535]]
    p = tmp_path / 'rgb.png'
    Image.new('RGB', (1, 1)).save(p)
    with pytest.raises(ValueError, match='indexed label'):
        load_label_map(str(p))


def test_count_mismatch_and_signed_resize():
    with pytest.raises(ValueError, match='counts differ'):
        evaluate_semantic([np.array([[0]])], [], 1)
    r = evaluate_semantic([np.zeros((2, 2), dtype=int)], [np.array([[-1]])], 1, rejection_label=-1)
    assert r.miou == 0 and r.gt_class_regions == 1


def test_union_and_gt_supported_denominators_are_separate():
    r = evaluate_semantic([np.array([[0, 0]])], [np.array([[0, 1]])], 2)
    assert r.ziou_gt_supported == 0
    assert r.ziou_union_supported == 0.5


def test_diagnostic_and_evaluator_agree_and_missing_is_error(tmp_path):
    spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).parents[1] / 'scripts/generate_diagnostic_tables.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    gt = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    pred = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    Image.fromarray(gt).save(tmp_path / 'gt.png')
    Image.fromarray(pred).save(tmp_path / 'a.png')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{"image_id":"a","image_path":"unused.jpg","mask_path":"gt.png"}\n')
    ds = module.DatasetSpec('toy', manifest, tmp_path/'vocab', 2, 255, -1)
    args = dict(method='toy', pred_dir=tmp_path, spec=ds, labels=['a', 'b'], include_biou=False, max_images=None, max_biou_regions=0)
    summary, _ = module._analyse_method_dataset(**args)
    expected = evaluate_semantic([gt], [pred.astype(int)-1], 2, rejection_label=-1)
    assert summary['mcmr_at_05'] == expected.mcmr_05
    assert summary['localized_pairs_at_05'] == expected.localized_pairs_at_05
    # Every GT pixel rejected still contributes to GT-supported class counts.
    Image.fromarray(np.zeros_like(pred)).save(tmp_path / 'a.png')
    summary, _ = module._analyse_method_dataset(**args)
    assert summary['n_valid_classes'] == 2
    assert summary['never_predicted_class_rate'] == 1
    assert summary['gt_region_naming_top1'] == 0
    (tmp_path / 'a.png').unlink()
    with pytest.raises(FileNotFoundError):
        module._analyse_method_dataset(**args)


def test_source_dedup_and_conflict(tmp_path):
    paths = [tmp_path/'a.csv', tmp_path/'b.csv']
    for p in paths:
        p.write_text('method,dataset,mcmr_at_05\na,x,0.5\n')
    assert len(read_unique_diagnostics(paths)) == 1
    paths[1].write_text('method,dataset,mcmr_at_05\na,x,0.6\n')
    with pytest.raises(ValueError, match='Conflicting'):
        read_unique_diagnostics(paths)


def test_center_crop_projection_preserves_position_and_rejects_unseen_border():
    import sys
    spec = importlib.util.spec_from_file_location('crop_diagnostics', Path(__file__).parents[1] / 'scripts/generate_diagnostic_tables.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    gt = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [1, 2]], dtype=np.int32)
    restored, full_gt = module._align_geometry(pred, gt, 'project-center-crop')
    assert np.array_equal(full_gt, gt)
    assert restored.tolist() == [[-1, 1, 2, -1], [-1, 1, 2, -1]]
    result = evaluate_semantic([full_gt], [restored], 4, rejection_label=-1)
    assert result.miou == 0.5 and result.gt_class_regions == 4
    native_pred, cropped_gt = module._align_geometry(pred, gt, 'short-side-center-crop')
    assert np.array_equal(native_pred, cropped_gt)
