# Reproducing the revised benchmark

The Python package is `tf_ovos`. Run commands from the repository root after
`python -m pip install -e .[dev,tables]`.

## 1. Check the code and replay the reported statistics

```bash
python -m pytest tests
python scripts/smoke_test.py
python scripts/verify_expanded_e4_code.py
python scripts/reproduce_published_statistics.py --out runs/replayed-statistics
```

Choose an output directory that does not already exist. The replay uses NumPy
and the included measurement records and confusion matrices. It recomputes 25
latency rows, 10 paired DINO overhead rows, and four CLIPtrase score rows, and
checks the published numeric values at absolute tolerance `1e-12`. The E4 source
check compares 36 runtime files with the frozen protocols. Neither command runs
model inference.

The DINO geometry tests need the optional `transformers` dependency. Install
`transformers==4.36.2` to run those tests as well.

## 2. Prepare data and evaluate saved predictions

Use the dataset helpers and manifests described in the [main README](../../README.md#data-preparation).
Evaluation labels are zero-based. Compact semantic targets use void 255;
Context-459 and ADE-847 use uint16 labels with void 65535. ADE-847 IDs come from
[`ade20k847_official_ids.json`](../../configs/label_maps/ade20k847_official_ids.json).

For example, evaluate an ADE-847 prediction manifest with:

```bash
python -m tf_ovos.eval \
  --manifest data/manifests/ade20k847_val.jsonl \
  --predictions runs/METHOD/ade847/predictions.jsonl \
  --task semantic --num-classes 847 --void-label 65535 \
  --out runs/METHOD/ade847/metrics.json
```

Semantic mIoU averages classes with nonzero union. Extra candidate predictions
count as errors on valid base-class pixels in a fixed-GT experiment. The
common-GT diagnostic matches each image--GT-class unit by maximum IoU and reports
coverage alongside conditional MCMR. Native CLIPtrase uses a 336-pixel center
crop and requires its own geometry-aware scoring command:
`python scripts/rescore_cliptrase_native_crop.py --help`.

Fixed-GT vocabulary construction is in
[`candidate_expansion.py`](../../src/tf_ovos/candidate_expansion.py). The reported
curves use candidate seed 0. The saved-output rescorer is
[`rescore_candidate_only_expansion.py`](../../scripts/rescore_candidate_only_expansion.py).
Native methods retain their declared settings. The ResCLIP config generator
includes its released VOC `slide_crop=0` setting.

## 3. Run the matched E4 inference configurations

Follow the [GPU setup and commands](../rebuttal/reproduction/README.md).
It includes the recorded environments, external model revisions and hashes,
CorrCLIP source patch, and an editable local path mapping. The portable entry is:

```bash
python scripts/reproduce_e4.py prepare --group proposal \
  --config /path/to/local-paths.json --run /path/to/new-proposal-run
python scripts/reproduce_e4.py run --run /path/to/new-proposal-run \
  --method sam_amg_clip --dataset voc20 --stop-after 1
```

Run the configurations serially on a compatible Linux/NVIDIA system. Remove
`--stop-after 1` to complete the selected 1,000-image method/target job. Each of
the five configurations uses the same 4,000 measured images across four targets.
The runner compares new raw prediction-array hashes with the published records.
Full-quality mIoU remains separate from timing-sample mIoU. The observed software
inventories describe the tested server environment rather than a fresh-install
lockfile for every native method.

## Source

These files were selected from
[`TF-OVOS` at `2b15d76`](https://github.com/Ricardo-ChenTY/TF-OVOS/tree/2b15d76e6250e8a2ffacf7c06b01e7043d8ae9f7).
The [import manifest](import_manifest.json) records the full source commit and
file hashes. Frozen measurements, protocols, and runtime snapshots retain their
original hashes. The full experiment archive remains in the source repository.
