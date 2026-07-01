# TF-OVS

Reproducibility code for the TF-OVS benchmark: a fixed-protocol evaluation
harness for training-free open-vocabulary segmentation.

The repository intentionally contains only code and lightweight configuration
needed to reproduce the benchmark pipeline. It does not include raw datasets,
model checkpoints, generated prediction maps, paper drafts, rendered figures,
or machine-specific run artifacts.

## Contents

- `src/tf_ovos/`: core dataset, adapter, evaluation, sharding, merging, and
  summarization code.
- `configs/`: benchmark datasets, method registry entries, and vocabulary
  files.
- `scripts/`: dataset preparation, official-config preparation, prediction
  post-processing, diagnostic-table generation, and analysis scripts.
- `tests/`: unit and smoke tests for the reproducible harness.

## Setup

```bash
conda env create -f environment.yml
conda activate tf-ovs
python -m pip install -e .[dev,tables]
```

For a minimal environment without conda:

```bash
python -m pip install -e .[dev,tables]
```

Optional method adapters can require additional third-party repositories,
CUDA-compatible PyTorch builds, and public model checkpoints. Keep those
external repositories outside this repo, for example under `external/` or as
sibling folders.

## Verify The Harness

```bash
python -m pytest tests
python scripts/smoke_test.py
python -m tf_ovos.check_ready
```

The tests and `smoke_test.py` use synthetic toy data. Missing real datasets and
model weights are expected before full reproduction.

## Data Preparation

Create ignored local workspace folders:

```bash
bash scripts/prepare_workspace.sh
```

Download standard datasets with the provided helper scripts where licenses and
network access permit:

```bash
bash scripts/dl_voc20.sh
bash scripts/dl_context.sh
bash scripts/dl_ade20k.sh
bash scripts/dl_coco_stuff.sh
```

Then build manifests:

```bash
python scripts/prepare_manifests.py --dataset all
```

Dataset manifests are JSONL files under `data/manifests/`. A class-aware row has
the form:

```json
{"image_id": "0001", "image_path": "../raw/images/0001.jpg", "mask_path": "../raw/labels/0001.png"}
```

External mask-only targets can omit semantic labels.

## Core Pipeline

Run one adapter on one dataset:

```bash
python -m tf_ovos.run_benchmark \
  --method debug_copy_gt \
  --dataset voc20 \
  --limit 20 \
  --num-shards 4 \
  --skip-existing
```

Run a single shard manually:

```bash
python -m tf_ovos.make_shards \
  --manifest data/manifests/voc20_val.jsonl \
  --num-shards 4 \
  --out-dir data/manifests/shards/voc20

python -m tf_ovos.run_method \
  --method debug_copy_gt \
  --manifest data/manifests/shards/voc20/part-000.jsonl \
  --out runs/debug_copy_gt/voc20/shards/part-000

python -m tf_ovos.merge_predictions \
  --manifest data/manifests/voc20_val.jsonl \
  --shards-dir runs/debug_copy_gt/voc20/shards \
  --out runs/debug_copy_gt/voc20/predictions.jsonl
```

Evaluate predictions:

```bash
python -m tf_ovos.eval \
  --manifest data/manifests/voc20_val.jsonl \
  --predictions runs/debug_copy_gt/voc20/predictions.jsonl \
  --task class-aware \
  --out runs/debug_copy_gt/voc20/metrics.json
```

Summarize completed runs:

```bash
python -m tf_ovos.summarize_results --method debug_copy_gt --out-dir runs/tables
```

## Analysis Reproduction

After official method outputs and logs are available under `runs/`, reproduce
tables and diagnostics with:

```bash
python scripts/postprocess_official_predictions.py --help
python scripts/generate_diagnostic_tables.py --help
python scripts/generate_exploratory_light_metrics.py --help
python scripts/analyze_official_results.py
python scripts/build_e2_filled_table.py
```

Generated outputs are written to ignored `runs/analysis/` paths.

## Repository Hygiene

The following are intentionally ignored and should not be committed:

- raw datasets and manifests under `data/`
- model checkpoints and weights
- third-party method repositories
- generated predictions, logs, tables, and figures under `runs/`
- paper drafts and local presentation/figure editing artifacts

