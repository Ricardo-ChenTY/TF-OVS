# Frozen reproduction inputs

Use the [reproduction guide](../reproduction/README.md) for the supported commands.
This directory contains only the inputs needed by the statistics replay and the
five matched E4 configurations.

- [`reproduction/`](reproduction/README.md): GPU setup, model inventories, and the CorrCLIP runtime patch.
- [`artifacts/`](artifacts/): two frozen runtime protocols and the 20,000 timing records, each with a SHA256 checksum.
- [`evidence/e4_expanded_results_20261016/`](evidence/e4_expanded_results_20261016/): published latency means, bootstrap intervals, and quality/latency pairs.
- [`evidence/cliptrase_native_crop_20261015/`](evidence/cliptrase_native_crop_20261015/): four confusion matrices and the published score table.

The frozen protocols retain historical paths as provenance. The GPU entrypoint
maps those paths to user-provided directories before running.
