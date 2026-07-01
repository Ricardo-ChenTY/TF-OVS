#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p \
  data/raw \
  data/manifests \
  data/manifests/shards \
  external \
  runs \
  weights

echo "Prepared Training-Free Open-Vocabulary Segmentation workspace directories under ${repo_root}"
