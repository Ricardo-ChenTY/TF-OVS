#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python -m tf_ovos.run_method --list-methods
python -m tf_ovos.check_ready
