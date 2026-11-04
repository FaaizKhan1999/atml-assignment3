#!/usr/bin/env bash
# Install dependencies and record the environment. Run from anywhere:
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
PLATFORM=$(python -c "from common.paths import detect_platform; print(detect_platform())")
mkdir -p results/env
python -m pip freeze > "results/env/${PLATFORM}-pip-freeze.txt"
python - <<'PY'
import torch, transformers
from common.paths import get_paths
p = get_paths()
print(f"platform={p.platform} scratch={p.scratch}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
if not torch.cuda.is_available():
    print("WARNING: no GPU visible (enable a GPU accelerator for training/evaluation)")
PY
