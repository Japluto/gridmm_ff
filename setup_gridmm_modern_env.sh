#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-gridmm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "Activating ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing PyTorch with CUDA 12.8 wheels"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing GridMM Python dependencies"
python -m pip install \
  jsonlines==2.0.0 \
  tqdm==4.62.0 \
  easydict==1.9 \
  Shapely==1.7.1 \
  h5py==3.11.0 \
  networkx==2.5.1 \
  numpy==1.26.4 \
  tensorboardX==2.4.1 \
  transformers==4.31.0 \
  tokenizers==0.13.3 \
  pillow==10.4.0 \
  protobuf==3.20.3 \
  six \
  jinja2 \
  sympy \
  progressbar2 \
  opencv-python \
  imutils

cat <<EOF

Environment ${ENV_NAME} is ready.

Next steps:
1. Rebuild / expose MatterSim into this env using Python 3.10.
2. Run:
   conda activate ${ENV_NAME}
   cd ${REPO_ROOT}/map_nav_src
   bash scripts/run_r2r.sh test

Optional:
- If you have a local BERT directory, set BERT_BASE_DIR before running.
- If you want to reuse your old env name, create a new env first and only swap after verifying torch.cuda works.
EOF
