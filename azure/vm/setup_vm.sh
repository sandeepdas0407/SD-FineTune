#!/usr/bin/env bash
# Run ON THE VM. Installs the NVIDIA driver + tools (no reboot needed), then Python 3.11 venv + libs.
set -euo pipefail
cd "$(dirname "$0")/../.."   # ~/SD-FineTune

drv_ver() { dpkg -l | grep -oE 'nvidia-[a-z-]*-([0-9]+)-server' | grep -oE '[0-9]+' | head -1; }

if [ -z "$(drv_ver)" ]; then
  echo ">> Installing NVIDIA datacenter (server) driver"
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers install --gpgpu   # on Azure kernels: headless, prebuilt open kernel module
fi
VER=$(drv_ver)
if ! command -v nvidia-smi >/dev/null; then
  # --gpgpu install is headless: add nvidia-smi + libcuda (needed by PyTorch)
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "nvidia-utils-${VER}-server" "libnvidia-compute-${VER}-server"
fi
# module isn't auto-loaded by the headless package -> load now and on every boot
echo nvidia | sudo tee /etc/modules-load.d/nvidia.conf >/dev/null
sudo modprobe nvidia
nvidia-smi

# tmux for long jobs; build-essential: Triton (used by TRL 1.x fused loss kernel) JIT-compiles with gcc
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tmux build-essential >/dev/null
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r azure/vm/requirements-gpu.txt
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
