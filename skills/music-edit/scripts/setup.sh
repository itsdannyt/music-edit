#!/bin/bash
# One-time setup. Creates ~/.music-edit with its own Python, so nothing touches other projects.
# Downloads: PyTorch and friends (about 1 GB), SAM 2 code from Meta's GitHub, the SAM 2 tiny model (156 MB).
# Add --whisper for lyric timing (mlx-whisper; its model downloads on first use, about 1.6 GB).
# Add --isnet for the optional anime matte used to fix enclosed holes (about 170 MB).
set -euo pipefail
R="$HOME/.music-edit"; mkdir -p "$R/src" "$R/model"
PYB=""; for p in python3.12 python3.11 python3.10; do if command -v "$p" >/dev/null; then PYB="$p"; break; fi; done
[ -n "$PYB" ] || { echo "Need Python 3.10-3.12. PyTorch may not support newer versions yet."; exit 1; }
command -v ffmpeg >/dev/null || { echo "Need ffmpeg on PATH."; exit 1; }
[ -x "$R/venv/bin/python" ] || "$PYB" -m venv "$R/venv"
"$R/venv/bin/pip" -q install --upgrade pip
"$R/venv/bin/pip" -q install "torch>=2.5.1" "torchvision>=0.20.1" opencv-python-headless librosa soundfile scipy "scenedetect" numpy
SAM_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
if [ ! -d "$R/src/sam2" ]; then git clone -q https://github.com/facebookresearch/sam2.git "$R/src/sam2"; fi
git -C "$R/src/sam2" checkout -q "$SAM_COMMIT"
SAM2_BUILD_CUDA=0 "$R/venv/bin/pip" -q install "$R/src/sam2"
CK="$R/model/sam2.1_hiera_tiny.pt"; SHA=7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
[ -f "$CK" ] || curl -sfL -o "$CK" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
[ "$(shasum -a 256 "$CK" | cut -d' ' -f1)" = "$SHA" ] || { echo "SAM 2 model checksum mismatch"; rm -f "$CK"; exit 1; }
for a in "$@"; do
  case "$a" in
    --whisper) "$R/venv/bin/pip" -q install mlx-whisper ;;
    --isnet)
      "$R/venv/bin/pip" -q install onnxruntime
      I="$R/model/isnet-anime.onnx"; IS=f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99
      [ -f "$I" ] || curl -sfL -o "$I" https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx
      [ "$(shasum -a 256 "$I" | cut -d' ' -f1)" = "$IS" ] || { echo "isnet checksum mismatch: compare with a trusted copy"; exit 1; } ;;
  esac
done
"$R/venv/bin/python" -I -c "import torch,sam2,cv2,librosa,scenedetect;print('ok, torch',torch.__version__,'mps',torch.backends.mps.is_available())"
echo "Python for all tools: $R/venv/bin/python"
