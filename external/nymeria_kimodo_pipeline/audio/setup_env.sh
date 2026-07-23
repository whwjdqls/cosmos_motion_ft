#!/usr/bin/env bash
# Create the `audio` conda env for stages 2-4 (VAD + SAM-Audio separation + verify).
# Stage 1 (extract_audio.py) uses the existing `nymeria_plus` env (projectaria_tools)
# and needs nothing here.
#
# SAM-Audio weights are GATED: request access at
#   https://huggingface.co/facebook/sam-audio-large
# then `hf auth login` with a token before running stage 3.
set -euo pipefail

ENV=audio
CONDA=/home/jungbin_cho/miniforge3

source "$CONDA/etc/profile.d/conda.sh"
if ! conda env list | grep -q "/envs/$ENV"; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"

# GPU torch (match the cluster CUDA; cosmos env uses cu128).
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy soundfile huggingface_hub

# --- FireRedVAD (stage 2 + stage 4) -----------------------------------------
pip install fireredvad
mkdir -p /home/jungbin_cho/audio_models
hf download FireRedTeam/FireRedVAD \
  --local-dir /home/jungbin_cho/audio_models/FireRedVAD
# -> FIRERED_VAD_DIR=/home/jungbin_cho/audio_models/FireRedVAD/VAD  (default in vad.py)

# --- SAM-Audio (stage 3) -----------------------------------------------------
# Clone + install from source; weights are gated (see header).
SAM_DIR=/home/jungbin_cho/audio_models/sam-audio
if [ ! -d "$SAM_DIR" ]; then
  git clone https://github.com/facebookresearch/sam-audio "$SAM_DIR"
fi
pip install -e "$SAM_DIR"
# Patch SAM-Audio's ModelHubMixin override for huggingface_hub 1.x: it no longer passes
# proxies/resume_download to _from_pretrained, and dropped resume_download from
# snapshot_download. Make those kwargs optional + drop the removed snapshot_download arg.
BASE="$SAM_DIR/sam_audio/model/base.py"
sed -i 's/^        proxies: Optional\[Dict\],/        proxies: Optional[Dict] = None,/' "$BASE"
sed -i 's/^        resume_download: bool,/        resume_download: bool = False,/' "$BASE"
sed -i 's/^        force_download: bool,/        force_download: bool = False,/' "$BASE"
sed -i 's/^        local_files_only: bool,/        local_files_only: bool = False,/' "$BASE"
sed -i 's/^        token: Union\[str, bool, None\],/        token: Union[str, bool, None] = None,/' "$BASE"
sed -i '/^                resume_download=resume_download,/d' "$BASE"
# SAM-Audio's deps pull torchvision from the default (cu130) index, which mismatches
# torch+cu128 ("PyTorch and torchvision were compiled with different CUDA major versions").
# Repair: reinstall the cu128 build without touching torch.
pip install --force-reinstall --no-deps torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
echo "After 'hf auth login', weights auto-download on first SAMAudio.from_pretrained('facebook/sam-audio-large')."

# --- Silero VAD fallback (no weights to pre-fetch; torch.hub pulls on first use) ---
echo "Done. Silero fallback available via: python vad.py --backend silero --cpu"
