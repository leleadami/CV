#!/bin/bash
# Setup automatico su nuovo PC (Linux con GPU NVIDIA).
# Uso: bash setup_new_pc.sh
# Richiede: miniconda/anaconda installato

set -e

ENV_NAME="hpe"
PY_VER="3.10"

echo "=== Setup progetto HPE (conda env: $ENV_NAME) ==="
echo

# Inizializza conda nello script
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Crea env se non esiste
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo ">> Creo env conda '$ENV_NAME' (python $PY_VER) ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

# Aggiorna pip
pip install --upgrade pip

# Per GPU NVIDIA: PyTorch con CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ">> GPU NVIDIA rilevata"
    nvidia-smi | head -3
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo ">> Nessuna GPU NVIDIA — installo PyTorch CPU"
    pip install torch torchvision
fi

# Resto dipendenze
pip install -r "$(dirname "$0")/requirements.txt"

echo
echo "=== Verifica installazione ==="
python -c "import torch; print(f'PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"

echo
echo "=== Lancio pipeline ==="
python "$(dirname "$0")/run_all.py"

echo
echo "Setup completato. Risultati in:"
echo "  - sanity_check_cameras.png"
echo "  - step2_triangulation/output/"
echo "  - step3_bundle_adjustment/output/"
echo
echo "Attiva env futura: conda activate $ENV_NAME"
