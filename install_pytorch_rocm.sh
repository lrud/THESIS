#!/bin/bash
# Install PyTorch with ROCm 7.0 support for AMD GPUs (Nightly - Cutting Edge)
# For Radeon RX 7900 XT (gfx1100)

set -e

echo "=========================================="
echo "PyTorch ROCm 7.0 Installation (Nightly)"
echo "=========================================="
echo ""

# Check ROCm version
echo "[1] Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname | head -20
    echo ""
else
    echo "⚠️  ROCm not found. Please install ROCm 7.0+ first."
    exit 1
fi

# Activate virtual environment
echo "[2] Activating virtual environment..."
cd "$(dirname "$0")"
source .venv/bin/activate

# Install PyTorch with ROCm 7.0 (Nightly - Cutting Edge)
echo ""
echo "[3] Installing PyTorch Nightly with ROCm 7.0 support..."
echo "    Using cutting edge nightly builds..."
echo "    This may take a few minutes..."
echo ""

pip install --upgrade pip setuptools wheel

# Install PyTorch Nightly with ROCm 7.0 (most cutting edge)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Install additional ML dependencies
pip install scikit-learn tensorboard matplotlib seaborn

echo ""
echo "[4] Verifying installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ ROCm version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # Test GPU
    print(f'\\n[Testing GPU...]')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f'✅ GPU computation successful!')
    print(f'   Device: {z.device}')
    print(f'   Matrix shape: {z.shape}')
else:
    print(f'⚠️  GPU not detected. Will use CPU.')
"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "PyTorch Nightly (ROCm 7.0) installed successfully!"
echo ""
echo "To train the baseline LSTM model:"
echo "  source .venv/bin/activate"
echo "  python scripts/modeling/lstm_baseline_train.py"
echo ""
