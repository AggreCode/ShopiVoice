#!/bin/bash
#
# Comprehensive Installation Script for Upgraded Voice Pipeline
# This script installs all dependencies with exact versions
#
# Tested on: Ubuntu 20.04/22.04 with CUDA 11.8+
# Hardware: NVIDIA RTX 4090 (or any CUDA-capable GPU)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_warning "This script should NOT be run as root for Python packages"
   log_info "System packages will be installed with sudo when needed"
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    log_info "Detected OS: $OS $VER"
else
    log_error "Cannot detect OS version"
    exit 1
fi

#==============================================================================
# STEP 1: INSTALL SYSTEM DEPENDENCIES
#==============================================================================
log_info "Step 1: Installing system dependencies..."

# Update package lists
log_info "Updating package lists..."
sudo apt-get update -qq

# Essential build tools
log_info "Installing build tools..."
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    ca-certificates

# Python development packages
log_info "Installing Python development packages..."
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

# Audio processing libraries
log_info "Installing audio processing libraries..."
sudo apt-get install -y -qq \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev

# Additional dependencies for phonemizer (optional but recommended)
log_info "Installing espeak-ng for phonemizer..."
sudo apt-get install -y -qq \
    espeak-ng \
    festival \
    mbrola

# KenLM dependencies (for language modeling)
log_info "Installing KenLM dependencies..."
sudo apt-get install -y -qq \
    libeigen3-dev \
    libboost-all-dev

log_success "System dependencies installed successfully!"

#==============================================================================
# STEP 2: SETUP PYTHON VIRTUAL ENVIRONMENT (RECOMMENDED)
#==============================================================================
log_info "Step 2: Setting up Python virtual environment..."

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    log_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip, setuptools, and wheel
log_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

log_success "Virtual environment setup complete!"

#==============================================================================
# STEP 3: INSTALL PYTORCH WITH CUDA SUPPORT
#==============================================================================
log_info "Step 3: Installing PyTorch with CUDA support..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
    log_info "CUDA detected: $CUDA_VERSION"
    
    # Install PyTorch with CUDA 11.8 (compatible with most modern GPUs)
    log_info "Installing PyTorch 2.1.2 with CUDA 11.8 support..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
else
    log_warning "CUDA not detected. Installing CPU-only PyTorch..."
    log_warning "TTS generation will be VERY slow without GPU!"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
fi

log_success "PyTorch installed successfully!"

#==============================================================================
# STEP 4: INSTALL CORE PYTHON DEPENDENCIES
#==============================================================================
log_info "Step 4: Installing core Python dependencies..."

# Create a temporary requirements file with exact versions
cat > /tmp/voice_pipeline_requirements.txt << 'EOF'
# Core Scientific Computing
numpy==1.24.3
scipy==1.11.4

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
audioread==3.0.1
resampy==0.4.2

# Coqui TTS (XTTS v2)
TTS==0.22.0

# Transformers & Hugging Face
transformers==4.36.2
datasets==2.16.1
evaluate==0.4.1
accelerate==0.25.0

# Text Processing
word2number==1.1
num2words==0.5.13

# Fuzzy Matching & Phonetics
rapidfuzz==3.5.2
phonemizer==3.2.1

# Data Processing
pandas==2.0.3
openpyxl==3.1.2

# Progress Bars & Logging
tqdm==4.66.1
coloredlogs==15.0.1

# Other Utilities
python-dateutil==2.8.2
pytz==2023.3
requests==2.31.0

# Optional: Vosk for ASR testing (backup)
vosk==0.3.45

# Optional: PyDub for audio manipulation
pydub==0.25.1

# Optional: ESPnet (if you want to try it later)
# espnet==202308
# espnet-model-zoo==0.1.7
EOF

log_info "Installing Python packages with exact versions..."
pip install -r /tmp/voice_pipeline_requirements.txt

log_success "Core Python dependencies installed successfully!"

#==============================================================================
# STEP 5: INSTALL KENLM (OPTIONAL - FOR LANGUAGE MODELING)
#==============================================================================
log_info "Step 5: Installing KenLM (optional, for language modeling)..."

if [ ! -d "kenlm" ]; then
    log_info "Cloning KenLM repository..."
    git clone https://github.com/kpu/kenlm.git
    
    log_info "Building KenLM..."
    cd kenlm
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install
    cd ../..
    
    log_success "KenLM installed successfully!"
else
    log_warning "KenLM directory already exists, skipping..."
fi

#==============================================================================
# STEP 6: INSTALL OPENFST & PYNINI (OPTIONAL - FOR GRAMMAR FST)
#==============================================================================
log_info "Step 6: Installing OpenFST & Pynini (optional, for grammar FST)..."

# Check if OpenFST is already installed
if ! command -v fstcompile &> /dev/null; then
    log_info "Installing OpenFST..."
    
    # Download and install OpenFST
    OPENFST_VERSION="1.8.2"
    wget -q "http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz"
    tar -xzf "openfst-${OPENFST_VERSION}.tar.gz"
    cd "openfst-${OPENFST_VERSION}"
    
    ./configure --enable-grm --enable-python
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
    cd ..
    rm -rf "openfst-${OPENFST_VERSION}" "openfst-${OPENFST_VERSION}.tar.gz"
    
    log_success "OpenFST installed successfully!"
else
    log_info "OpenFST already installed"
fi

# Install Pynini (Python bindings for OpenFST)
log_info "Installing Pynini..."
pip install pynini==2.1.5

log_success "OpenFST & Pynini installed successfully!"

#==============================================================================
# STEP 7: VERIFY INSTALLATIONS
#==============================================================================
log_info "Step 7: Verifying installations..."

echo ""
log_info "===== Installation Verification ====="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Check PyTorch
if python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    log_success "✓ PyTorch installed and working"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        log_success "✓ CUDA available - GPU: $GPU_NAME"
    else
        log_warning "⚠ CUDA not available - will use CPU (slow!)"
    fi
else
    log_error "✗ PyTorch installation failed"
fi

# Check TTS
if python -c "from TTS.api import TTS" 2>/dev/null; then
    log_success "✓ Coqui TTS (XTTS v2) installed"
else
    log_error "✗ Coqui TTS installation failed"
fi

# Check Transformers
if python -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null; then
    log_success "✓ Hugging Face Transformers installed"
else
    log_error "✗ Transformers installation failed"
fi

# Check datasets
if python -c "import datasets" 2>/dev/null; then
    log_success "✓ Hugging Face Datasets installed"
else
    log_error "✗ Datasets installation failed"
fi

# Check librosa
if python -c "import librosa; print('Librosa:', librosa.__version__)" 2>/dev/null; then
    log_success "✓ Librosa installed"
else
    log_error "✗ Librosa installation failed"
fi

# Check soundfile
if python -c "import soundfile" 2>/dev/null; then
    log_success "✓ SoundFile installed"
else
    log_error "✗ SoundFile installation failed"
fi

# Check numpy
if python -c "import numpy; print('NumPy:', numpy.__version__)" 2>/dev/null; then
    log_success "✓ NumPy installed"
else
    log_error "✗ NumPy installation failed"
fi

# Check scipy
if python -c "import scipy; print('SciPy:', scipy.__version__)" 2>/dev/null; then
    log_success "✓ SciPy installed"
else
    log_error "✗ SciPy installation failed"
fi

# Check pandas
if python -c "import pandas; print('Pandas:', pandas.__version__)" 2>/dev/null; then
    log_success "✓ Pandas installed"
else
    log_error "✗ Pandas installation failed"
fi

# Check word2number
if python -c "from word2number import w2n" 2>/dev/null; then
    log_success "✓ word2number installed"
else
    log_warning "⚠ word2number installation failed (optional)"
fi

# Check rapidfuzz
if python -c "from rapidfuzz import fuzz" 2>/dev/null; then
    log_success "✓ rapidfuzz installed"
else
    log_warning "⚠ rapidfuzz installation failed (optional)"
fi

# Check phonemizer
if python -c "from phonemizer import phonemize" 2>/dev/null; then
    log_success "✓ phonemizer installed"
else
    log_warning "⚠ phonemizer installation failed (optional)"
fi

# Check system commands
echo ""
log_info "Checking system commands..."

for cmd in ffmpeg sox git cmake; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | head -n1)
        log_success "✓ $cmd installed: $VERSION"
    else
        log_warning "⚠ $cmd not found"
    fi
done

# Check KenLM (optional)
if command -v lmplz &> /dev/null; then
    log_success "✓ KenLM installed"
else
    log_warning "⚠ KenLM not installed (optional, needed for language modeling)"
fi

# Check OpenFST (optional)
if command -v fstcompile &> /dev/null; then
    log_success "✓ OpenFST installed"
else
    log_warning "⚠ OpenFST not installed (optional, needed for grammar FST)"
fi

echo ""
log_success "===== Installation verification complete! ====="
echo ""

#==============================================================================
# STEP 8: CREATE ACTIVATION SCRIPT
#==============================================================================
log_info "Step 8: Creating activation script..."

cat > activate_pipeline.sh << 'EOF'
#!/bin/bash
# Activation script for the voice pipeline environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✓ Virtual environment activated"
    
    # Display environment info
    echo ""
    echo "Python: $(python --version)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "CUDA: Available - $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    else
        echo "CUDA: Not available (using CPU)"
    fi
    echo ""
    
    # Update PATH to include KenLM if installed
    if [ -d "$SCRIPT_DIR/kenlm/build/bin" ]; then
        export PATH="$SCRIPT_DIR/kenlm/build/bin:$PATH"
        echo "✓ KenLM tools added to PATH"
    fi
    
    echo "Ready to use the voice pipeline!"
    echo "To deactivate, run: deactivate"
else
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    exit 1
fi
EOF

chmod +x activate_pipeline.sh

log_success "Activation script created: activate_pipeline.sh"

#==============================================================================
# STEP 9: FINAL INSTRUCTIONS
#==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    INSTALLATION COMPLETE!                              ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
log_success "All dependencies have been installed successfully!"
echo ""
log_info "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   ${GREEN}source activate_pipeline.sh${NC}"
echo ""
echo "2. Test the XTTS pipeline (GPU):"
echo "   ${GREEN}python3 test_xtts_indian.py --dataset-size 10 --speaker-wav Recording_13.wav${NC}"
echo ""
echo "3. Run the full training pipeline:"
echo "   ${GREEN}bash run_training_pipeline.sh${NC}"
echo ""
echo "4. Fine-tune Whisper model:"
echo "   ${GREEN}python3 whisper_pharma_train.py --audio_dir training_output/audio --output_dir whisper_model${NC}"
echo ""
echo "5. See documentation:"
echo "   - ${BLUE}WHISPER_QUICKSTART.md${NC} - Whisper fine-tuning guide"
echo "   - ${BLUE}PIPELINE_CHANGES_SUMMARY.md${NC} - Pipeline overview"
echo "   - ${BLUE}README.md${NC} - General documentation"
echo ""
log_warning "Important Notes:"
echo "  - Always activate the environment before running scripts"
echo "  - For GPU acceleration, ensure CUDA is properly configured"
echo "  - Use num_workers=1 for XTTS to avoid CUDA errors (not thread-safe)"
echo "  - ESPnet is optional and complex - Whisper is recommended"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Happy Training! 🚀                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Save package list for future reference
log_info "Saving installed package list..."
pip list --format=freeze > installed_packages.txt
log_success "Package list saved to: installed_packages.txt"

echo ""
log_info "Installation log completed at: $(date)"





