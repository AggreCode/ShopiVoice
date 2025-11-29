# Complete Dependencies List - Voice Pipeline

This document lists all dependencies required for the Upgraded Voice Pipeline with exact versions.

## 📋 Table of Contents
- [System Dependencies](#system-dependencies)
- [Python Dependencies](#python-dependencies)
- [Optional Dependencies](#optional-dependencies)
- [Installation](#installation)
- [Version Matrix](#version-matrix)

---

## 🖥️ System Dependencies

### Build Tools
```bash
build-essential      # GCC, G++, make
cmake >= 3.16        # Build system
git >= 2.25          # Version control
wget                 # File downloader
curl                 # HTTP client
pkg-config           # Compile flag manager
ca-certificates      # SSL certificates
```

### Python Environment
```bash
python3 >= 3.8       # Python interpreter
python3-pip          # Package installer
python3-dev          # Python headers
python3-venv         # Virtual environment support
```

### Audio Processing Libraries
```bash
ffmpeg >= 4.2        # Audio/video converter
sox >= 14.4          # Sound eXchange tool
libsox-dev           # SoX development files
libsox-fmt-all       # SoX format handlers
libsndfile1          # Audio file library
libsndfile1-dev      # libsndfile headers
portaudio19-dev      # Audio I/O library
```

### Text-to-Speech Support
```bash
espeak-ng >= 1.50    # Speech synthesizer (for phonemizer)
festival             # Speech synthesis system
mbrola               # Speech synthesizer
```

### Language Modeling (Optional)
```bash
libeigen3-dev        # Linear algebra library (for KenLM)
libboost-all-dev     # Boost C++ libraries (for KenLM)
```

---

## 🐍 Python Dependencies

### Core Deep Learning

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `torch` | 2.1.2 | PyTorch deep learning framework | All TTS & ASR |
| `torchvision` | 0.16.2 | PyTorch vision utilities | Transformers |
| `torchaudio` | 2.1.2 | PyTorch audio processing | Audio features |
| `transformers` | 4.36.2 | Hugging Face Transformers | Whisper fine-tuning |
| `datasets` | 2.16.1 | Hugging Face Datasets | Data loading |
| `evaluate` | 0.4.1 | Evaluation metrics | Model evaluation |
| `accelerate` | 0.25.0 | Multi-GPU training | Distributed training |

**Used by:** `whisper_pharma_train.py`, `xtts_indian_pipeline.py`

### Text-to-Speech (TTS)

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `TTS` | 0.22.0 | Coqui TTS (XTTS v2) | Voice generation |

**Used by:** `xtts_indian_pipeline.py`, `test_xtts_indian.py`

**Important:** XTTS v2 is NOT thread-safe! Always use `num_workers=1`

### Audio Processing

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `librosa` | 0.10.1 | Audio analysis & processing | Data augmentation |
| `soundfile` | 0.12.1 | Audio file I/O | Reading/writing WAV |
| `audioread` | 3.0.1 | Audio file decoding | Librosa backend |
| `resampy` | 0.4.2 | Audio resampling | Sample rate conversion |
| `pydub` | 0.25.1 | Simple audio manipulation | Audio conversion |

**Used by:** `data_augmentation.py`, `xtts_indian_pipeline.py`, `whisper_pharma_train.py`

### Scientific Computing

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `numpy` | 1.24.3 | Numerical computing | All scripts |
| `scipy` | 1.11.4 | Scientific computing | Signal processing |
| `pandas` | 2.0.3 | Data manipulation | CSV processing |

**Used by:** All Python scripts

### Text Processing

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `word2number` | 1.1 | Word to number conversion | Slot parsing |
| `num2words` | 0.5.13 | Number to word conversion | Dataset generation |

**Used by:** `slot_parser.py`, `dataset_generator.py`

### Fuzzy Matching & Phonetics

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `rapidfuzz` | 3.5.2 | Fast fuzzy string matching | Entity extraction |
| `phonemizer` | 3.2.1 | Text to phonemes conversion | Phonetic matching |

**Used by:** `slot_parser.py`

**Requirements:**
- `phonemizer` requires `espeak-ng` system package
- `rapidfuzz` is pure Python (no system deps)

### Utilities

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `tqdm` | 4.66.1 | Progress bars | All scripts |
| `coloredlogs` | 15.0.1 | Colored logging | Better logs |
| `requests` | 2.31.0 | HTTP requests | Model downloads |
| `python-dateutil` | 2.8.2 | Date utilities | Timestamps |
| `pytz` | 2023.3 | Timezone support | Timestamps |
| `openpyxl` | 3.1.2 | Excel file support | Optional export |

**Used by:** Various scripts for UI/UX improvements

---

## 🔧 Optional Dependencies

### ASR Testing (Vosk)

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `vosk` | 0.3.45 | Lightweight ASR | Legacy testing scripts |

**Used by:** `backup/simple_test.py`, `backup/integrated_test.py`

**Note:** Only needed for testing with Vosk models (legacy)

### ESPnet (Not Recommended)

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `espnet` | 202308 | Speech processing toolkit | Complex, use Whisper instead |
| `espnet-model-zoo` | 0.1.7 | Pre-trained models | With ESPnet |

**Used by:** `espnet_setup.py` (optional)

**Warning:** ESPnet is very complex! Use Whisper fine-tuning instead (see `WHISPER_QUICKSTART.md`)

### Language Modeling (KenLM)

**System package** - Built from source

- **Repository:** https://github.com/kpu/kenlm
- **Purpose:** N-gram language model building
- **Used by:** `kenlm_builder.py`

**Installation:**
```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Grammar FST (OpenFST + Pynini)

**System package:** OpenFST 1.8.2
- **Purpose:** Finite State Transducers for grammar
- **Used by:** `grammar_fst.py`, `wfst_decoder.py`

**Python package:** `pynini==2.1.5`
- **Purpose:** Python bindings for OpenFST

**Installation:**
```bash
# OpenFST
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz
tar -xzf openfst-1.8.2.tar.gz
cd openfst-1.8.2
./configure --enable-grm --enable-python
make -j$(nproc) && sudo make install
sudo ldconfig

# Pynini
pip install pynini==2.1.5
```

---

## 📦 Installation

### Quick Install (Recommended)

```bash
# Run the comprehensive installation script
bash install_all_dependencies.sh
```

This script will:
1. ✅ Install all system dependencies
2. ✅ Create a Python virtual environment
3. ✅ Install PyTorch with CUDA support
4. ✅ Install all Python packages with exact versions
5. ✅ Optionally install KenLM and OpenFST
6. ✅ Verify all installations
7. ✅ Create activation script

### Manual Install

If you prefer manual installation:

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl \
    python3 python3-pip python3-dev python3-venv \
    ffmpeg sox libsox-dev libsndfile1 portaudio19-dev \
    espeak-ng festival

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install Python packages
pip install numpy==1.24.3 scipy==1.11.4 librosa==0.10.1 \
    soundfile==0.12.1 TTS==0.22.0 transformers==4.36.2 \
    datasets==2.16.1 evaluate==0.4.1 accelerate==0.25.0 \
    word2number==1.1 rapidfuzz==3.5.2 phonemizer==3.2.1 \
    pandas==2.0.3 tqdm==4.66.1 coloredlogs==15.0.1
```

---

## 🔄 Version Matrix

### Tested Configurations

| Configuration | OS | Python | PyTorch | CUDA | TTS | Status |
|---------------|----|---------| --------|------|-----|--------|
| **Production** | Ubuntu 22.04 | 3.10.12 | 2.1.2 | 11.8 | 0.22.0 | ✅ Working |
| **Development** | Ubuntu 20.04 | 3.8.10 | 2.1.2 | 11.8 | 0.22.0 | ✅ Working |
| **GPU Server** | Ubuntu 22.04 | 3.10 | 2.1.2 | 12.1 | 0.22.0 | ✅ Working |
| **CPU Only** | Ubuntu 22.04 | 3.10 | 2.1.2 | N/A | 0.22.0 | ⚠️ Slow |

### Compatibility Notes

#### CUDA Compatibility
- **CUDA 11.8:** Recommended, tested with RTX 4090
- **CUDA 12.x:** Works, but use PyTorch built for CUDA 12.1
- **CPU Only:** Works but TTS generation is 10-20x slower

#### Python Compatibility
- **Python 3.8:** Minimum required
- **Python 3.9-3.10:** Recommended
- **Python 3.11+:** Not fully tested, may have issues with older packages

#### PyTorch Compatibility
- **PyTorch 2.0.x:** Minimum for XTTS v2
- **PyTorch 2.1.x:** Recommended
- **PyTorch 2.2.x:** Should work but not tested

---

## 🐛 Known Issues & Workarounds

### Issue 1: XTTS Thread Safety
**Problem:** CUDA errors when using `num_workers > 1`

**Solution:**
```python
# Always use num_workers=1 for XTTS
pipeline = XTTSIndianPipeline({
    "num_workers": 1  # CRITICAL!
})
```

### Issue 2: CUDA Device-Side Assert
**Problem:** Errors with short or problematic text

**Solution:** Already fixed in `xtts_indian_pipeline.py` with text sanitization

### Issue 3: ESPnet PATH Issues
**Problem:** `activate_python.sh` not found

**Solution:** Use Whisper instead (see `WHISPER_QUICKSTART.md`)

### Issue 4: phonemizer Installation
**Problem:** `phonemizer` import fails

**Solution:**
```bash
sudo apt-get install espeak-ng
pip install phonemizer==3.2.1
```

### Issue 5: librosa/soundfile Issues
**Problem:** `soundfile` can't load audio

**Solution:**
```bash
sudo apt-get install libsndfile1 libsndfile1-dev
pip install --upgrade soundfile==0.12.1
```

---

## 📊 Disk Space Requirements

| Component | Size | Purpose |
|-----------|------|---------|
| System packages | ~500 MB | System libraries |
| Virtual environment | ~100 MB | Python venv |
| PyTorch + CUDA | ~5 GB | Deep learning |
| Python packages | ~3 GB | All dependencies |
| XTTS v2 model | ~2 GB | Downloaded on first use |
| Whisper models | ~1-3 GB | Per model size |
| KenLM (optional) | ~50 MB | Language modeling |
| OpenFST (optional) | ~100 MB | Grammar FST |
| **Total (minimum)** | **~11 GB** | Without optional |
| **Total (full)** | **~15 GB** | With all optional |

---

## 🚀 Performance Benchmarks

### TTS Generation Speed (XTTS v2)

| Hardware | Speed | Samples/sec |
|----------|-------|-------------|
| RTX 4090 | Fast | ~15-20 |
| RTX 3090 | Fast | ~12-15 |
| RTX 2080 Ti | Medium | ~8-10 |
| CPU (16-core) | Slow | ~1-2 |

### Whisper Fine-Tuning Time

| Model Size | GPU | Time (1000 samples) |
|------------|-----|---------------------|
| Base | RTX 4090 | ~30 min |
| Small | RTX 4090 | ~1 hour |
| Medium | RTX 4090 | ~2 hours |
| Large | RTX 4090 | ~4 hours |

---

## 📝 Maintenance

### Update All Packages
```bash
source venv/bin/activate
pip list --outdated
pip install --upgrade pip setuptools wheel
# Review and update specific packages as needed
```

### Clean Installation
```bash
# Remove virtual environment
rm -rf venv

# Re-run installation
bash install_all_dependencies.sh
```

### Export Current Environment
```bash
source venv/bin/activate
pip freeze > current_environment.txt
```

---

## 🔗 Related Documentation

- **[WHISPER_QUICKSTART.md](WHISPER_QUICKSTART.md)** - Whisper fine-tuning guide
- **[PIPELINE_CHANGES_SUMMARY.md](PIPELINE_CHANGES_SUMMARY.md)** - Pipeline overview
- **[CUDA_FIXES_SUMMARY.md](CUDA_FIXES_SUMMARY.md)** - CUDA error fixes
- **[ESPNET_MANUAL_INSTALLATION.md](ESPNET_MANUAL_INSTALLATION.md)** - ESPnet setup (not recommended)
- **[README.md](README.md)** - General documentation

---

## 📞 Support

If you encounter issues:

1. Check this document for known issues
2. Verify all dependencies are installed: `bash install_all_dependencies.sh` (verify section)
3. Check GPU availability: `nvidia-smi`
4. Review logs in your script output
5. For XTTS issues, ensure `num_workers=1`

---

**Last Updated:** November 2025  
**Tested With:** Ubuntu 22.04, Python 3.10, PyTorch 2.1.2, CUDA 11.8, RTX 4090





