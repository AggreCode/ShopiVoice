# ShopiVoice - Pharmaceutical Voice Order Pipeline with Whisper ASR

A complete voice-to-order system for pharmaceutical products using fine-tuned Whisper ASR, ESP32 audio recording, and intelligent order parsing with inventory management.

## 🎯 Project Overview

This pipeline converts spoken pharmaceutical orders into structured JSON output with automatic product matching, brand detection, quantity extraction, and pricing calculation.

**Example:**
- **Input (spoken):** "five strip paracetamol, two bottles dettol"
- **Output:** Structured JSON with product names, quantities, units, brands, and total pricing

## 🏗️ Architecture

```
┌─────────────────┐
│  ESP32 Audio    │  → Records voice via serial port
│   Recording     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Whisper ASR     │  → Fine-tuned model transcription
│  Transcription  │     (trained on 5000 pharma samples)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Text Processing │  → Normalizes & maps quantities/units
│   & Parsing     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Product Matching│  → Fuzzy matching to inventory
│   & Pricing     │     with brand detection
└────────┬────────┘
         ↓
┌─────────────────┐
│  JSON Output    │  → Structured order with pricing
└─────────────────┘
```

## 📋 Key Features

- **Fine-tuned Whisper ASR**: Domain-specific model trained on 5000 pharmaceutical voice samples
- **ESP32 Integration**: Direct serial port audio recording
- **Intelligent Parsing**: Extracts quantities, units, products with fuzzy matching
- **Inventory Management**: 20+ pharmaceutical products with brand variants
- **Automatic Pricing**: Calculates total order value
- **Comprehensive Logging**: Detailed logs at every pipeline stage
- **Hardware-free Testing**: Test with pre-recorded audio files

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers soundfile librosa pyserial rapidfuzz
```

### 1. Test with Pre-recorded Audio (Recommended)

```bash
# Quick test with 5 sample files
bash quick_test_pipeline.sh

# Or test specific files
python3 test_whisper_pipeline.py training_output/audio/test 10
```

### 2. Live Recording with ESP32

```bash
# Configure serial port in whisper_pharma_voice_pipeline.py
# SERIAL_PORT = '/dev/ttyUSB0'  # Change to your port

# Run the pipeline
python3 whisper_pharma_voice_pipeline.py
```

## 📁 Project Structure

```
upgraded_voice_pipeline/
├── whisper_pharma_voice_pipeline.py   # Main pipeline
├── test_whisper_pipeline.py           # Testing script
├── quick_test_pipeline.sh             # Automated test
├── whisper_pharma_train.py            # Model training script
├── pharma_glossary.json               # Domain glossaries
├── xtts_indian_pipeline.py            # TTS data generation
├── data_augmentation.py               # Audio augmentation
├── run_training_pipeline.sh           # Complete training pipeline
└── Documentation/
    ├── WHISPER_PIPELINE_README.md     # Detailed pipeline guide
    ├── IMPLEMENTATION_SUMMARY.md      # Architecture details
    ├── WHISPER_QUICKSTART.md          # Training guide
    └── RUN_PIPELINE_PARAMS.md         # Parameter reference
```

## 🎓 Training Your Own Model

### 1. Generate Training Data

```bash
# Generate 5000 pharmaceutical voice samples using TTS
bash run_training_pipeline.sh
```

This creates:
- Synthetic voice data using XTTS v2
- Metadata with quantities, units, products
- Train/validation/test splits

### 2. Train Whisper Model

```bash
python3 whisper_pharma_train.py \
  --audio_dir training_output/audio \
  --output_dir whisper_pharma_model \
  --model_size base \
  --num_epochs 10 \
  --learning_rate 1e-5
```

**Training Details:**
- Base model: OpenAI Whisper Base
- Dataset: 5000 samples (4000 train, 500 val, 500 test)
- Test WER: ~44% (domain-specific)
- Training time: ~4-5 hours on GPU

### 3. Download Pre-trained Model

Since model files are too large for GitHub, download them separately:

```bash
# Option 1: From Hugging Face (recommended)
# Coming soon: huggingface.co/AggreCode/whisper-pharma-base

# Option 2: Train your own (see above)
```

## 📊 Example Output

**Input (spoken):**
```
"five strip paracetamol, two bottles dettol, ten tablet brufen"
```

**Output (JSON):**
```json
{
  "original_transcription": "five strip paracetamol two bottles dettol ten tablet brufen",
  "processed_text": "5 strip paracetamol 2 bottle dettol 10 tablet brufen",
  "parsed_order": [
    {
      "product_name": "Paracetamol",
      "brand_name": "Crocin",
      "quantity": 5,
      "unit": "strip",
      "price_per_unit": 20,
      "total_price": 100
    },
    {
      "product_name": "Dettol",
      "brand_name": null,
      "quantity": 2,
      "unit": "bottle",
      "price_per_unit": 80,
      "total_price": 160
    },
    {
      "product_name": "Ibuprofen",
      "brand_name": "Brufen",
      "quantity": 10,
      "unit": "strip",
      "price_per_unit": 30,
      "total_price": 300
    }
  ]
}

Total Order Value: ₹560.00
```

## 🗂️ Pharmaceutical Glossaries

### Supported Products (20+)
- Generic medicines: paracetamol, ibuprofen, metformin, cetirizine
- Brand names: crocin, brufen, dolo 650, glucophage
- First aid: dettol, band-aid, vicks vaporub

### Units
- Weights: kg, gram, mg, milligram
- Volumes: ml, liter, milliliter
- Packaging: strip, tablet, bottle, box, piece

### Quantities
- Numbers: zero through hundred
- Fractions: half, quarter
- Auto-mapping: "five" → "5"

## ⚙️ Configuration

Edit `whisper_pharma_voice_pipeline.py`:

```python
# Model path (change after downloading/training)
WHISPER_MODEL_PATH = "runpod_backup/whisper_pharma_model"

# Serial port (change for your system)
SERIAL_PORT = '/dev/ttyUSB0'  # Linux
# SERIAL_PORT = 'COM3'        # Windows

# Device selection (auto-detected)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🔧 Troubleshooting

### Serial Port Issues
```bash
# Linux: Find your port
ls /dev/ttyUSB* /dev/ttyACM*

# Grant permissions
sudo chmod 666 /dev/ttyUSB0
```

### Model Not Found
Ensure the trained model exists:
```bash
ls runpod_backup/whisper_pharma_model/
# Should contain: config.json, preprocessor_config.json, model.safetensors
```

### Poor Transcription
1. Test with pre-recorded audio first
2. Check audio quality (16kHz, mono, clear speech)
3. Verify model was trained on similar domain

## 📚 Documentation

- **[WHISPER_PIPELINE_README.md](WHISPER_PIPELINE_README.md)** - Complete pipeline guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Architecture details
- **[WHISPER_QUICKSTART.md](WHISPER_QUICKSTART.md)** - Training guide
- **[RUN_PIPELINE_PARAMS.md](RUN_PIPELINE_PARAMS.md)** - Parameter reference

## 🎯 Performance

- **Transcription Speed**: ~0.5-1.5s per utterance (GPU)
- **End-to-End Latency**: ~2-3s (recording + processing)
- **Test WER**: ~44% (pharmaceutical domain)
- **Parsing Accuracy**: High for known products

## 🛠️ Technology Stack

- **ASR**: OpenAI Whisper (fine-tuned)
- **TTS**: Coqui XTTS v2
- **Audio Processing**: soundfile, librosa
- **ML Framework**: PyTorch, Transformers
- **Fuzzy Matching**: RapidFuzz
- **Hardware**: ESP32 for recording

## 📖 Citation

Built on top of:
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For issues or questions, please open a GitHub issue.

---

**Note**: This is a domain-specific pharmaceutical voice order system for English. For other domains or languages, retrain the Whisper model on appropriate data.
