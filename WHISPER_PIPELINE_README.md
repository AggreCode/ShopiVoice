# Whisper Pharmaceutical Voice Order Pipeline

A complete voice-to-order pipeline using a fine-tuned Whisper ASR model for pharmaceutical domain transcription and structured order parsing.

## 🎯 Features

- **ESP32 Serial Recording**: Records audio directly from ESP32 microphone via serial port
- **Whisper ASR**: Uses fine-tuned Whisper model trained on pharmaceutical voice data
- **Domain-Specific Parsing**: Extracts quantities, units, and products with fuzzy matching
- **Inventory Matching**: Maps spoken product names to standardized inventory
- **Pricing Calculation**: Automatically calculates total order value
- **Comprehensive Logging**: Detailed logging at every pipeline stage

## 📋 Requirements

```bash
# Python packages
pip install torch transformers soundfile librosa pyserial rapidfuzz
```

## 🏗️ Architecture

```
ESP32 Recording → Whisper Transcription → Text Processing → Order Parsing → JSON Output
```

### Components

1. **Audio Recording** (`record_audio_from_esp32()`)
   - Connects to ESP32 via serial port
   - Records audio on button press
   - Saves as WAV file (16kHz mono)

2. **Whisper Transcription** (`WhisperTranscriber`)
   - Loads fine-tuned model from `whisper_pharma_model/`
   - Transcribes audio to text
   - Logs raw output and processing time

3. **Text Processing** (`process_text()`)
   - Normalizes transcription
   - Maps quantities (e.g., "five" → "5")
   - Maps units (e.g., "milligram" → "mg")

4. **Order Parsing** (`parse_order()`)
   - Extracts structured items from text
   - Matches products to inventory
   - Calculates pricing

5. **Output**
   - JSON with product, quantity, unit, pricing
   - Total order value

## 🚀 Usage

### Live ESP32 Recording

```bash
# Configure serial port in the script
# SERIAL_PORT = '/dev/ttyUSB0'  # Linux/Mac
# SERIAL_PORT = 'COM3'          # Windows

python whisper_pharma_voice_pipeline.py
```

**Workflow:**
1. Script loads Whisper model
2. Connects to ESP32
3. Press button to start recording
4. Speak your order (e.g., "5 strip paracetamol, 2 bottles dettol")
5. Press button to stop
6. Script transcribes and parses order
7. Displays structured JSON output

### Testing with Pre-recorded Audio

```bash
# Test single file
python test_whisper_pipeline.py training_output/audio/test/wavs/utt_000001.wav

# Test batch of files (10 samples)
python test_whisper_pipeline.py training_output/audio/test 10
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
      "unit": "tablet",
      "price_per_unit": 30,
      "total_price": 300
    }
  ]
}

Total Order Value: ₹560.00
```

## 🗂️ Glossaries

The pipeline uses pharmaceutical glossaries from `pharma_glossary.json`:

### Quantities
- Numbers: zero-hundred, half, quarter
- Mapped to numeric values: "five" → "5"

### Units
- Weights: kg, gram, mg
- Volumes: ml, liter
- Packaging: strip, tablet, bottle, box, pcs

### Products
- Generic medicines: paracetamol, ibuprofen, metformin
- Brand names: crocin, dolo 650, brufen, glucophage
- First aid: dettol, band-aid, vicks

## 📝 Logging

Comprehensive logging includes:

- **Model Loading**: Device (cuda/cpu), load time
- **Audio Processing**: Duration, sample rate, shape
- **Transcription**: Raw output, processing time
- **Text Processing**: Normalization steps, mappings
- **Parsing**: Item extraction, product matching, scoring
- **Final Output**: Complete JSON structure

Logs are written to console with timestamps and log levels.

## ⚙️ Configuration

Edit these variables in `whisper_pharma_voice_pipeline.py`:

```python
# Model path
WHISPER_MODEL_PATH = "whisper_pharma_model"

# Serial port
SERIAL_PORT = '/dev/ttyUSB0'  # Change for your system
BAUD_RATE = 921600

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 🔧 Troubleshooting

### Serial Port Issues

```bash
# Linux: Check available ports
ls /dev/ttyUSB*
ls /dev/ttyACM*

# Grant permissions
sudo chmod 666 /dev/ttyUSB0

# Mac: Check ports
ls /dev/cu.*
```

### Model Not Found

```bash
# Ensure model exists
ls whisper_pharma_model/

# Should contain:
# - config.json
# - preprocessor_config.json
# - pytorch_model.bin or model.safetensors
```

### CUDA Out of Memory

```python
# Switch to CPU mode
DEVICE = "cpu"
```

### Poor Transcription Quality

1. Check audio quality:
   - Clear speech, minimal background noise
   - Good microphone placement
   
2. Verify model was trained on similar domain

3. Check audio format:
   - 16kHz sample rate
   - Mono channel
   - WAV format

## 🎓 Training Data

The Whisper model was fine-tuned on 5000 pharmaceutical voice samples with:
- Products: 30+ common medicines
- Units: mg, ml, kg, strip, tablet, bottle
- Quantities: 0-100
- Format: "{quantity} {unit} {product}"

Test WER: ~44% (domain-specific)

## 📚 References

- Whisper Model: [OpenAI Whisper](https://github.com/openai/whisper)
- Hugging Face Transformers: [transformers](https://huggingface.co/docs/transformers)
- Fuzzy Matching: [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

## 🤝 Support

For issues or questions:
1. Check logs for detailed error messages
2. Test with pre-recorded audio first
3. Verify all dependencies are installed
4. Ensure model path is correct

---

**Note**: This pipeline is designed for pharmaceutical voice orders in English. For other domains or languages, retrain the Whisper model on appropriate data.



