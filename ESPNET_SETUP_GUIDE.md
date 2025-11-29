# ESPnet Setup Guide

## Current Status
✅ **TTS Pipeline:** Working perfectly on GPU  
⚠️ **ESPnet Setup:** Commented out (requires additional dependencies)

---

## Why ESPnet Setup Was Disabled

The error you encountered:
```
2025-11-16 05:39:09,460 - ESPnetSetup - ERROR - Required command not found: sox
ESPnet setup failed
```

**Problem:** ESPnet requires `sox` (Sound eXchange) for audio processing, which is not installed on your GPU server.

---

## How to Enable ESPnet Setup (When Ready)

### Step 1: Install Required Dependencies

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y sox libsox-fmt-all ffmpeg
```

**On CentOS/RHEL:**
```bash
sudo yum install -y sox ffmpeg
```

**Verify installation:**
```bash
sox --version
# Should show: SoX v14.4.x or similar
```

---

### Step 2: Uncomment ESPnet Setup in Script

Edit `run_training_pipeline.sh` and uncomment lines **79-136**:

```bash
# Find this section (currently commented):
# # Step 2: Apply Data Augmentation (Optional)
# echo -e "${GREEN}========================================${NC}"
# ...

# Remove the # from all lines in that section
```

Or use this command to uncomment automatically:
```bash
sed -i '79,136s/^# //' run_training_pipeline.sh
```

---

### Step 3: Run Full Pipeline

```bash
bash run_training_pipeline.sh
```

Now when prompted:
- "Apply data augmentation? (y/n)" → Enter `y` if you want augmented data
- "Setup ESPnet training environment? (y/n)" → Enter `y`

---

## Alternative: Manual ESPnet Setup

If you prefer to set up ESPnet manually:

### 1. Clone ESPnet
```bash
cd ~
git clone https://github.com/espnet/espnet.git
cd espnet
```

### 2. Install ESPnet
```bash
# Create conda environment
conda create -n espnet python=3.9
conda activate espnet

# Install ESPnet
pip install -e .
pip install torch torchaudio
```

### 3. Prepare Your Data
```bash
# Run the espnet_setup.py manually
cd /path/to/upgraded_voice_pipeline
python3 espnet_setup.py --data-dir training_output/audio
```

---

## What ESPnet Setup Does

The `espnet_setup.py` script:
1. ✅ Creates ESPnet-compatible directory structure
2. ✅ Generates `wav.scp`, `text`, `utt2spk`, `spk2utt` files
3. ✅ Prepares data for ASR training
4. ✅ Sets up configuration files

---

## For Now: Focus on TTS Generation

The current pipeline works perfectly for TTS generation:

```bash
# Generate 5000 audio files with XTTS v2
bash run_training_pipeline.sh 5000

# Or with custom settings
bash run_training_pipeline.sh 10000  # 10k samples
```

**Output:**
- ✅ High-quality XTTS v2 audio files
- ✅ Metadata CSV files (audio-to-text mapping)
- ✅ Coverage statistics
- ✅ Train/val/test splits (80/10/10)

You can use this data for:
- ✅ Whisper fine-tuning
- ✅ Wav2Vec2 fine-tuning
- ✅ Kaldi training
- ✅ ESPnet training (after installing dependencies)

---

## Quick Comparison: ASR Training Options

| Toolkit | Setup Difficulty | Training Speed | Best For |
|---------|-----------------|----------------|----------|
| **Whisper Fine-tuning** | Easy | Fast | General purpose, quick results |
| **Wav2Vec2** | Medium | Fast | Low-resource languages |
| **ESPnet** | Hard | Medium | Research, customization |
| **Kaldi** | Very Hard | Slow | Production, maximum control |

**Recommendation for beginners:** Start with Whisper fine-tuning!

---

## Next Steps

1. ✅ **Done:** Generate TTS audio files
2. 🔄 **Optional:** Install sox and enable ESPnet
3. 🎯 **Recommended:** Try Whisper fine-tuning first (easier!)

---

## Whisper Fine-tuning Quick Start

Instead of ESPnet, try Whisper fine-tuning (simpler):

```python
# Install
pip install transformers datasets

# Fine-tune on your data
python whisper_finetune.py \
  --audio-dir training_output/audio \
  --model-name openai/whisper-small \
  --output-dir whisper_finetuned
```

This is **much easier** than ESPnet and works great for pharmaceutical vocabulary! 🚀

---

For questions or issues, check:
- ESPnet Docs: https://espnet.github.io/espnet/
- Whisper Docs: https://github.com/openai/whisper









