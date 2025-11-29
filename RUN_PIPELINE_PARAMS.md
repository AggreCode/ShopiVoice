# run_training_pipeline.sh - Complete Parameter Reference

## Overview

The `run_training_pipeline.sh` script now includes:
1. **TTS Generation** using `xtts_indian_pipeline.py`
2. **Data Augmentation** using `data_augmentation.py` (ENABLED by default)
3. Optional ESPnet setup (commented out, not recommended)

---

## Input Parameters (Default Values)

### Command Line Arguments

```bash
bash run_training_pipeline.sh [DATASET_SIZE] [MAX_SAMPLES] [OUTPUT_DIR]
```

| Argument | Position | Default | Description |
|----------|----------|---------|-------------|
| `DATASET_SIZE` | 1 | `5000` | Total number of samples to generate |
| `MAX_SAMPLES` | 2 | _(empty/all)_ | Max samples to process per split |
| `OUTPUT_DIR` | 3 | `training_output` | Base output directory |

### Script Configuration Variables

#### Dataset & Output
```bash
DATASET_SIZE=5000                           # Total samples
MAX_SAMPLES=""                              # Max per split (empty = all)
OUTPUT_DIR="training_output"                # Base directory
AUGMENTED_DIR="training_output_augmented"   # Augmented output
SPEAKER_WAV="Recording_13.wav"              # Voice sample
```

#### TTS Parameters
```bash
NUM_WORKERS=1                               # MUST be 1 (XTTS not thread-safe!)
LANGUAGE="en"                               # Language code
SAMPLE_RATE=16000                           # Audio sample rate (Hz)
```

#### Data Augmentation
```bash
ENABLE_AUGMENTATION=true                    # Enable/disable augmentation
NUM_AUGMENTATIONS=3                         # Augmented versions per file
AUGMENTATION_WORKERS=4                      # Parallel workers

# Noise augmentation
SNR_DB_MIN=15                               # Min signal-to-noise ratio (dB)
SNR_DB_MAX=25                               # Max signal-to-noise ratio (dB)

# Speed augmentation  
SPEED_MIN=0.9                               # Min speed factor (90%)
SPEED_MAX=1.1                               # Max speed factor (110%)

# Pitch augmentation
PITCH_MIN=-1.5                              # Min pitch shift (semitones)
PITCH_MAX=1.5                               # Max pitch shift (semitones)
```

---

## Pipeline Steps

### Step 1: TTS Generation

**Script:** `test_xtts_indian.py`  
**Uses:** `xtts_indian_pipeline.py`, `updated_smart_dataset_generator.py`

**Parameters passed:**
```bash
python3 test_xtts_indian.py \
  --dataset-size $DATASET_SIZE \
  --speaker-wav $SPEAKER_WAV \
  --output-dir $OUTPUT_DIR \
  --num-workers $NUM_WORKERS
```

**What it does:**
- Generates smart dataset with glossary coverage
- Creates train/val/test splits (80/10/10)
- Synthesizes audio using XTTS v2 with Indian voice
- Creates `metadata.csv` files (pipe-delimited: `file_id|text`)

**Output:**
```
training_output/
├── dataset/
│   ├── train/metadata.csv    (4000 entries)
│   ├── val/metadata.csv      (500 entries)
│   └── test/metadata.csv     (500 entries)
└── audio/
    ├── train/wavs/           (4000 WAV files)
    ├── val/wavs/             (500 WAV files)
    └── test/wavs/            (500 WAV files)
```

---

### Step 2: Data Augmentation

**Script:** `data_augmentation.py`

**Parameters passed:**
```bash
python3 data_augmentation.py \
  --dataset $OUTPUT_DIR/audio \
  --output-dir $AUGMENTED_DIR \
  --num-augmentations $NUM_AUGMENTATIONS \
  --num-workers $AUGMENTATION_WORKERS \
  --sample-rate $SAMPLE_RATE
```

**What it does:**
- Applies multiple augmentation techniques:
  - **Noise:** White & pink noise (SNR: 15-25 dB)
  - **Speed:** Time-stretch (0.9x - 1.1x)
  - **Pitch:** Pitch shift (-1.5 to +1.5 semitones)
  - **RIR:** Room impulse response (reverberation)
- Creates 3 augmented versions per training file
- **Only augments training data** (val/test are copied unchanged)

**Output:**
```
training_output_augmented/
├── train/
│   ├── wavs/
│   │   ├── utt_000000.wav           (original)
│   │   ├── utt_000000_aug0.wav      (augmented v1)
│   │   ├── utt_000000_aug1.wav      (augmented v2)
│   │   ├── utt_000000_aug2.wav      (augmented v3)
│   │   └── ...                      (16,000 total)
│   └── metadata.csv
├── val/wavs/                        (500 files, copied)
└── test/wavs/                       (500 files, copied)
```

---

## Expected Output Summary

### For DATASET_SIZE=5000

| Split | Original Files | Augmented Files | Total Files | Size |
|-------|---------------|-----------------|-------------|------|
| Train | 4,000 | 12,000 | **16,000** | ~3.1 GB |
| Val | 500 | - | 500 | ~96 MB |
| Test | 500 | - | 500 | ~96 MB |
| **TOTAL** | **5,000** | **12,000** | **17,000** | **~3.3 GB** |

### Augmentation Multiplier

Each original training file creates:
- 1 original + 3 augmented = **4x training data**

---

## Usage Examples

### Basic Usage

```bash
# Default: 5000 samples with augmentation
bash run_training_pipeline.sh

# Generate 10,000 samples
bash run_training_pipeline.sh 10000

# Test with 100 samples
bash run_training_pipeline.sh 100
```

### Advanced Usage

```bash
# 5000 samples, max 100 per split (for quick testing)
bash run_training_pipeline.sh 5000 100

# Custom output directory
bash run_training_pipeline.sh 5000 "" my_custom_output

# Large dataset
bash run_training_pipeline.sh 20000
```

### Disabling Augmentation

Edit the script and change:
```bash
ENABLE_AUGMENTATION=false
```

---

## Estimated Processing Times

**Hardware:** RTX 4090 GPU

| Dataset Size | TTS (Step 1) | Augmentation (Step 2) | Total Time |
|--------------|--------------|----------------------|------------|
| 1,000 | ~6-9 min | ~4-6 min | ~10-15 min |
| 5,000 | ~30-45 min | ~20-30 min | **~50-75 min** |
| 10,000 | ~60-90 min | ~40-60 min | **~100-150 min** |
| 20,000 | ~120-180 min | ~80-120 min | **~200-300 min** |

**Note:** Times vary based on:
- GPU model (RTX 4090 is fastest)
- CPU cores (for augmentation)
- Disk I/O speed
- Text complexity

---

## Storage Requirements

| Dataset Size | Original Audio | Augmented Audio | Total Storage |
|--------------|---------------|-----------------|---------------|
| 5,000 | ~960 MB | ~2.3 GB | **~3.3 GB** |
| 10,000 | ~1.9 GB | ~4.6 GB | **~6.6 GB** |
| 20,000 | ~3.8 GB | ~9.2 GB | **~13 GB** |

**Formula:** Total ≈ Dataset_Size × 0.66 MB

---

## Parameter Tuning Guide

### When to Increase NUM_AUGMENTATIONS

**Increase to 5-7 if:**
- You have a small original dataset (<1000 samples)
- Model is overfitting (high train accuracy, low val accuracy)
- You need more training data variety

**Example:**
```bash
NUM_AUGMENTATIONS=5  # Creates 5 augmented versions = 6x training data
```

### When to Adjust Augmentation Intensity

**More aggressive augmentation (harder training):**
```bash
SNR_DB_MIN=10        # More noise
SNR_DB_MAX=20
SPEED_MIN=0.8        # Wider speed range
SPEED_MAX=1.2
PITCH_MIN=-2.0       # Wider pitch range
PITCH_MAX=2.0
```

**Lighter augmentation (easier training):**
```bash
SNR_DB_MIN=20        # Less noise
SNR_DB_MAX=30
SPEED_MIN=0.95       # Narrower speed range
SPEED_MAX=1.05
PITCH_MIN=-0.5       # Narrower pitch range
PITCH_MAX=0.5
```

### When to Increase AUGMENTATION_WORKERS

**Default:** 4 workers
**Increase to 8-16 if:**
- You have a powerful CPU (16+ cores)
- Augmentation is slow
- CPU usage is low during augmentation

**Example:**
```bash
AUGMENTATION_WORKERS=8  # For high-end CPUs
```

---

## Important Notes

### ⚠️ Critical Settings

1. **`NUM_WORKERS=1` for TTS**  
   MUST remain 1! XTTS v2 is NOT thread-safe.  
   Changing this will cause CUDA errors.

2. **`ENABLE_AUGMENTATION=true`**  
   Augmentation significantly improves model robustness.  
   Only disable for testing purposes.

3. **Val/Test Not Augmented**  
   Validation and test sets are NOT augmented.  
   This ensures fair evaluation.

### 💡 Best Practices

1. **Start Small**  
   Test with 100-500 samples first to verify the pipeline.

2. **Monitor Disk Space**  
   Ensure you have 2-3x the expected storage available.

3. **GPU Memory**  
   Minimum 16GB GPU RAM (RTX 4080).  
   Recommended: 24GB (RTX 4090, RTX 3090).

4. **Speaker WAV Quality**  
   Use clean, high-quality recording (16kHz+, no background noise).  
   Recording_13.wav should be 5-10 seconds long.

---

## Output File Structure

### Metadata Format

**File:** `training_output/audio/train/metadata.csv`  
**Format:** Pipe-delimited (`|`)

```
utt_000000|fifteen kilogram paracetamol
utt_000001|twenty pieces aspirin
utt_000002|three litre cetzine
```

**Fields:**
1. `file_id` - Unique utterance ID
2. `text` - Transcription text

### Audio Files

**Format:** WAV, 16kHz, mono  
**Naming:**
- Original: `utt_NNNNNN.wav`
- Augmented: `utt_NNNNNN_augN.wav` (N = 0, 1, 2, ...)

---

## Next Steps After Running

### 1. Verify Output

```bash
# Check file counts
find training_output_augmented/train/wavs -name "*.wav" | wc -l

# Check metadata
head -10 training_output_augmented/train/metadata.csv

# Check one audio file
ffprobe training_output_augmented/train/wavs/utt_000000.wav
```

### 2. Train Whisper Model (Recommended)

```bash
python3 whisper_pharma_train.py \
  --audio_dir training_output_augmented \
  --output_dir whisper_pharma_model \
  --model_size base \
  --num_epochs 10
```

### 3. OR Use ESPnet (Not Recommended)

```bash
python3 espnet_setup.py --data-dir training_output_augmented
cd ~/espnet/egs2/custom_asr/asr1
bash run.sh --stage 1 --stop_stage 100 --data_dir /path/to/training_output_augmented
```

---

## Troubleshooting

### Problem: CUDA Errors During TTS

**Solution:**
```bash
# Ensure NUM_WORKERS=1 in the script
export CUDA_LAUNCH_BLOCKING=1  # Uncomment in script
```

### Problem: Augmentation Too Slow

**Solutions:**
1. Increase `AUGMENTATION_WORKERS`
2. Reduce `NUM_AUGMENTATIONS`
3. Use faster disk (SSD)

### Problem: Out of Disk Space

**Solutions:**
1. Reduce `DATASET_SIZE`
2. Reduce `NUM_AUGMENTATIONS`
3. Free up disk space before running

### Problem: Out of GPU Memory

**Solutions:**
1. Use smaller XTTS model (if available)
2. Close other GPU-using applications
3. Use a GPU with more VRAM

---

## Summary Table

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `DATASET_SIZE` | 5000 | 100-50000 | Total dataset size |
| `NUM_WORKERS` | 1 | **MUST be 1** | TTS parallelism |
| `NUM_AUGMENTATIONS` | 3 | 1-10 | Augmentations per file |
| `AUGMENTATION_WORKERS` | 4 | 1-16 | Augmentation parallelism |
| `ENABLE_AUGMENTATION` | true | true/false | Enable augmentation |
| `SNR_DB_MIN` | 15 | 5-30 | Min noise level |
| `SNR_DB_MAX` | 25 | 10-40 | Max noise level |
| `SPEED_MIN` | 0.9 | 0.7-1.0 | Min speed factor |
| `SPEED_MAX` | 1.1 | 1.0-1.3 | Max speed factor |
| `PITCH_MIN` | -1.5 | -5 to 0 | Min pitch shift |
| `PITCH_MAX` | 1.5 | 0 to 5 | Max pitch shift |

---

**Last Updated:** November 2025  
**Tested With:** RTX 4090, Ubuntu 22.04, Python 3.10





