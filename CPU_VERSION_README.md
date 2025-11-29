# XTTS Indian Pipeline - CPU Version

## Overview

This CPU-only version of the XTTS Indian Pipeline allows you to run voice synthesis locally without a GPU. Perfect for testing, development, and small-scale data generation.

⚠️ **Warning:** CPU processing is 10-20x slower than GPU. Use only for small datasets (<50 samples).

---

## Files

| File | Purpose |
|------|---------|
| `xtts_indian_pipeline_cpu.py` | Main CPU-only TTS pipeline |
| `test_xtts_indian_cpu.py` | Test script for CPU pipeline |

---

## Quick Start

### 1. Test with 5 Samples (Recommended First Run)

```bash
python3 test_xtts_indian_cpu.py --dataset-size 5
```

**Output:** `xtts_test_cpu/` directory with audio files  
**Time:** ~15-30 seconds

### 2. Test with 10 Samples

```bash
python3 test_xtts_indian_cpu.py --dataset-size 10 --speaker-wav Recording_13.wav
```

**Time:** ~30-60 seconds

### 3. Synthesize Single Text

```bash
python3 xtts_indian_pipeline_cpu.py \
  --text "fifteen kilogram paracetamol" \
  --speaker-wav Recording_13.wav
```

**Output:** `xtts_output_cpu/output.wav`  
**Time:** ~2-5 seconds

---

## Performance Comparison

| Dataset Size | GPU Time | CPU Time | Recommended |
|--------------|----------|----------|-------------|
| 5 samples | ~5-10 sec | ~15-30 sec | ✅ CPU OK |
| 10 samples | ~10-20 sec | ~30-60 sec | ✅ CPU OK |
| 50 samples | ~1-2 min | ~5-10 min | ⚠️ CPU Slow |
| 100 samples | ~2-4 min | ~10-20 min | ❌ Use GPU |
| 500 samples | ~10-20 min | ~50-100 min | ❌ Use GPU |
| 5000 samples | ~60-90 min | ~8-15 hours | ❌ GPU Only |

---

## Command Reference

### test_xtts_indian_cpu.py

Generate complete dataset with TTS on CPU.

```bash
python3 test_xtts_indian_cpu.py [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-size` | 10 | Number of samples to generate |
| `--speaker-wav` | `Recording_13.wav` | Path to speaker voice sample |
| `--output-dir` | `xtts_test_cpu` | Output directory |
| `--max-samples` | None | Max samples per split (None = all) |

**Examples:**

```bash
# Quick test
python3 test_xtts_indian_cpu.py --dataset-size 5

# Custom output
python3 test_xtts_indian_cpu.py --dataset-size 10 --output-dir my_test

# Limit processing
python3 test_xtts_indian_cpu.py --dataset-size 50 --max-samples 10
```

### xtts_indian_pipeline_cpu.py

Direct TTS synthesis (no dataset generation).

```bash
python3 xtts_indian_pipeline_cpu.py [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--text` | None | Text to synthesize |
| `--dataset` | None | Path to existing dataset |
| `--speaker-wav` | `Recording_13.wav` | Speaker voice sample |
| `--output-dir` | `xtts_output_cpu` | Output directory |
| `--max-samples` | None | Max samples to process |

**Examples:**

```bash
# Synthesize single text
python3 xtts_indian_pipeline_cpu.py \
  --text "twenty pieces aspirin" \
  --speaker-wav Recording_13.wav

# Process existing dataset (first 20 samples)
python3 xtts_indian_pipeline_cpu.py \
  --dataset training_output/dataset \
  --max-samples 20
```

---

## Output Structure

```
xtts_test_cpu/
├── dataset/
│   ├── train/
│   │   ├── metadata.csv
│   │   ├── structured.jsonl
│   │   └── utterances.txt
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── audio/
    ├── train/
    │   ├── wavs/
    │   │   ├── utt_000000.wav
    │   │   ├── utt_000001.wav
    │   │   └── ...
    │   └── metadata.csv
    ├── val/
    │   ├── wavs/
    │   └── metadata.csv
    └── test/
        ├── wavs/
        └── metadata.csv
```

### Metadata Format

**File:** `audio/train/metadata.csv`  
**Format:** Pipe-delimited (`|`)

```
utt_000000|fifteen kilogram paracetamol
utt_000001|twenty pieces aspirin
utt_000002|three litre cetzine
```

---

## System Requirements

### Minimum
- **CPU:** Any modern CPU (2+ cores)
- **RAM:** 4 GB free
- **Disk:** 2 GB free (for model download)
- **Python:** 3.8+

### Recommended
- **CPU:** 8+ cores (faster processing)
- **RAM:** 8 GB free
- **Disk:** 5 GB free
- **Python:** 3.10+

---

## Key Features

### CPU-Specific Features

1. **Force CPU Mode**
   - Ignores GPU even if available
   - Ensures consistent local testing

2. **Progress Tracking**
   - Shows percentage completion
   - Displays estimated time remaining

3. **Safety Warnings**
   - Warns about large datasets
   - Suggests GPU for production

4. **Automatic Confirmation**
   - Asks before processing >20 samples
   - Prevents accidental long runs

### Same as GPU Version

1. **Voice Cloning**
   - Uses XTTS v2 for natural speech
   - Clones Indian English accent
   - High-quality 16kHz audio

2. **Smart Dataset Generation**
   - Balanced glossary coverage
   - Train/val/test splits (80/10/10)
   - Sanitized text output

3. **Thread Safety**
   - `num_workers=1` enforced
   - Prevents CUDA-like errors on CPU

---

## When to Use CPU vs GPU

### Use CPU Version When:
- ✅ Testing locally without GPU
- ✅ Developing/debugging scripts
- ✅ Generating <50 samples
- ✅ Verifying voice quality
- ✅ Prototyping new features
- ✅ No GPU access

### Use GPU Version When:
- ✅ Generating >100 samples
- ✅ Production data generation
- ✅ Time-sensitive projects
- ✅ GPU available
- ✅ Training data preparation
- ✅ Large-scale deployment

---

## Troubleshooting

### Problem: Too Slow

**Symptoms:**
- Processing takes forever
- One sample takes >10 seconds

**Solutions:**
1. Reduce `--dataset-size` to 5-10
2. Use `--max-samples` to limit processing
3. Switch to GPU version

### Problem: Out of Memory

**Symptoms:**
- Python crashes
- "MemoryError" or "Killed"

**Solutions:**
1. Close other applications
2. Reduce `--dataset-size`
3. Process in smaller batches using `--max-samples`
4. Upgrade RAM to 8GB+

### Problem: Model Download Fails

**Symptoms:**
- "Failed to download model"
- Connection timeout

**Solutions:**
1. Check internet connection
2. Retry (download may resume)
3. Use VPN if blocked
4. Manual download: https://github.com/coqui-ai/TTS

### Problem: Poor Audio Quality

**Symptoms:**
- Robotic voice
- Pronunciation errors
- Background noise

**Solutions:**
1. Use better quality `Recording_13.wav`
2. Ensure speaker WAV is clean (no noise)
3. Recording should be 5-10 seconds long
4. Use 16kHz+ sample rate

---

## Best Practices

### 1. Start Small
Always test with 5-10 samples first to verify:
- Voice quality
- Processing time
- Output format

### 2. Use Quality Speaker Sample
- Clean recording (no background noise)
- 5-10 seconds duration
- Clear speech
- 16kHz or higher sample rate

### 3. Monitor Resources
```bash
# Check CPU usage
htop

# Check RAM usage
free -h

# Check disk space
df -h
```

### 4. Batch Processing
For larger datasets, process in batches:

```bash
# Process first 20
python3 test_xtts_indian_cpu.py --dataset-size 100 --max-samples 20

# Then next 20, etc.
```

### 5. Verify Output
After generation:

```bash
# Count files
find xtts_test_cpu/audio/train/wavs -name "*.wav" | wc -l

# Check file sizes
ls -lh xtts_test_cpu/audio/train/wavs/ | head -10

# Play sample audio
ffplay xtts_test_cpu/audio/train/wavs/utt_000000.wav
```

---

## Comparison: CPU vs GPU

| Feature | CPU Version | GPU Version |
|---------|------------|-------------|
| **Speed** | 2-5 sec/sample | 0.3-0.5 sec/sample |
| **Hardware** | Any CPU | NVIDIA GPU required |
| **Setup** | Easy | Requires CUDA |
| **Cost** | Free (local) | GPU rental cost |
| **Quality** | Identical | Identical |
| **Use Case** | Testing, dev | Production |
| **Dataset Size** | <50 samples | 100-50000 samples |
| **Memory** | 4-6 GB RAM | 8-16 GB VRAM |

---

## Migration Guide

### From CPU to GPU

When ready to scale up:

1. **Transfer to GPU Server:**
   ```bash
   # On local machine
   scp -r upgraded_voice_pipeline/ user@gpu-server:/path/
   ```

2. **Switch to GPU Script:**
   ```bash
   # On GPU server
   python3 test_xtts_indian.py --dataset-size 5000
   ```

3. **Same Parameters:**
   - All parameters are identical
   - No code changes needed
   - Just use GPU script

### From GPU to CPU (for testing)

To test on local machine:

1. **Copy to Local:**
   ```bash
   scp -r user@gpu-server:/path/upgraded_voice_pipeline/ ./
   ```

2. **Use CPU Script:**
   ```bash
   python3 test_xtts_indian_cpu.py --dataset-size 10
   ```

---

## FAQ

**Q: Is audio quality different on CPU?**  
A: No, CPU and GPU produce identical audio quality. Only processing speed differs.

**Q: Can I use multiple CPUs?**  
A: No, `num_workers` must be 1 because XTTS v2 is not thread-safe.

**Q: How much faster is GPU?**  
A: GPU is 10-20x faster (0.3-0.5 sec vs 2-5 sec per sample).

**Q: Can I run both versions?**  
A: Yes, they use different output directories by default.

**Q: Will GPU version use CPU if no GPU?**  
A: Yes, GPU version falls back to CPU automatically.

**Q: Then why use CPU version?**  
A: CPU version forces CPU mode for testing, even if GPU exists. Useful for debugging.

---

## Support

For issues or questions:

1. Check this README
2. See main documentation: `README.md`
3. Review: `WHISPER_QUICKSTART.md`
4. Check: `DEPENDENCIES.md`

---

**Last Updated:** November 2025  
**Tested On:** Ubuntu 22.04, Python 3.10, Intel i7/i9, AMD Ryzen 7/9





