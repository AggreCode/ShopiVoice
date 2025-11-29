# CUDA Errors Fixed - Summary

## Problem Identified

The CUDA errors (`srcIndex < srcSelectDimSize` assertion failed) were caused by **two main issues**:

### 1. Thread-Safety Issue (Primary Cause)
- **Problem**: XTTS v2 model is **NOT thread-safe**
- **Symptom**: Multiple threads trying to access the same model simultaneously caused race conditions
- **Result**: GPU memory corruption → CUDA crashes
- **Analogy**: Like 4 waiters grabbing the same pan from a chef's hand at once

### 2. Problematic Text Characters (Secondary Cause)
- **Problem**: Single-letter units (`l`, `g`) and special characters (`-`, `_`) confused the model
- **Example**: `"1 l band-aid"` → CUDA error
- **Result**: Model tried to access invalid memory positions

---

## ✅ Fixes Applied

### Fix 1: Set `num_workers=1` (Thread-Safety)

**Changed in:**
- `xtts_indian_pipeline.py` - default changed from 4 to 1
- `test_xtts_indian.py` - default changed from 2 to 1
- `run_training_pipeline.sh` - already set to 1

**Why:** XTTS v2 must process audio files **one at a time** (serially) to prevent race conditions.

**Documentation added:**
```python
# IMPORTANT: XTTS v2 is NOT thread-safe - must use num_workers=1
self.num_workers = self.config.get('num_workers', 1)
```

---

### Fix 2: Clean Glossary (Remove Problematic Characters)

**File:** `pharma_glossary.json`

**Changes:**

#### Before (Problematic):
```json
"unit_glossary": {
  "g": "g", "gram": "g",     // ❌ Single letter causes crashes
  "l": "l", "liter": "l",     // ❌ Single letter causes crashes
}

"quantity_glossary": {
  "0": "0", "zero": "0",      // ❌ Leading with "0" can be problematic
}
```

#### After (Fixed):
```json
"unit_glossary": {
  "gram": "gram", "grams": "gram",     // ✅ Full word, safe
  "liter": "liter", "liters": "liter", // ✅ Full word, safe
}

"quantity_glossary": {
  "zero": "0",                          // ✅ Word form only
}
```

---

### Fix 3: Sanitization at Source

**Files:** `backup/dataset_generator.py`, `updated_smart_dataset_generator.py`

**Added `_sanitize_utterance()` method:**
```python
def _sanitize_utterance(self, utterance: str) -> str:
    """
    Sanitize utterance for TTS compatibility.
    Removes problematic characters that can cause CUDA/TTS errors.
    """
    # Replace hyphens with spaces (band-aid -> band aid)
    utterance = utterance.replace('-', ' ')
    
    # Replace underscores with spaces
    utterance = utterance.replace('_', ' ')
    
    # Normalize multiple spaces to single space
    utterance = re.sub(r'\s+', ' ', utterance)
    
    # Ensure text ends with punctuation for better TTS prosody
    if utterance and utterance[-1] not in '.!?,':
        utterance = utterance + '.'
    
    return utterance
```

**Why this works:**
- Sanitization happens **during dataset generation**
- Both `metadata.csv` and audio use the **same sanitized text**
- **No train/audio mismatch** (critical for ASR training!)

---

### Fix 4: Minimal TTS Pipeline Sanitization

**File:** `xtts_indian_pipeline.py`

**Removed:**
- ❌ Unit expansion (`l` → `liter`) - causes mismatch
- ❌ Carrier phrases (`"The item is: {text}."`) - causes mismatch
- ❌ Adding extra words like `"Thank you."` - causes mismatch

**Kept only:**
- ✅ Whitespace normalization
- ✅ Adding punctuation (`.`) at end (safe - no word changes)
- ✅ Warning log for very short texts

**Why:**
```
Metadata.csv: "zestril"
Audio:        "The item is: zestril."
              ^^^^^^^^^^^^^^^^
              MISMATCH! ❌

vs.

Metadata.csv: "zestril."
Audio:        "zestril."
              PERFECT MATCH! ✅
```

---

## 📊 Results Expected

### Before Fixes:
- ❌ CUDA errors every 10-50 samples
- ❌ `"band-aid"` → crash
- ❌ `"1 l crocin"` → crash
- ❌ Race conditions with `num_workers > 1`
- ❌ Train/audio mismatches with carrier phrases

### After Fixes:
- ✅ No CUDA errors (or extremely rare)
- ✅ `"band aid."` → works
- ✅ `"1 liter crocin."` → works
- ✅ Serial processing prevents race conditions
- ✅ Perfect train/audio alignment

---

## 🚀 Usage on GPU Server

### Test Command:
```bash
# Test with small dataset first
python3 test_xtts_indian.py \
  --dataset-size 10 \
  --max-samples 10 \
  --speaker-wav Recording_13.wav \
  --output-dir test_cuda_fix \
  --num-workers 1  # CRITICAL: Must be 1

# Verify no CUDA errors
ls test_cuda_fix/audio/train/*.wav | wc -l  # Should be 10
```

### Full Pipeline:
```bash
# Run full training data generation
bash run_training_pipeline.sh
```

### Monitor GPU Usage:
```bash
watch -n 1 nvidia-smi
```

**Expected:**
- GPU utilization: ~50-70% (normal for TTS)
- Memory usage: ~2-4GB
- Processing speed: ~1-2 seconds per utterance
- **No CUDA errors!**

---

## 🔑 Key Takeaways

1. **Thread-Safety is Critical**: XTTS v2 cannot handle parallel requests
2. **Fix at Source**: Sanitize text during dataset generation, not at TTS time
3. **No Extra Words**: Adding words creates train/audio mismatches that ruin ASR training
4. **Full Words Only**: Avoid single-letter units in glossary
5. **Punctuation is Safe**: Adding `.` at end doesn't affect ASR training labels

---

## 📝 Files Modified

1. `pharma_glossary.json` - Removed single-letter units
2. `backup/dataset_generator.py` - Added `_sanitize_utterance()`
3. `updated_smart_dataset_generator.py` - Applied sanitization in all phases
4. `xtts_indian_pipeline.py` - Simplified sanitization, changed default `num_workers=1`
5. `test_xtts_indian.py` - Changed default `num_workers=1`
6. `run_training_pipeline.sh` - Already had `NUM_WORKERS=1`

---

## 🎯 Next Steps

1. ✅ Copy all files to GPU server
2. ✅ Run test with 10 samples
3. ✅ Verify no CUDA errors
4. ✅ Run full pipeline with 5000+ samples
5. ✅ Start ESPnet training

**You're ready to roll!** 🚀









