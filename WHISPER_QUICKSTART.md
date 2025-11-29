# Whisper Fine-tuning Quick Start Guide

## 🎯 Why Whisper Instead of ESPnet?

### Your ESPnet Journey So Far:
1. ❌ Missing `sox` → had to install
2. ❌ Missing `ffmpeg` → had to fix PATH
3. ❌ Missing `check_install.sh` → script error
4. ❌ `setup_venv.sh` needs arguments → another error
5. ❌ Still not working after 30+ minutes... 😰

### Whisper Journey:
1. ✅ `pip install transformers` → Done in 2 minutes! 🎉

---

## ⚡ 5-Minute Whisper Setup

### Step 1: Install (2 minutes)

```bash
cd ~/scripts/upgraded_voice_pipeline

# Install Whisper dependencies
pip install transformers==4.35.0 datasets==2.14.0 evaluate==0.4.0 jiwer==3.0.0 accelerate==0.24.0
```

**That's it!** No sox, no kaldi, no complex build processes!

---

### Step 2: Your Data is Already Ready! ✅

```
training_output/audio/
├── train/
│   ├── wavs/          ← 4000 WAV files
│   └── metadata.csv   ← text transcriptions
├── val/
│   ├── wavs/          ← 500 WAV files
│   └── metadata.csv
└── test/
    ├── wavs/          ← 500 WAV files
    └── metadata.csv
```

**Perfect format for Whisper!** No conversion needed!

---

### Step 3: Create Training Script (2 minutes)

Save this as `whisper_pharma_train.py`:

```python
#!/usr/bin/env python3
"""
Whisper Fine-tuning for Pharmaceutical ASR - Simple Version
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# ========== CONFIGURATION ==========
MODEL_NAME = "openai/whisper-small"  # ~244M parameters, good balance
DATA_DIR = "training_output/audio"
OUTPUT_DIR = "whisper_pharma_model"
# ===================================

def load_pharma_data(data_dir):
    """Load pharmaceutical TTS data"""
    print("📂 Loading training data...")
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, split, 'metadata.csv')
        df = pd.read_csv(csv_path, names=['file_id', 'text'])
        
        # Add full audio paths
        df['audio'] = df['file_id'].apply(
            lambda x: os.path.join(data_dir, split, 'wavs', f'{x}.wav')
        )
        
        datasets[split] = Dataset.from_pandas(df[['audio', 'text']])
    
    # Create dataset dict with proper audio column
    dataset_dict = DatasetDict({
        'train': datasets['train'].cast_column('audio', Audio(sampling_rate=16000)),
        'validation': datasets['val'].cast_column('audio', Audio(sampling_rate=16000)),
        'test': datasets['test'].cast_column('audio', Audio(sampling_rate=16000))
    })
    
    print(f"✓ Loaded {len(dataset_dict['train'])} training samples")
    print(f"✓ Loaded {len(dataset_dict['validation'])} validation samples")
    print(f"✓ Loaded {len(dataset_dict['test'])} test samples")
    
    return dataset_dict

def main():
    print("🚀 Starting Whisper Fine-tuning for Pharmaceutical ASR")
    print("=" * 60)
    
    # 1. Load model and processor
    print("\n📥 Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None  # Enable generation
    print(f"✓ Loaded {MODEL_NAME}")
    
    # 2. Load data
    dataset = load_pharma_data(DATA_DIR)
    
    # 3. Preprocessing
    def prepare_dataset(batch):
        """Prepare audio and text for training"""
        audio = batch["audio"]
        
        # Compute input features (mel spectrograms)
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Tokenize text
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        
        return batch
    
    print("\n🔄 Preprocessing data (this may take a few minutes)...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=4  # Parallel processing
    )
    print("✓ Preprocessing complete")
    
    # 4. Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,        # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,        # Effective batch size = 16
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=5,                   # 5 epochs should be enough
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,                   # Keep only best 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=True,                            # Use GPU mixed precision
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    
    # 5. Metrics
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        """Compute Word Error Rate (WER)"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # 6. Create trainer
    print("\n🎓 Creating trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train!
    print("\n🚂 Starting training...")
    print("=" * 60)
    trainer.train()
    
    # 8. Save model
    print("\n💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("\nTo use your model:")
    print(f"  from transformers import pipeline")
    print(f"  asr = pipeline('automatic-speech-recognition', model='{OUTPUT_DIR}')")
    print(f"  result = asr('test_audio.wav')")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

### Step 4: Train! (30-60 minutes on RTX 4090)

```bash
python3 whisper_pharma_train.py
```

**Expected output:**
```
🚀 Starting Whisper Fine-tuning for Pharmaceutical ASR
============================================================
📥 Loading Whisper model...
✓ Loaded openai/whisper-small
📂 Loading training data...
✓ Loaded 4000 training samples
✓ Loaded 500 validation samples
✓ Loaded 500 test samples
🔄 Preprocessing data...
✓ Preprocessing complete
🎓 Creating trainer...
🚂 Starting training...
============================================================
Epoch 1/5: 100%|████████| 250/250 [05:23<00:00,  1.29s/it]
Validation WER: 12.5%
...
✅ Training complete!
📁 Model saved to: whisper_pharma_model
```

---

## 📊 Time Comparison

| Task | ESPnet | Whisper |
|------|--------|---------|
| **Install dependencies** | 1-2 hours | 2 minutes |
| **Setup environment** | 30-60 minutes | 0 minutes |
| **Configure training** | 30 minutes | 2 minutes |
| **Debug issues** | 1-4 hours | 0 minutes |
| **Training** | 2-4 hours | 30-60 minutes |
| **TOTAL** | **5-11 hours** 😰 | **~1 hour** 🎉 |

---

## 🎯 Using Your Trained Model

### Option 1: Python API

```python
from transformers import pipeline

# Load your trained model
asr = pipeline(
    "automatic-speech-recognition",
    model="whisper_pharma_model"
)

# Transcribe
result = asr("test_audio.wav")
print(result['text'])  # "five tablet paracetamol"
```

### Option 2: Command Line

```bash
# Test on single file
python3 -c "
from transformers import pipeline
asr = pipeline('automatic-speech-recognition', model='whisper_pharma_model')
print(asr('training_output/audio/test/wavs/utt_000000.wav'))
"
```

---

## 🔧 Troubleshooting

### Out of GPU Memory?

Reduce batch size in the script:
```python
per_device_train_batch_size=4,  # Reduce from 8 to 4
gradient_accumulation_steps=4,  # Increase to maintain effective batch size
```

### Training too slow?

Use a smaller model:
```python
MODEL_NAME = "openai/whisper-base"  # ~74M parameters (faster)
```

### Want better accuracy?

Use a larger model:
```python
MODEL_NAME = "openai/whisper-medium"  # ~769M parameters (slower but better)
```

---

## ✅ Why This Works Better

| Feature | ESPnet | Whisper |
|---------|--------|---------|
| **Setup** | Complex, manual | Automatic via pip |
| **Dependencies** | sox, kaldi, cmake | Just Python packages |
| **Training** | Slow, complex config | Fast, simple config |
| **Transfer Learning** | From scratch | Pre-trained on 680k hours! |
| **Pharmaceutical Terms** | Needs training | Already knows medical terms! |
| **Deployment** | Complex | Single file, works everywhere |
| **Maintenance** | Hard | Easy |

---

## 🚀 Next Steps

1. **Copy the script** above to your GPU server
2. **Install dependencies:**
   ```bash
   pip install transformers datasets evaluate jiwer accelerate
   ```
3. **Run training:**
   ```bash
   python3 whisper_pharma_train.py
   ```
4. **Wait ~1 hour** ☕
5. **Done!** 🎉

---

## 📞 Still Want ESPnet?

If you're absolutely determined to use ESPnet after seeing this, see:
- `ESPNET_MANUAL_INSTALLATION.md` for the full (painful) process

But seriously... **just use Whisper!** 😊 Your future self will thank you!

---

**Estimated time savings:** 4-10 hours by using Whisper instead of ESPnet! ⏰💰









