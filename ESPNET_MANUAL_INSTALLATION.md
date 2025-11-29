# ESPnet Manual Installation Guide

## ⚠️ Important Notice

**ESPnet installation is VERY complex** and the automated script cannot handle all cases. Here's what you need to know:

### The Error You're Seeing:
```
Error installing ESPnet dependencies: [Errno 2] No such file or directory: './tools/check_install.sh'
```

**Root Cause:** ESPnet's repository structure has changed over time. The automated installation script expects files that may not exist in the current version.

---

## 🤔 Do You Really Need ESPnet?

**Short Answer:** Probably not! Here's why:

| Feature | ESPnet | Whisper Fine-tuning | Wav2Vec2 |
|---------|--------|-------------------|----------|
| **Setup Difficulty** | Very Hard (hours) | Easy (minutes) | Medium (30 mins) |
| **Dependencies** | sox, kaldi, cmake, etc. | Just pip packages | Just pip packages |
| **Training Time** | Long | Fast | Fast |
| **Best For** | Research, max customization | Production, quick results | Low-resource languages |
| **Documentation** | Complex | Excellent | Good |
| **Our Use Case** | Overkill | ✅ Perfect | ✅ Good |

**Recommendation:** Use **Whisper fine-tuning** instead! It's:
- ✅ Much easier to set up
- ✅ Works with your TTS data directly
- ✅ Better for pharmaceutical vocabulary
- ✅ Faster training
- ✅ Easier deployment

---

## 📋 Option 1: Manual ESPnet Installation (If You Insist)

### Step 1: Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    sox libsox-fmt-all \
    ffmpeg \
    git cmake make \
    build-essential \
    python3-dev \
    libsndfile1-dev

# Install Kaldi (ESPnet dependency)
cd ~
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j 4

cd ../src
./configure --shared
make depend -j 4
make -j 4
```

**Time Required:** 1-2 hours for Kaldi alone! ⏰

### Step 2: Install ESPnet

```bash
# Clone ESPnet (you already did this)
cd ~
# git clone https://github.com/espnet/espnet.git  # Already done

cd espnet

# Install tools
cd tools
make

# Set up Python environment
./setup_anaconda.sh anaconda espnet 3.9

# Activate environment
source ~/anaconda3/bin/activate espnet

# Install ESPnet
pip install -e .
```

**Time Required:** 30-60 minutes ⏰

### Step 3: Verify Installation

```bash
source ~/anaconda3/bin/activate espnet
python3 -c "import espnet; print(espnet.__version__)"
```

---

## 🚀 Option 2: Whisper Fine-tuning (RECOMMENDED)

**Total setup time:** 5-10 minutes! ⚡

### Step 1: Install Dependencies

```bash
# In your virtual environment
pip install transformers datasets evaluate jiwer accelerate
```

### Step 2: Prepare Data

Your TTS data is already in the perfect format!
```
training_output/audio/
├── train/
│   ├── wavs/
│   └── metadata.csv
├── val/
│   ├── wavs/
│   └── metadata.csv
└── test/
    ├── wavs/
    └── metadata.csv
```

### Step 3: Create Whisper Training Script

Create `whisper_finetune.py`:

```python
#!/usr/bin/env python3
"""
Whisper Fine-tuning for Pharmaceutical ASR
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

def load_data(data_dir):
    """Load data from metadata.csv files"""
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, split, 'metadata.csv')
        df = pd.read_csv(csv_path, names=['file_id', 'text'])
        
        # Add full audio paths
        df['audio'] = df['file_id'].apply(
            lambda x: os.path.join(data_dir, split, 'wavs', f'{x}.wav')
        )
        
        datasets[split] = Dataset.from_pandas(df)
    
    # Cast audio column
    dataset = DatasetDict({
        'train': datasets['train'].cast_column('audio', Audio(sampling_rate=16000)),
        'validation': datasets['val'].cast_column('audio', Audio(sampling_rate=16000)),
        'test': datasets['test'].cast_column('audio', Audio(sampling_rate=16000))
    })
    
    return dataset

def main():
    # Configuration
    model_name = "openai/whisper-small"  # or "whisper-base", "whisper-medium"
    data_dir = "training_output/audio"
    output_dir = "whisper_pharma_model"
    
    # Load model and processor
    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Load data
    print("Loading training data...")
    dataset = load_data(data_dir)
    
    # Preprocessing function
    def prepare_dataset(batch):
        audio = batch["audio"]
        
        # Compute input features
        batch["input_features"] = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Encode target text
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        
        return batch
    
    # Preprocess datasets
    print("Preprocessing data...")
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset.column_names["train"]
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=True,  # Use mixed precision on GPU
        report_to=["tensorboard"],
    )
    
    # Metrics
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Decode predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    # Train!
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
```

### Step 4: Train!

```bash
python3 whisper_finetune.py
```

**Training time on RTX 4090:** ~30-60 minutes for 5000 samples! 🚀

---

## 📊 Comparison: ESPnet vs Whisper

### ESPnet Setup Process:
1. ⏰ Install Kaldi (1-2 hours)
2. ⏰ Install ESPnet tools (30-60 mins)
3. ⏰ Configure recipe (30 mins)
4. ⏰ Debug issues (1-3 hours usually!)
5. ⏰ Train model (hours)

**Total:** 4-8 hours minimum 😰

### Whisper Setup Process:
1. ⚡ `pip install transformers datasets` (2 mins)
2. ⚡ Create training script (5 mins)
3. ⚡ Train model (30-60 mins)

**Total:** 40-70 minutes 🎉

---

## 🎯 What to Do Right Now

### Recommendation: Skip ESPnet!

**Instead:**

1. **Stop the ESPnet installation** (it's getting too complex)

2. **Use your existing TTS data with Whisper:**
   ```bash
   cd ~/scripts/upgraded_voice_pipeline
   
   # Install Whisper dependencies
   pip install transformers datasets evaluate jiwer accelerate
   
   # Create and run Whisper fine-tuning
   # (I can provide the complete script if you switch to agent mode)
   ```

3. **Benefits:**
   - ✅ Works with your data as-is
   - ✅ No complex dependencies
   - ✅ Faster training
   - ✅ Better results for your use case
   - ✅ Easier to deploy

---

## 🆘 If You Still Want ESPnet

Follow the official guide:
- **ESPnet Installation:** https://espnet.github.io/espnet/installation.html
- **ESPnet ASR Tutorial:** https://espnet.github.io/espnet/tutorial.html

**Warning:** Be prepared to spend 4-8 hours debugging dependencies! 😅

---

## 📝 Summary

| What | Status | Recommendation |
|------|--------|----------------|
| **TTS Data Generation** | ✅ Complete | Keep it! |
| **ESPnet Setup** | ❌ Complex/Broken | Skip it! |
| **Whisper Fine-tuning** | 🌟 Recommended | Use this! |

**Next Step:** Let me help you set up Whisper fine-tuning instead! It's **10x easier** and will give you better results faster. 🚀

---

## 💡 Want to Continue with ESPnet Anyway?

If you're determined to use ESPnet, here's what you need to do manually:

```bash
# 1. Fix the current error by installing ESPnet properly
cd /root/espnet
pip install -e .

# 2. Install required Python packages
pip install torch torchaudio
pip install espnet-model-zoo

# 3. Follow official ESPnet recipe for custom data
# https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE
```

But seriously, **consider Whisper instead!** 😊









