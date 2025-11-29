#!/usr/bin/env python3
"""
Whisper Fine-tuning for Pharmaceutical ASR

This script fine-tunes OpenAI's Whisper model on pharmaceutical TTS data.
Much simpler than ESPnet - just works!
"""

import os
import sys
import argparse
import pandas as pd
import dataclasses
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding
)
import evaluate
import torch

def load_pharma_data(data_dir):
    """Load pharmaceutical TTS data from metadata.csv files"""
    print("📂 Loading training data...")
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(data_dir, split, 'metadata.csv')
        
        if not os.path.exists(csv_path):
            print(f"⚠️  Warning: {csv_path} not found, skipping {split} split")
            continue
        
        print(f"📄 Reading {csv_path}...")
        
        # Try multiple parsing strategies
        df = None
        
        # Strategy 1: CSV with headers (standard format from TTS pipeline)
        try:
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
            print(f"   ✓ Parsed CSV with headers, {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)}")
            
            # Check if we have the expected columns
            if 'id' in df.columns and 'utterance' in df.columns:
                # Use 'id' and 'utterance' columns
                df = df[['id', 'utterance']].copy()
                df.columns = ['file_id', 'text']
                print(f"   ✓ Using columns ['id', 'utterance'] as [file_id, text]")
            elif 'file_id' in df.columns and 'text' in df.columns:
                # Already in correct format
                df = df[['file_id', 'text']].copy()
                print(f"   ✓ Using columns ['file_id', 'text']")
            elif len(df.columns) >= 2:
                # Use first 2 columns
                df = df.iloc[:, :2].copy()
                df.columns = ['file_id', 'text']
                print(f"   ✓ Using first 2 columns as [file_id, text]")
            else:
                df = None
                print(f"   ✗ Not enough columns")
        except Exception as e:
            print(f"   ✗ CSV parsing failed: {e}")
        
        # Strategy 2: Pipe-delimited, no header
        if df is None or len(df) == 0:
            try:
                df = pd.read_csv(csv_path, sep='|', engine='python', header=None, 
                               on_bad_lines='skip', skipinitialspace=True)
                print(f"   ✓ Parsed with pipe delimiter (|), {len(df)} rows")
                if len(df.columns) >= 2:
                    df = df.iloc[:, :2].copy()
                    df.columns = ['file_id', 'text']
                else:
                    df = None
            except Exception as e:
                print(f"   ✗ Pipe parsing failed: {e}")
        
        # Strategy 3: Comma-delimited, no header
        if df is None or len(df) == 0:
            try:
                df = pd.read_csv(csv_path, names=['file_id', 'text'], 
                               header=None, on_bad_lines='skip')
                print(f"   ✓ Parsed with comma delimiter, {len(df)} rows")
            except Exception as e:
                print(f"   ✗ Comma parsing failed: {e}")
        
        if df is None or len(df) == 0:
            print(f"   ⚠️  Could not parse {csv_path}, skipping...")
            continue
        
        # Clean data
        try:
            print(f"   Before cleaning: {len(df)} rows")
            
            # Convert to string first (handles NaN gracefully)
            df['file_id'] = df['file_id'].astype(str).str.strip()
            df['text'] = df['text'].astype(str).str.strip()
            
            # Remove rows where EITHER column is empty, 'nan', or 'None'
            df = df[df['file_id'].str.lower() != 'nan']
            df = df[df['text'].str.lower() != 'nan']
            df = df[df['file_id'].str.lower() != 'none']
            df = df[df['text'].str.lower() != 'none']
            df = df[df['file_id'] != '']
            df = df[df['text'] != '']
            
            print(f"   After cleaning: {len(df)} rows")
            
            if len(df) > 0:
                print(f"   Sample row: {df.iloc[0]['file_id']} -> {df.iloc[0]['text'][:50]}...")
            else:
                print(f"   ⚠️  No valid rows after cleaning")
                continue
                
            # Add full audio paths
            df['audio'] = df['file_id'].apply(
                lambda x: os.path.join(data_dir, split, 'wavs', f'{x}.wav')
            )
        except Exception as e:
            print(f"   ❌ Error processing data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Filter out missing audio files
        before_filter = len(df)
        df = df[df['audio'].apply(os.path.exists)]
        after_filter = len(df)
        
        if before_filter != after_filter:
            print(f"   ⚠️  {before_filter - after_filter} audio files not found")
        
        if len(df) > 0:
            datasets[split] = Dataset.from_pandas(df[['audio', 'text']])
            print(f"   ✓ {len(df)} samples ready for {split}")
        else:
            print(f"   ⚠️  No valid samples found for {split}")
    
    if not datasets:
        raise ValueError(f"No data found in {data_dir}. Check that metadata.csv files exist and audio files are present.")
    
    # Create dataset dict (don't cast to Audio - we'll load manually with soundfile)
    dataset_dict = {}
    for split, ds in datasets.items():
        dataset_dict[split] = ds
        print(f"✓ Loaded {len(ds)} {split} samples")
    
    return DatasetDict(dataset_dict)

def main():
    parser = argparse.ArgumentParser(description="Whisper Fine-tuning for Pharmaceutical ASR")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--audio_dir", type=str, default="training_output/audio",
                        help="Directory containing train/val/test splits")
    parser.add_argument("--output_dir", type=str, default="whisper_pharma_model",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs (alias for --epochs)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--language", type=str, default="english",
                        help="Language for training")
    
    # Backward compatibility arguments (hidden)
    parser.add_argument("--model-name", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=str, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle argument aliases
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"openai/whisper-{args.model_size}"
        
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = args.audio_dir

    if args.num_epochs:
        epochs = args.num_epochs
    else:
        epochs = args.epochs
    
    print("🚀 Starting Whisper Fine-tuning for Pharmaceutical ASR")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Language: {args.language}")
    print("=" * 60)
    
    # 1. Load model and processor
    print("\n📥 Loading Whisper model...")
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Set language and task
        model.generation_config.language = args.language.lower()
        model.generation_config.task = "transcribe"
        
        # Enable generation without forced tokens
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        print(f"✓ Loaded {model_name}")
        print(f"✓ Language: {args.language}")
        print(f"✓ Task: transcribe")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # 2. Load data
    try:
        dataset = load_pharma_data(data_dir)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # 3. Preprocessing
    def prepare_dataset(batch):
        """Prepare audio and text for training"""
        import soundfile as sf
        
        # Load audio manually with soundfile (avoids torchcodec)
        audio_path = batch["audio"]
        audio_array, sampling_rate = sf.read(audio_path)
        
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
        
        # Compute input features (mel spectrograms)
        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate
        ).input_features[0]
        
        # Tokenize text
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        
        return batch
    
    print("\n🔄 Preprocessing data (this may take a few minutes)...")
    try:
        # Use num_proc=1 to avoid torchcodec issues with parallel processing
        # soundfile backend works fine in single-process mode
        dataset = dataset.map(
            prepare_dataset,
            remove_columns=dataset.column_names["train"],
            num_proc=1  # Single process to avoid torchcodec dependency
        )
        print("✓ Preprocessing complete")
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Data Collator
    from transformers import DataCollatorWithPadding
    
    @dataclasses.dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        """
        processor: WhisperProcessor
        decoder_start_token_id: int

        def __call__(self, features):
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # 5. Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use GPU mixed precision if available
        report_to=[],  # Disable tensorboard (not installed)
        push_to_hub=False,
        remove_unused_columns=False,  # Required for custom collator
    )
    
    # 6. Metrics
    # Create cache directory if it doesn't exist
    cache_dir = "/workspace/.cache/huggingface/metrics/wer/default"
    os.makedirs(cache_dir, exist_ok=True)
    
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        """Compute Word Error Rate (WER)"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # 7. Create trainer
    print("\n🎓 Creating trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("val")),
        processing_class=processor.tokenizer,  # Updated from 'tokenizer' (deprecated)
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. Train!
    print("\n🚂 Starting training...")
    print("=" * 60)
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 9. Save model
    print("\n💾 Saving final model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # 10. Test on test set if available
    if "test" in dataset:
        print("\n📊 Evaluating on test set...")
        results = trainer.evaluate(dataset["test"])
        print(f"Test WER: {results['eval_wer']:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")
    print("\nTo use your model:")
    print(f"  from transformers import pipeline")
    print(f"  asr = pipeline('automatic-speech-recognition', model='{args.output_dir}')")
    print(f"  result = asr('test_audio.wav')")
    print("=" * 60)

if __name__ == "__main__":
    main()
