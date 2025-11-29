# Training Data Generation for ASR Models

This guide explains how to use the training data generation pipeline to create a dataset for training ASR models specialized in pharmaceutical domain.

## Overview

We've created a complete pipeline for generating training data:

1. **Smart Dataset Generator**: Creates a text dataset with balanced coverage of all glossary items
2. **TTS Pipeline**: Converts text to speech using various TTS engines
3. **Data Augmentation**: Applies acoustic augmentations to increase dataset size and variability

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install numpy scipy librosa soundfile gtts pyttsx3
sudo apt-get install espeak ffmpeg
```

### Quick Start

For a quick test of the pipeline, run:

```bash
./run_data_generation.sh
```

This will generate a small dataset (5000 utterances), create audio files using eSpeak, and apply basic augmentations.

## Cloud GPU Options for Training

Once you've generated the dataset, you can use various cloud GPU options to train your ASR model:

1. **Google Colab Pro** ($9.99/month)
   - Provides access to T4/P100 GPUs
   - Good for training small to medium models
   - Easy to use with Jupyter notebooks

2. **Kaggle Kernels**
   - Free tier with P100 GPU (30 hours/week)
   - Good for training and experimentation

3. **Paperspace Gradient**
   - Pay-as-you-go pricing starting at ~$0.30/hour for basic GPUs
   - More powerful options available as needed

4. **Lambda Labs**
   - Competitive pricing (~$0.80/hour for A100)
   - Developer-friendly

5. **Vast.ai**
   - Marketplace for renting GPUs from individuals
   - Often has very competitive rates (as low as $0.20/hour)

## Training Approaches

For your specific use case with medical terms, we recommend:

1. **Fine-tune a smaller model**:
   - Fine-tune Whisper-tiny or Whisper-base on your generated dataset
   - These can run on CPU for inference after training
   - Focus on domain adaptation with your specific vocabulary

2. **Use Vosk with custom model**:
   - Train a Kaldi model using the generated dataset
   - Convert to Vosk format for deployment
   - This approach works well on CPU-only systems

## Customization

### Glossary Customization

Edit the `pharma_glossary.json` file to add or modify terms:
- Add more products to the `product_glossary` section
- Add more units to the `unit_glossary` section
- Add more templates to the `templates` section

### Dataset Size

Adjust the dataset size based on your needs:
- For a small test dataset: 1,000-5,000 utterances
- For a medium dataset: 10,000-50,000 utterances
- For a large dataset: 100,000+ utterances

### TTS Engine Selection

Choose the TTS engine based on your requirements:
- `gtts`: Google Text-to-Speech (online, high quality, slow)
- `espeak`: eSpeak (offline, lower quality, fast)
- `pyttsx3`: Local TTS engine (offline, quality varies)

## Full Pipeline Command

Run the complete pipeline with custom settings:

```bash
python3 generate_training_data.py \
  --dataset-dir dataset \
  --tts-dir tts_output \
  --augmented-dir augmented_data \
  --dataset-size 30000 \
  --min-occurrences 20 \
  --glossary-file pharma_glossary.json \
  --tts-engine espeak \
  --num-augmentations 5 \
  --num-workers 4 \
  --sample-rate 16000
```

## Conclusion

This pipeline provides a complete solution for generating training data for ASR models specialized in pharmaceutical domain. By using this pipeline, you can create a high-quality dataset with good coverage of all glossary items, which is essential for training robust ASR models.

For more detailed information, refer to the `TRAINING_DATA_README.md` file.















