# Training Data Generation Pipeline

This directory contains scripts to generate training data for ASR models specialized in pharmaceutical domain. The pipeline consists of three main steps:

1. **Dataset Generation**: Creates a text dataset with balanced coverage of all glossary items
2. **TTS Audio Generation**: Converts text to speech using various TTS engines
3. **Data Augmentation**: Applies acoustic augmentations to increase dataset size and variability

## Quick Start

For a quick test of the pipeline, run:

```bash
./run_data_generation.sh
```

This will generate a small dataset (5000 utterances), create audio files using eSpeak, and apply basic augmentations.

## Components

### 1. Smart Dataset Generator

The `smart_dataset_generator.py` script creates a dataset with balanced coverage of all glossary items:

```bash
python3 smart_dataset_generator.py \
  --output-dir dataset \
  --glossary-file pharma_glossary.json \
  --dataset-size 30000 \
  --min-occurrences 20
```

Key options:
- `--dataset-size`: Target number of utterances
- `--min-occurrences`: Minimum occurrences for each glossary item
- `--glossary-file`: Custom glossary file (JSON format)

### 2. TTS Pipeline

The `tts_pipeline.py` script converts text to speech:

```bash
python3 tts_pipeline.py \
  --engine espeak \
  --output-dir tts_output \
  --dataset dataset
```

Supported TTS engines:
- `gtts`: Google Text-to-Speech (online, high quality, slow)
- `espeak`: eSpeak (offline, lower quality, fast)
- `pyttsx3`: Local TTS engine (offline, quality varies)

### 3. Data Augmentation

The `data_augmentation.py` script applies acoustic augmentations:

```bash
python3 data_augmentation.py \
  --output-dir augmented_data \
  --dataset tts_output \
  --num-augmentations 5
```

Available augmentations:
- Noise addition
- Speed perturbation
- Pitch shifting
- Room impulse response (RIR) simulation
- Time masking

## Complete Pipeline

The `generate_training_data.py` script runs the complete pipeline:

```bash
python3 generate_training_data.py \
  --dataset-dir dataset \
  --tts-dir tts_output \
  --augmented-dir augmented_data \
  --dataset-size 30000 \
  --tts-engine espeak \
  --num-augmentations 5
```

You can skip specific steps if needed:
- `--skip-dataset`: Skip dataset generation
- `--skip-tts`: Skip TTS audio generation
- `--skip-augmentation`: Skip data augmentation

## Custom Glossaries

The `pharma_glossary.json` file contains custom glossaries for pharmaceutical terms. You can modify this file to add or remove terms as needed.

## Output Format

The generated dataset follows a standard structure:
- `dataset/`: Text dataset
  - `train/`, `val/`, `test/`: Train/validation/test splits
  - `metadata.csv`: Utterance metadata
  - `utterances.txt`: Plain text utterances
  - `structured.jsonl`: Structured data (JSON Lines format)
- `tts_output/`: Audio files
  - `train/wavs/`, `val/wavs/`, `test/wavs/`: WAV files
  - `metadata.csv`: Updated metadata with audio file paths
- `augmented_data/`: Augmented audio files
  - Similar structure to `tts_output/` but with augmented files

## Requirements

- Python 3.6+
- Required packages: numpy, scipy, librosa, soundfile
- TTS engines: gtts, espeak, pyttsx3 (at least one)

## Training ASR Models

The generated dataset can be used to train various ASR models:

1. **ESPnet**:
   - Use the `--espnet` flag to generate ESPnet-compatible files
   - Follow ESPnet documentation for training

2. **Kaldi**:
   - Use the `--kaldi` flag to generate Kaldi-compatible files
   - Follow Kaldi documentation for training

3. **Vosk**:
   - Vosk models can be trained using Kaldi
   - Follow Vosk documentation for converting Kaldi models to Vosk format

4. **Whisper Fine-tuning**:
   - Use the generated dataset to fine-tune Whisper models
   - Follow OpenAI's documentation for fine-tuning















