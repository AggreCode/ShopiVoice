#!/bin/bash
#
# Complete Training Pipeline for Pharmaceutical Voice Commands
# =============================================================
#
# This script runs the full pipeline:
#   1. Dataset Generation (using updated_smart_dataset_generator.py)
#   2. TTS Audio Generation (using xtts_indian_pipeline.py via test_xtts_indian.py)
#   3. Data Augmentation (using data_augmentation.py) - ENABLED
#   4. ESPnet Setup (optional, commented out)
#
# Usage:
#   bash run_training_pipeline.sh [DATASET_SIZE] [MAX_SAMPLES] [OUTPUT_DIR]
#
# Examples:
#   bash run_training_pipeline.sh                    # Use defaults (5000 samples)
#   bash run_training_pipeline.sh 10000              # Generate 10000 samples
#   bash run_training_pipeline.sh 5000 100 test_run # 5000 samples, max 100 per split
#
# Requirements:
#   - Recording_13.wav (speaker voice sample)
#   - pharma_glossary.json
#   - Python packages: TTS, librosa, soundfile, numpy, scipy
#   - GPU recommended (RTX 4090 or similar)
#
# =============================================================

set -e  # Exit on error

# Enable CUDA error detection (optional - slows down but helps debug)
# Uncomment the line below if you're getting CUDA errors
# export CUDA_LAUNCH_BLOCKING=1

# ============================================
# CONFIGURATION PARAMETERS
# ============================================
DATASET_SIZE=${1:-5000}          # Total dataset size (train+val+test)
MAX_SAMPLES=${2:-}               # Max samples to process per split (empty = all)
OUTPUT_DIR=${3:-"training_output"}
AUGMENTED_DIR="${OUTPUT_DIR}_augmented"
SPEAKER_WAV="Recording_13.wav"

# TTS Parameters
NUM_WORKERS=1                    # CRITICAL: Must be 1 for XTTS (not thread-safe)
LANGUAGE="en"                    # Language code
SAMPLE_RATE=16000                # Audio sample rate

# Data Augmentation Parameters
ENABLE_AUGMENTATION=true         # Set to false to skip augmentation
NUM_AUGMENTATIONS=3              # Number of augmented versions per audio file
SNR_DB_MIN=15                    # Minimum SNR for noise (dB)
SNR_DB_MAX=25                    # Maximum SNR for noise (dB)
SPEED_MIN=0.9                    # Minimum speed factor
SPEED_MAX=1.1                    # Maximum speed factor
PITCH_MIN=-1.5                   # Minimum pitch shift (semitones)
PITCH_MAX=1.5                    # Maximum pitch shift (semitones)
AUGMENTATION_WORKERS=4           # Parallel workers for augmentation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pharmaceutical Voice Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Dataset size: ${YELLOW}${DATASET_SIZE}${NC}"
echo -e "  Max samples per split: ${YELLOW}${MAX_SAMPLES:-All}${NC}"
echo -e "  Output directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Speaker WAV: ${YELLOW}${SPEAKER_WAV}${NC}"
echo ""

# Check if speaker WAV exists
if [ ! -f "$SPEAKER_WAV" ]; then
    echo -e "${YELLOW}Warning: Speaker WAV not found at $SPEAKER_WAV${NC}"
    echo "Please update SPEAKER_WAV variable or provide the correct path"
    exit 1
fi

# Step 1: Generate TTS Dataset
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 1: Generating TTS Dataset${NC}"
echo -e "${GREEN}========================================${NC}"

if [ -z "$MAX_SAMPLES" ]; then
    # No max samples limit
    python3 test_xtts_indian.py \
        --dataset-size "$DATASET_SIZE" \
        --speaker-wav "$SPEAKER_WAV" \
        --output-dir "$OUTPUT_DIR" \
        --num-workers "$NUM_WORKERS"
else
    # With max samples limit
    python3 test_xtts_indian.py \
        --dataset-size "$DATASET_SIZE" \
        --max-samples "$MAX_SAMPLES" \
        --speaker-wav "$SPEAKER_WAV" \
        --output-dir "$OUTPUT_DIR" \
        --num-workers "$NUM_WORKERS"
fi

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Error: TTS generation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ TTS dataset generated successfully${NC}"
echo ""

# ============================================
# STEP 2: APPLY DATA AUGMENTATION
# ============================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 2: Applying Data Augmentation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ "$ENABLE_AUGMENTATION" = "true" ]; then
    echo -e "Augmentation settings:"
    echo -e "  Input: ${YELLOW}${OUTPUT_DIR}/audio${NC}"
    echo -e "  Output: ${YELLOW}${AUGMENTED_DIR}${NC}"
    echo -e "  Augmentations per file: ${YELLOW}${NUM_AUGMENTATIONS}${NC}"
    echo -e "  Workers: ${YELLOW}${AUGMENTATION_WORKERS}${NC}"
    echo ""
    
    # Run data augmentation using data_augmentation.py
    python3 data_augmentation.py \
        --dataset "$OUTPUT_DIR/audio" \
        --output-dir "${AUGMENTED_DIR}" \
        --num-augmentations "$NUM_AUGMENTATIONS" \
        --num-workers "$AUGMENTATION_WORKERS" \
        --sample-rate "$SAMPLE_RATE"
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Data augmentation failed, continuing with original data${NC}"
        FINAL_OUTPUT="$OUTPUT_DIR/audio"
    else
        echo -e "${GREEN}✓ Data augmentation completed${NC}"
        FINAL_OUTPUT="${AUGMENTED_DIR}"
        
        # Count augmented files
        AUG_TRAIN_COUNT=$(find ${AUGMENTED_DIR}/train/wavs -name "*.wav" 2>/dev/null | wc -l)
        echo -e "${GREEN}Created ${AUG_TRAIN_COUNT} augmented training files${NC}"
    fi
else
    echo -e "${YELLOW}Data augmentation is disabled${NC}"
    echo -e "To enable, set ENABLE_AUGMENTATION=true in this script"
    FINAL_OUTPUT="$OUTPUT_DIR/audio"
fi

echo ""

# ============================================
# STEP 3: PREPARE ESPNET TRAINING (OPTIONAL)
# ============================================
# Uncomment this section if you want to use ESPnet
# echo -e "${GREEN}========================================${NC}"
# echo -e "${GREEN}STEP 3: ESPnet Training Preparation${NC}"
# echo -e "${GREEN}========================================${NC}"
# 
# # Check if sox is installed (required by ESPnet)
# if ! command -v sox &> /dev/null; then
#     echo -e "${YELLOW}Warning: sox is not installed. Install it with:${NC}"
#     echo -e "${YELLOW}  Ubuntu/Debian: sudo apt-get install sox${NC}"
#     echo -e "${YELLOW}Skipping ESPnet setup...${NC}"
# else
#     python3 espnet_setup.py --data-dir "$FINAL_OUTPUT"
#     
#     if [ $? -ne 0 ]; then
#         echo -e "${YELLOW}Warning: ESPnet setup encountered issues${NC}"
#     else
#         echo -e "${GREEN}✓ ESPnet environment prepared${NC}"
#     fi
# fi
# 
# echo ""

# ============================================
# FINAL SUMMARY
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Pipeline Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display configuration
echo -e "${GREEN}Configuration Used:${NC}"
echo -e "  Dataset size: ${YELLOW}${DATASET_SIZE}${NC} samples"
echo -e "  Speaker WAV: ${YELLOW}${SPEAKER_WAV}${NC}"
echo -e "  TTS workers: ${YELLOW}${NUM_WORKERS}${NC} (XTTS thread-safe)"
echo -e "  Augmentation: ${YELLOW}${ENABLE_AUGMENTATION}${NC}"
if [ "$ENABLE_AUGMENTATION" = "true" ]; then
    echo -e "  Augmentations per file: ${YELLOW}${NUM_AUGMENTATIONS}${NC}"
    echo -e "  Augmentation workers: ${YELLOW}${AUGMENTATION_WORKERS}${NC}"
fi
echo ""

# Display output locations
echo -e "${GREEN}Output Locations:${NC}"
echo -e "  Dataset (metadata): ${YELLOW}${OUTPUT_DIR}/dataset/${NC}"
echo -e "  TTS audio: ${YELLOW}${OUTPUT_DIR}/audio/${NC}"
if [ "$ENABLE_AUGMENTATION" = "true" ] && [ "$FINAL_OUTPUT" = "${AUGMENTED_DIR}" ]; then
    echo -e "  Augmented data: ${YELLOW}${AUGMENTED_DIR}${NC}"
fi
echo ""

# Count files
echo -e "${GREEN}Generated Files:${NC}"
TRAIN_COUNT=$(find ${OUTPUT_DIR}/audio/train/wavs -name "*.wav" 2>/dev/null | wc -l)
VAL_COUNT=$(find ${OUTPUT_DIR}/audio/val/wavs -name "*.wav" 2>/dev/null | wc -l)
TEST_COUNT=$(find ${OUTPUT_DIR}/audio/test/wavs -name "*.wav" 2>/dev/null | wc -l)
TOTAL_ORIGINAL=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))

echo -e "  ${BLUE}Original TTS:${NC}"
echo -e "    Train: ${YELLOW}${TRAIN_COUNT}${NC} files"
echo -e "    Val:   ${YELLOW}${VAL_COUNT}${NC} files"
echo -e "    Test:  ${YELLOW}${TEST_COUNT}${NC} files"
echo -e "    Subtotal: ${YELLOW}${TOTAL_ORIGINAL}${NC} files"

if [ "$ENABLE_AUGMENTATION" = "true" ] && [ -d "${AUGMENTED_DIR}/train/wavs" ]; then
    AUG_TRAIN_COUNT=$(find ${AUGMENTED_DIR}/train/wavs -name "*.wav" 2>/dev/null | wc -l)
    AUG_VAL_COUNT=$(find ${AUGMENTED_DIR}/val/wavs -name "*.wav" 2>/dev/null | wc -l)
    AUG_TEST_COUNT=$(find ${AUGMENTED_DIR}/test/wavs -name "*.wav" 2>/dev/null | wc -l)
    TOTAL_AUGMENTED=$((AUG_TRAIN_COUNT + AUG_VAL_COUNT + AUG_TEST_COUNT))
    
    echo -e "  ${BLUE}Augmented Data:${NC}"
    echo -e "    Train: ${YELLOW}${AUG_TRAIN_COUNT}${NC} files"
    echo -e "    Val:   ${YELLOW}${AUG_VAL_COUNT}${NC} files"
    echo -e "    Test:  ${YELLOW}${AUG_TEST_COUNT}${NC} files"
    echo -e "    Subtotal: ${YELLOW}${TOTAL_AUGMENTED}${NC} files"
    echo ""
    echo -e "  ${GREEN}TOTAL: ${TOTAL_AUGMENTED} audio files (for training)${NC}"
else
    echo ""
    echo -e "  ${GREEN}TOTAL: ${TOTAL_ORIGINAL} audio files${NC}"
fi
echo ""

# Storage usage
echo -e "${GREEN}Storage Usage:${NC}"
ORIGINAL_SIZE=$(du -sh ${OUTPUT_DIR}/audio 2>/dev/null | cut -f1)
echo -e "  Original: ${YELLOW}${ORIGINAL_SIZE}${NC}"
if [ "$ENABLE_AUGMENTATION" = "true" ] && [ -d "${AUGMENTED_DIR}" ]; then
    AUGMENTED_SIZE=$(du -sh ${AUGMENTED_DIR} 2>/dev/null | cut -f1)
    echo -e "  Augmented: ${YELLOW}${AUGMENTED_SIZE}${NC}"
fi
echo ""

# Next steps
echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo -e "${BLUE}1. Verify the data:${NC}"
echo -e "   ${YELLOW}ls -lh ${FINAL_OUTPUT}/train/wavs/ | head -10${NC}"
echo -e "   ${YELLOW}head -10 ${FINAL_OUTPUT}/train/metadata.csv${NC}"
echo -e "   ${YELLOW}cat ${OUTPUT_DIR}/dataset/coverage_stats.json | jq${NC}"
echo ""
echo -e "${BLUE}2. Train Whisper model (RECOMMENDED):${NC}"
echo -e "   ${YELLOW}python3 whisper_pharma_train.py \\${NC}"
echo -e "   ${YELLOW}  --audio_dir ${FINAL_OUTPUT} \\${NC}"
echo -e "   ${YELLOW}  --output_dir whisper_pharma_model \\${NC}"
echo -e "   ${YELLOW}  --model_size base \\${NC}"
echo -e "   ${YELLOW}  --num_epochs 10${NC}"
echo ""
echo -e "${BLUE}3. OR use ESPnet (complex, not recommended):${NC}"
echo -e "   ${YELLOW}python3 espnet_setup.py --data-dir ${FINAL_OUTPUT}${NC}"
echo -e "   ${YELLOW}cd ~/espnet/egs2/custom_asr/asr1${NC}"
echo -e "   ${YELLOW}bash run.sh --stage 1 --stop_stage 100 --data_dir ${FINAL_OUTPUT}${NC}"
echo ""
echo -e "${BLUE}4. To disable augmentation:${NC}"
echo -e "   Edit this script and set: ${YELLOW}ENABLE_AUGMENTATION=false${NC}"
echo ""

# Show parameter summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Parameter Summary${NC}"
echo -e "${BLUE}========================================${NC}"
cat << EOF

TTS Generation:
  - test_xtts_indian.py
    --dataset-size ${DATASET_SIZE}
    --speaker-wav ${SPEAKER_WAV}
    --output-dir ${OUTPUT_DIR}
    --num-workers ${NUM_WORKERS}
    --language ${LANGUAGE}

Data Augmentation (if enabled):
  - data_augmentation.py
    --dataset ${OUTPUT_DIR}/audio
    --output-dir ${AUGMENTED_DIR}
    --num-augmentations ${NUM_AUGMENTATIONS}
    --num-workers ${AUGMENTATION_WORKERS}
    --sample-rate ${SAMPLE_RATE}

EOF

echo -e "${GREEN}✅ Pipeline execution complete!${NC}"
echo ""

