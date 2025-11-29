#!/bin/bash

# Quick test script for Whisper Pharmaceutical Pipeline
# Tests the pipeline with sample audio files

echo "=============================================="
echo "Whisper Pipeline Quick Test"
echo "=============================================="

# Check if model exists
if [ ! -d "whisper_pharma_model" ]; then
    echo "❌ Error: whisper_pharma_model directory not found"
    echo "   Please ensure the trained model is in the current directory"
    exit 1
fi

echo "✓ Model found"

# Check if training audio exists
if [ ! -d "training_output/audio/test/wavs" ]; then
    echo "❌ Error: training_output/audio/test/wavs not found"
    echo "   Please ensure training audio files exist for testing"
    exit 1
fi

echo "✓ Test audio found"

# Count audio files
NUM_FILES=$(ls training_output/audio/test/wavs/*.wav 2>/dev/null | wc -l)
echo "✓ Found $NUM_FILES test audio files"

if [ $NUM_FILES -eq 0 ]; then
    echo "❌ No WAV files found in training_output/audio/test/wavs"
    exit 1
fi

echo ""
echo "=============================================="
echo "Running Pipeline Tests"
echo "=============================================="

# Test with 5 random samples
echo ""
echo "Testing with 5 sample files..."
python3 test_whisper_pipeline.py training_output/audio/test 5

echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
echo ""
echo "To test with ESP32 live recording:"
echo "  1. Configure SERIAL_PORT in whisper_pharma_voice_pipeline.py"
echo "  2. Connect ESP32 device"
echo "  3. Run: python3 whisper_pharma_voice_pipeline.py"
echo ""



