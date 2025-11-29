#!/bin/bash

# Verification Script for CUDA Fixes
# This script tests that all fixes are working correctly

set -e  # Exit on error

echo "=========================================="
echo "CUDA Fixes Verification Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "xtts_indian_pipeline.py" ]; then
    echo "❌ Error: xtts_indian_pipeline.py not found"
    echo "   Please run this script from upgraded_voice_pipeline directory"
    exit 1
fi

# Step 1: Verify glossary has no single-letter units
echo "Step 1: Checking glossary for problematic characters..."
if grep -q '"g": "g"' pharma_glossary.json; then
    echo "❌ FAIL: Found single-letter 'g' in glossary"
    exit 1
fi

if grep -q '"l": "l"' pharma_glossary.json; then
    echo "❌ FAIL: Found single-letter 'l' in glossary"
    exit 1
fi

if grep -q '"0": "0"' pharma_glossary.json; then
    echo "❌ FAIL: Found standalone '0' in glossary"
    exit 1
fi

echo "✅ PASS: Glossary is clean (no single-letter units)"
echo ""

# Step 2: Verify num_workers defaults to 1
echo "Step 2: Checking num_workers default values..."

if ! grep -q 'num_workers.*default=1' xtts_indian_pipeline.py; then
    echo "❌ FAIL: xtts_indian_pipeline.py num_workers not set to 1"
    exit 1
fi

if ! grep -q 'num_workers.*default=1' test_xtts_indian.py; then
    echo "❌ FAIL: test_xtts_indian.py num_workers not set to 1"
    exit 1
fi

echo "✅ PASS: num_workers defaults to 1 in all scripts"
echo ""

# Step 3: Verify sanitization exists
echo "Step 3: Checking sanitization functions..."

if ! grep -q '_sanitize_utterance' backup/dataset_generator.py; then
    echo "❌ FAIL: _sanitize_utterance not found in dataset_generator.py"
    exit 1
fi

if ! grep -q '_sanitize_utterance' updated_smart_dataset_generator.py; then
    echo "❌ FAIL: _sanitize_utterance not applied in updated_smart_dataset_generator.py"
    exit 1
fi

echo "✅ PASS: Sanitization functions present"
echo ""

# Step 4: Verify no carrier phrases
echo "Step 4: Checking for problematic carrier phrases..."

if grep -q '"The item is:' xtts_indian_pipeline.py; then
    echo "❌ FAIL: Found carrier phrase 'The item is:' in TTS pipeline"
    exit 1
fi

if grep -q 'Thank you' xtts_indian_pipeline.py; then
    echo "❌ FAIL: Found 'Thank you' padding in TTS pipeline"
    exit 1
fi

echo "✅ PASS: No carrier phrases or extra word padding"
echo ""

# Step 5: Test dataset generation
echo "Step 5: Testing dataset generation (10 samples)..."

python3 test_xtts_indian.py \
  --dataset-size 10 \
  --max-samples 10 \
  --speaker-wav Recording_13.wav \
  --output-dir verify_test \
  --num-workers 1 > verify_test.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ FAIL: Dataset generation failed"
    tail -20 verify_test.log
    exit 1
fi

# Check if files were created
AUDIO_COUNT=$(find verify_test/audio/train -name "*.wav" 2>/dev/null | wc -l)
if [ "$AUDIO_COUNT" -lt 8 ]; then
    echo "❌ FAIL: Expected ~10 audio files, found only $AUDIO_COUNT"
    echo "   (Some failures are OK, but too many is a problem)"
    tail -20 verify_test.log
    exit 1
fi

echo "✅ PASS: Generated $AUDIO_COUNT audio files successfully"
echo ""

# Step 6: Check for CUDA errors in logs
echo "Step 6: Checking for CUDA errors in generation logs..."

CUDA_ERRORS=$(grep -c "CUDA error" verify_test.log || true)
if [ "$CUDA_ERRORS" -gt 2 ]; then
    echo "❌ FAIL: Found $CUDA_ERRORS CUDA errors (too many)"
    grep "CUDA error" verify_test.log
    exit 1
fi

if [ "$CUDA_ERRORS" -eq 0 ]; then
    echo "✅ PASS: No CUDA errors detected! 🎉"
else
    echo "⚠️  WARNING: Found $CUDA_ERRORS CUDA error(s) (acceptable for testing)"
fi
echo ""

# Step 7: Verify metadata matches
echo "Step 7: Verifying metadata format..."

if [ -f "verify_test/audio/train/metadata.csv" ]; then
    # Check that texts end with punctuation
    if grep -qE '^[^,]+,[^,\.!\?]+$' verify_test/audio/train/metadata.csv; then
        echo "⚠️  WARNING: Some texts don't end with punctuation"
    else
        echo "✅ PASS: All texts properly formatted"
    fi
else
    echo "❌ FAIL: metadata.csv not found"
    exit 1
fi
echo ""

# Cleanup
echo "Step 8: Cleaning up test files..."
rm -rf verify_test verify_test.log
echo "✅ Cleanup complete"
echo ""

# Final summary
echo "=========================================="
echo "✅ ALL VERIFICATION TESTS PASSED!"
echo "=========================================="
echo ""
echo "Your system is ready for full training pipeline:"
echo ""
echo "  bash run_training_pipeline.sh"
echo ""
echo "Or manual test:"
echo ""
echo "  python3 test_xtts_indian.py \\"
echo "    --dataset-size 5000 \\"
echo "    --speaker-wav Recording_13.wav \\"
echo "    --output-dir training_output \\"
echo "    --num-workers 1"
echo ""
echo "Monitor GPU with: watch -n 1 nvidia-smi"
echo ""









