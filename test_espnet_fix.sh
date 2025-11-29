#!/bin/bash

# Quick test script to verify espnet_setup.py fixes
# Run this on your GPU server

echo "=========================================="
echo "Testing ESPnet Setup Fixes"
echo "=========================================="
echo ""

# Test 1: Check for SyntaxWarnings
echo "Test 1: Checking for SyntaxWarnings..."
WARNINGS=$(python3 -W default espnet_setup.py --data-dir training_output/audio 2>&1 | grep -i "SyntaxWarning" | wc -l)

if [ "$WARNINGS" -eq 0 ]; then
    echo "✅ PASS: No SyntaxWarnings found"
else
    echo "❌ FAIL: Found $WARNINGS SyntaxWarning(s)"
fi
echo ""

# Test 2: Check if shutil.which finds commands
echo "Test 2: Checking command detection with shutil.which()..."
python3 << 'PYEOF'
import shutil

commands = ['git', 'make', 'cmake', 'sox', 'ffmpeg', 'python3']
all_found = True

for cmd in commands:
    path = shutil.which(cmd)
    if path:
        print(f"  ✓ {cmd:12} → {path}")
    else:
        print(f"  ✗ {cmd:12} → NOT FOUND")
        all_found = False

if all_found:
    print("\n✅ PASS: All commands found")
else:
    print("\n⚠️  WARNING: Some commands missing (install with apt-get)")
PYEOF
echo ""

# Test 3: Run actual espnet_setup.py
echo "Test 3: Running espnet_setup.py dependency check..."
python3 espnet_setup.py --data-dir training_output/audio 2>&1 | head -20
echo ""

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "If all tests passed, you can now:"
echo "  1. Run: bash run_training_pipeline.sh 5000"
echo "  2. Or: Uncomment ESPnet section in run_training_pipeline.sh"
echo ""









