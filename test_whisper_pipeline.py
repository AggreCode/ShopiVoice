#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Whisper Pharmaceutical Voice Pipeline
Tests the pipeline with pre-recorded audio files
"""

import sys
import json
import glob
from whisper_pharma_voice_pipeline import (
    WhisperTranscriber,
    process_audio_order,
    logger
)

def test_with_audio_file(audio_path, transcriber):
    """Test the pipeline with a specific audio file"""
    print("\n" + "=" * 70)
    print(f"Testing with: {audio_path}")
    print("=" * 70)
    
    try:
        result = process_audio_order(audio_path, transcriber)
        
        if result:
            print("\n" + "-" * 70)
            print("RESULTS:")
            print("-" * 70)
            print(f"Original Transcription: {result['original_transcription']}")
            print(f"Processed Text: {result['processed_text']}")
            print(f"Parsed Order ({len(result['parsed_order'])} items):")
            print(json.dumps(result['parsed_order'], indent=2, ensure_ascii=False))
            
            total_price = sum(
                item.get('total_price', 0)
                for item in result['parsed_order']
                if item.get('total_price')
            )
            if total_price > 0:
                print(f"\nTotal Order Value: ₹{total_price:.2f}")
            
            return result
        else:
            print("❌ Failed to process audio file")
            return None
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_batch(audio_dir, num_samples=5):
    """Test with multiple audio files from a directory"""
    print("=" * 70)
    print("BATCH TESTING MODE")
    print("=" * 70)
    
    # Find audio files
    audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Testing with {min(num_samples, len(audio_files))} samples\n")
    
    # Initialize transcriber once
    print("Loading Whisper model...")
    transcriber = WhisperTranscriber()
    if not transcriber.load_model():
        print("❌ Failed to load model")
        return
    
    # Test each file
    results = []
    for i, audio_path in enumerate(audio_files[:num_samples]):
        result = test_with_audio_file(audio_path, transcriber)
        if result:
            results.append({
                "file": audio_path,
                "transcription": result['original_transcription'],
                "parsed_items": len(result['parsed_order'])
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH TEST SUMMARY")
    print("=" * 70)
    print(f"Total files tested: {len(results)}")
    print(f"Successful: {len([r for r in results if r['parsed_items'] > 0])}")
    print(f"Failed to parse: {len([r for r in results if r['parsed_items'] == 0])}")

def test_single(audio_path):
    """Test with a single audio file"""
    print("=" * 70)
    print("SINGLE FILE TEST MODE")
    print("=" * 70)
    
    # Initialize transcriber
    print("Loading Whisper model...")
    transcriber = WhisperTranscriber()
    if not transcriber.load_model():
        print("❌ Failed to load model")
        return
    
    # Test the file
    test_with_audio_file(audio_path, transcriber)

def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file test: python test_whisper_pipeline.py <audio_file.wav>")
        print("  Batch test: python test_whisper_pipeline.py <audio_directory> [num_samples]")
        print("\nExample:")
        print("  python test_whisper_pipeline.py training_output/audio/test/wavs/utt_000001.wav")
        print("  python test_whisper_pipeline.py training_output/audio/test 10")
        return
    
    path = sys.argv[1]
    
    if path.endswith('.wav'):
        # Single file test
        test_single(path)
    else:
        # Batch test
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_batch(path, num_samples)

if __name__ == "__main__":
    main()



