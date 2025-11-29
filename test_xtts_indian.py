#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test XTTS Indian Voice Pipeline

Quick test script for the XTTS v2 Indian voice pipeline.
"""

import os
import sys
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestXTTSIndian")

def generate_dataset(output_dir: str, size: int = 10):
    """Generate a small test dataset."""
    from updated_smart_dataset_generator import UpdatedSmartDatasetGenerator
    
    logger.info(f"Generating test dataset with {size} samples...")
    
    # Adjust min_occurrences based on dataset size
    if size <= 100:
        min_occ = 2
    elif size <= 500:
        min_occ = 5
    elif size <= 2000:
        min_occ = 10
    else:
        min_occ = 20
    
    logger.info(f"Using min_occurrences={min_occ} for dataset_size={size}")
    
    generator = UpdatedSmartDatasetGenerator({
        "output_dir": output_dir,
        "language": "en",
        "dataset_size": size,
        "min_occurrences": min_occ,
        "seed": 42
    })
    
    generator.set_language_templates("en")
    
    # Load glossaries
    glossary_file = "pharma_glossary.json"
    if os.path.exists(glossary_file):
        generator.load_glossaries_from_file(glossary_file)
    
    # Generate dataset
    dataset_splits = generator.generate_and_save_smart_dataset(size)
    
    logger.info(f"Dataset generated: {output_dir}")
    return output_dir

def generate_audio(dataset_dir: str, output_dir: str, speaker_wav: str, max_samples: int = None, num_workers: int = 2):
    """Generate audio using XTTS Indian pipeline."""
    from xtts_indian_pipeline import XTTSIndianPipeline
    
    logger.info(f"Generating audio with XTTS Indian voice (max_samples={max_samples}, num_workers={num_workers})...")
    
    tts_pipeline = XTTSIndianPipeline({
        "output_dir": output_dir,
        "speaker_wav": speaker_wav,
        "language": "en",
        "num_workers": num_workers,
        "max_samples": max_samples
    })
    
    # Process dataset
    tts_pipeline.process_dataset(dataset_dir)
    
    logger.info(f"Audio generated: {output_dir}")
    return output_dir

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test XTTS Indian Voice Pipeline")
    parser.add_argument("--dataset-size", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--speaker-wav", type=str,
                        default="/home/biswajit-mohanty/Recording_13.wav",
                        help="Path to speaker WAV file")
    parser.add_argument("--output-dir", type=str, default="xtts_indian_test",
                        help="Base output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process per split (default: all)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (WARNING: XTTS v2 is NOT thread-safe, use 1)")
    
    args = parser.parse_args()
    
    # Create output directories
    base_dir = args.output_dir
    dataset_dir = os.path.join(base_dir, "dataset")
    audio_dir = os.path.join(base_dir, "audio")
    
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        # Step 1: Generate dataset
        logger.info("="*60)
        logger.info("STEP 1: Generate Dataset")
        logger.info("="*60)
        dataset_dir = generate_dataset(dataset_dir, args.dataset_size)
        
        # Step 2: Generate audio
        logger.info("")
        logger.info("="*60)
        logger.info("STEP 2: Generate Audio with XTTS Indian Voice")
        logger.info("="*60)
        audio_dir = generate_audio(dataset_dir, audio_dir, args.speaker_wav, args.max_samples, args.num_workers)
        
        # Summary
        logger.info("")
        logger.info("="*60)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Dataset: {dataset_dir}")
        logger.info(f"Audio files: {audio_dir}")
        logger.info("")
        logger.info("Listen to generated audio:")
        logger.info(f"  ls -lh {audio_dir}/train/wavs/")
        logger.info(f"  ffplay {audio_dir}/train/wavs/utt_000000.wav")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

