#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test XTTS Indian Voice Pipeline - CPU Version

This script tests the XTTS v2 pipeline with Indian voice cloning on CPU.
Suitable for local development and testing without GPU.

WARNING: CPU processing is 10-20x slower than GPU.
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
logger = logging.getLogger("TestXTTSIndianCPU")


def generate_dataset(output_dir: str, size: int = 10):
    """
    Generate a test dataset using UpdatedSmartDatasetGenerator.
    
    Args:
        output_dir: Output directory for the dataset
        size: Dataset size
    """
    from updated_smart_dataset_generator import UpdatedSmartDatasetGenerator
    
    logger.info(f"Generating test dataset with {size} samples...")
    
    # Adjust min_occurrences based on dataset size
    # For small datasets, use lower min_occurrences to ensure data generation
    if size <= 10:
        min_occ = 1  # Very small dataset
    elif size <= 50:
        min_occ = 2  # Small dataset
    elif size <= 200:
        min_occ = 5  # Medium dataset
    else:
        min_occ = 10  # Large dataset
    
    logger.info(f"Using min_occurrences={min_occ} for dataset_size={size}")
    
    generator = UpdatedSmartDatasetGenerator({
        "output_dir": output_dir,
        "language": "en",
        "dataset_size": size,
        "min_occurrences": min_occ,
        "seed": 42
    })
    
    logger.info("Generating dataset...")
    # Use the smart dataset generation method which saves files
    dataset_splits = generator.generate_and_save_smart_dataset(target_size=size)
    
    # Verify files were created
    total_samples = 0
    for split in ['train', 'val', 'test']:
        structured_file = os.path.join(output_dir, split, "structured.jsonl")
        if os.path.exists(structured_file):
            with open(structured_file) as f:
                count = sum(1 for _ in f)
            logger.info(f"✓ {split}: {count} samples in structured.jsonl")
            total_samples += count
        else:
            logger.warning(f"⚠ {split}: structured.jsonl not found (split may be empty)")
    
    if total_samples == 0:
        logger.error("No samples were generated! Check dataset_size and min_occurrences settings")
        raise ValueError("Dataset generation produced no samples")
    
    logger.info(f"✓ Dataset generated in {output_dir}")
    return os.path.join(output_dir)


def run_cpu_test(
    dataset_size: int = 10,
    speaker_wav: str = "Recording_13.wav",
    output_dir: str = "xtts_test_cpu",
    num_workers: int = 1,
    max_samples: int = None
):
    """
    Run the XTTS Indian voice pipeline test on CPU.
    
    Args:
        dataset_size: Number of samples to generate
        speaker_wav: Path to speaker WAV file
        output_dir: Output directory
        num_workers: Number of workers (should be 1)
        max_samples: Maximum samples to process per split
    """
    # Import the CPU pipeline
    from xtts_indian_pipeline_cpu import XTTSIndianPipelineCPU
    
    logger.info("=" * 60)
    logger.info("XTTS Indian Voice Pipeline - CPU Test")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {dataset_size}")
    logger.info(f"Speaker WAV: {speaker_wav}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
    logger.info("=" * 60)
    
    # Check if speaker WAV exists
    if not os.path.exists(speaker_wav):
        logger.error(f"Speaker WAV not found: {speaker_wav}")
        logger.info("Please provide a valid speaker WAV file")
        sys.exit(1)
    
    # Generate dataset
    dataset_dir = os.path.join(output_dir, "dataset")
    audio_dir = os.path.join(output_dir, "audio")
    
    logger.info("Step 1: Generating dataset...")
    dataset_path = generate_dataset(dataset_dir, dataset_size)
    
    # Create XTTS pipeline configuration
    config = {
        "output_dir": audio_dir,
        "speaker_wav": speaker_wav,
        "language": "en",
        "num_workers": num_workers,
        "max_samples": max_samples
    }
    
    logger.info("Step 2: Synthesizing speech with XTTS v2 (CPU)...")
    logger.info("=" * 60)
    
    # Calculate expected time
    total_samples = dataset_size
    estimated_time_min = (total_samples * 2) // 60
    estimated_time_max = (total_samples * 5) // 60
    
    logger.info(f"Estimated processing time: {estimated_time_min}-{estimated_time_max} minutes")
    logger.info("=" * 60)
    
    # Create and run pipeline
    tts_pipeline = XTTSIndianPipelineCPU(config)
    results = tts_pipeline.process_dataset(dataset_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test completed successfully!")
    logger.info("=" * 60)
    
    for split, items in results.items():
        logger.info(f"{split}: {len(items)} audio files generated")
    
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Audio files: {audio_dir}")
    logger.info(f"Dataset files: {dataset_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test XTTS Indian Voice Pipeline on CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 samples (recommended for CPU)
  python3 test_xtts_indian_cpu.py --dataset-size 5

  # Test with 10 samples
  python3 test_xtts_indian_cpu.py --dataset-size 10 --speaker-wav Recording_13.wav

  # Test with custom output directory
  python3 test_xtts_indian_cpu.py --dataset-size 10 --output-dir my_test_cpu

  # Process only first 10 samples per split
  python3 test_xtts_indian_cpu.py --dataset-size 50 --max-samples 10

Note:
  - CPU processing is 10-20x slower than GPU
  - Recommended dataset size for testing: 5-10 samples
  - Each sample takes ~2-5 seconds on modern CPU
  - For production use, switch to GPU version (test_xtts_indian.py)
        """
    )
    
    parser.add_argument("--dataset-size", type=int, default=10,
                        help="Dataset size to generate (default: 10, recommended: 5-10 for CPU)")
    parser.add_argument("--speaker-wav", type=str, default="Recording_13.wav",
                        help="Path to speaker WAV file")
    parser.add_argument("--output-dir", type=str, default="xtts_test_cpu",
                        help="Output directory")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers (default: 1, must be 1 for XTTS)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process per split (default: all)")
    
    args = parser.parse_args()
    
    # Validate num_workers
    if args.num_workers != 1:
        logger.warning("=" * 60)
        logger.warning("WARNING: XTTS v2 is NOT thread-safe!")
        logger.warning("Setting num_workers to 1 (forced)")
        logger.warning("=" * 60)
        args.num_workers = 1
    
    # Warn about large datasets on CPU
    if args.dataset_size > 20:
        logger.warning("=" * 60)
        logger.warning(f"WARNING: Generating {args.dataset_size} samples on CPU will be VERY slow!")
        logger.warning(f"Estimated time: {args.dataset_size * 2 // 60}-{args.dataset_size * 5 // 60} minutes")
        logger.warning("Consider using --dataset-size 5 for quick testing")
        logger.warning("For larger datasets, use GPU version (test_xtts_indian.py)")
        logger.warning("=" * 60)
        
        # Ask for confirmation
        try:
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                sys.exit(0)
        except KeyboardInterrupt:
            logger.info("\nAborted by user")
            sys.exit(0)
    
    # Run the test
    try:
        run_cpu_test(
            dataset_size=args.dataset_size,
            speaker_wav=args.speaker_wav,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

