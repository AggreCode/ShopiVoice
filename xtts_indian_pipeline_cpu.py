#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XTTS v2 Indian Voice Pipeline - CPU Version

This module provides a CPU-only TTS pipeline using Coqui's XTTS v2 model
with Indian English accent voice cloning. This version forces CPU usage
even if GPU is available, making it suitable for local development and testing.

NOTE: CPU processing is MUCH slower than GPU (10-20x slower).
      Recommended only for small batches or testing.
"""

import os
import json
import logging
import concurrent.futures
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("XTTSIndianPipeline-CPU")

# Try to import TTS
try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
    logger.info("Coqui TTS (XTTS) is available")
except ImportError:
    XTTS_AVAILABLE = False
    logger.error("Coqui TTS not installed. Please install with: pip install TTS")


class XTTSIndianPipelineCPU:
    """
    CPU-only pipeline for generating Indian English speech using XTTS v2.
    
    This version forces CPU usage for local development without GPU.
    Processing will be significantly slower than the GPU version.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the XTTS pipeline (CPU-only).
        
        Args:
            config: Configuration dictionary with the following keys:
                - output_dir: Directory to save generated audio
                - speaker_wav: Path to reference speaker WAV file
                - language: Language code (default: 'en')
                - sample_rate: Sample rate for generated audio (default: 16000)
                - num_workers: Number of workers for parallel processing (default: 1)
                  NOTE: Even on CPU, XTTS v2 is NOT thread-safe. Use num_workers=1.
                - max_samples: Maximum number of samples to process per split (default: None = all)
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'xtts_output')
        self.speaker_wav = self.config.get('speaker_wav', 'Recording_13.wav')
        self.language = self.config.get('language', 'en')
        self.sample_rate = self.config.get('sample_rate', 16000)
        # IMPORTANT: XTTS v2 is NOT thread-safe - must use num_workers=1
        self.num_workers = self.config.get('num_workers', 1)
        self.max_samples = self.config.get('max_samples', None)  # None means process all
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Verify speaker WAV exists
        if not os.path.exists(self.speaker_wav):
            raise FileNotFoundError(f"Speaker WAV file not found: {self.speaker_wav}")
        
        logger.info(f"Using speaker WAV: {self.speaker_wav}")
        logger.warning("=" * 60)
        logger.warning("CPU-ONLY MODE: Processing will be 10-20x slower than GPU")
        logger.warning("Recommended only for testing with small datasets (<100 samples)")
        logger.warning("=" * 60)
        
        # Initialize XTTS v2 on CPU
        self._init_xtts_cpu()
        
        logger.info(f"XTTSIndianPipelineCPU initialized")
    
    def _init_xtts_cpu(self) -> None:
        """Initialize the XTTS v2 model on CPU only."""
        if not XTTS_AVAILABLE:
            raise ImportError("Coqui TTS not installed. Please install with: pip install TTS")
        
        try:
            # Force CPU usage
            import torch
            logger.info("=" * 60)
            logger.info("FORCING CPU MODE (ignoring GPU even if available)")
            
            # Check if GPU is available but being ignored
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.warning(f"GPU detected ({gpu_name}) but will NOT be used")
                logger.warning("To use GPU, use xtts_indian_pipeline.py instead")
            else:
                logger.info("No GPU detected - using CPU")
            
            logger.info("=" * 60)
            
            # Load XTTS v2 model with CPU only (gpu=False)
            logger.info("Loading XTTS v2 model on CPU...")
            logger.info("This may take 30-60 seconds...")
            
            self.tts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=False  # FORCE CPU
            )
            
            logger.info("✓ XTTS v2 model loaded on CPU")
            logger.info("=" * 60)
            logger.info("Performance expectations:")
            logger.info("  - ~2-5 seconds per utterance (CPU)")
            logger.info("  - ~0.3-0.5 seconds per utterance (GPU)")
            logger.info("  - For 100 samples: ~5-10 minutes (CPU) vs ~1-2 minutes (GPU)")
            logger.info("=" * 60)
            logger.info("XTTS v2 model initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing XTTS v2: {e}")
            raise
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text to prevent errors.
        
        NOTE: Main sanitization happens in dataset_generator.py.
        This is a safety layer that should NOT add/change words
        to avoid train/audio mismatch.
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        import re
        
        # Strip and normalize whitespace
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure text ends with punctuation for better prosody
        # This is safe - doesn't add words, only punctuation
        if text and text[-1] not in '.!?,':
            text = text + '.'
        
        # Warn about very short texts (should be rare now)
        if len(text.split()) < 2:
            logger.warning(f"Very short text detected: '{text}' - may cause TTS issues")
        
        return text
    
    def synthesize_text(self, text: str, output_file: str) -> bool:
        """
        Synthesize speech from text using the reference speaker voice.
        
        Args:
            text: Input text
            output_file: Output audio filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Sanitize text to prevent errors
            processed_text = self._sanitize_text(text)
            
            if processed_text != text.strip():
                logger.debug(f"Sanitized text: '{text}' -> '{processed_text}'")
            
            # Generate speech with XTTS v2 on CPU
            self.tts_model.tts_to_file(
                text=processed_text,
                file_path=output_file,
                speaker_wav=self.speaker_wav,
                language=self.language
            )
            
            logger.debug(f"Generated speech: {output_file}")
            return True
            
        except RuntimeError as e:
            # Handle runtime errors
            logger.error(f"Runtime error for text '{text}': {e}")
            logger.warning(f"Skipping problematic text")
            return False
        except Exception as e:
            logger.error(f"Error synthesizing speech for '{text}': {e}")
            return False
    
    def synthesize_batch(self, texts: List[Dict[str, Any]], output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Synthesize speech for a batch of texts.
        
        Args:
            texts: List of dictionaries containing text data
            output_dir: Output directory (default: self.output_dir)
            
        Returns:
            List of dictionaries with added audio_file field
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        # Process in parallel (though CPU is slow, some parallelism may help)
        # NOTE: Even with num_workers=1, we use ThreadPoolExecutor for consistency
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for i, item in enumerate(texts):
                text = item["utterance"]
                item_id = item.get("id", f"utt_{i:06d}")
                
                # Create output filename
                output_file = os.path.join(output_dir, f"{item_id}.wav")
                
                # Submit task
                future = executor.submit(self.synthesize_text, text, output_file)
                futures.append((future, item, output_file))
            
            # Process results with progress tracking
            total = len(futures)
            for idx, (future, item, output_file) in enumerate(futures, 1):
                try:
                    success = future.result()
                    if success:
                        # Add audio file to item
                        item_copy = item.copy()
                        item_copy["audio_file"] = output_file
                        results.append(item_copy)
                        logger.info(f"Progress: {idx}/{total} ({idx*100//total}%)")
                    else:
                        logger.warning(f"Failed to synthesize speech for: {item['utterance']}")
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
        
        logger.info(f"Synthesized speech for {len(results)}/{len(texts)} texts")
        return results
    
    def process_dataset(self, dataset_path: str, output_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a dataset generated by DatasetGenerator.
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory (default: self.output_dir)
            
        Returns:
            Dictionary containing processed datasets
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        results = {}
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(dataset_path, split)
            if not os.path.exists(split_dir):
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            # Create output directory
            split_output_dir = os.path.join(output_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            
            # Create wavs directory
            wavs_dir = os.path.join(split_output_dir, "wavs")
            os.makedirs(wavs_dir, exist_ok=True)
            
            # Load structured data
            structured_path = os.path.join(split_dir, "structured.jsonl")
            if not os.path.exists(structured_path):
                logger.warning(f"Structured data file not found: {structured_path}")
                continue
            
            # Load data
            items = []
            with open(structured_path, 'r') as f:
                for line in f:
                    items.append(json.loads(line))
            
            # Limit to max_samples if specified
            if self.max_samples is not None and self.max_samples > 0:
                original_count = len(items)
                items = items[:self.max_samples]
                logger.info(f"Processing first {len(items)} of {original_count} samples in {split}")
            
            logger.info(f"=" * 60)
            logger.info(f"Processing {split} split: {len(items)} samples")
            logger.info(f"Estimated time: {len(items) * 3 // 60}-{len(items) * 5 // 60} minutes")
            logger.info(f"=" * 60)
            
            # Synthesize speech
            processed_items = self.synthesize_batch(items, wavs_dir)
            results[split] = processed_items
            
            # Update metadata
            metadata_path = os.path.join(split_output_dir, "metadata.csv")
            with open(metadata_path, 'w', newline='') as f:
                import csv
                writer = csv.writer(f, delimiter='|')
                
                for item in processed_items:
                    writer.writerow([
                        item["id"],
                        item["utterance"]
                    ])
            
            logger.info(f"✓ Processed {split}: {len(processed_items)} files")
        
        logger.info(f"Dataset processing complete: {dataset_path}")
        return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XTTS v2 Indian Voice Pipeline - CPU Version")
    parser.add_argument("--output-dir", type=str, default="xtts_output_cpu",
                        help="Output directory for generated audio")
    parser.add_argument("--speaker-wav", type=str, 
                        default="Recording_13.wav",
                        help="Path to reference speaker WAV file")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers (WARNING: XTTS v2 is NOT thread-safe, use 1)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process per split (default: all)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize")
    
    args = parser.parse_args()
    
    # Warn about CPU usage
    if args.max_samples is None or args.max_samples > 100:
        logger.warning("=" * 60)
        logger.warning("WARNING: Processing large datasets on CPU is VERY slow!")
        logger.warning("Consider using --max-samples 10 for testing")
        logger.warning("For production, use xtts_indian_pipeline.py with GPU")
        logger.warning("=" * 60)
        import time
        time.sleep(3)
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "speaker_wav": args.speaker_wav,
        "language": args.language,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples
    }
    
    # Create XTTS pipeline (CPU)
    tts_pipeline = XTTSIndianPipelineCPU(config)
    
    # Process dataset
    if args.dataset:
        tts_pipeline.process_dataset(args.dataset)
    
    # Synthesize text
    if args.text:
        output_file = os.path.join(args.output_dir, "output.wav")
        success = tts_pipeline.synthesize_text(args.text, output_file)
        if success:
            print(f"Speech synthesized and saved to {output_file}")
        else:
            print("Failed to synthesize speech")





