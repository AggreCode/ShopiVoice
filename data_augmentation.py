#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Acoustic Data Augmentation

This module provides tools for augmenting audio data to improve ASR model
robustness. It implements various augmentation techniques such as adding
noise, changing speed, pitch shifting, and room impulse response simulation.

Features:
- Multiple augmentation techniques
- Batch processing
- Configurable augmentation parameters
- Integration with TTS pipeline and dataset generator
"""

import os
import re
import json
import random
import logging
import tempfile
import subprocess
import concurrent.futures
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataAugmentation")

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not installed. Some features will be disabled.")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa or soundfile not installed. Some features will be disabled.")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Some features will be disabled.")


class DataAugmentation:
    """
    Class for augmenting audio data for ASR training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data augmentation with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - output_dir: Directory to save augmented audio
                - sample_rate: Sample rate for audio (default: 16000)
                - num_workers: Number of workers for parallel processing (default: 4)
                - augmentations: List of augmentation techniques to apply
                - augmentation_params: Dictionary of parameters for each augmentation
                - num_augmentations: Number of augmented versions to generate per file (default: 5)
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'augmented_data')
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.num_workers = self.config.get('num_workers', 4)
        self.augmentations = self.config.get('augmentations', ['noise', 'speed', 'pitch', 'rir'])
        self.augmentation_params = self.config.get('augmentation_params', {})
        self.num_augmentations = self.config.get('num_augmentations', 5)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize noise samples
        self.noise_samples = {}
        self._init_noise_samples()
        
        # Initialize room impulse responses
        self.rir_samples = {}
        self._init_rir_samples()
        
        logger.info(f"DataAugmentation initialized with augmentations: {self.augmentations}")
    
    def _init_noise_samples(self) -> None:
        """Initialize noise samples for augmentation."""
        # Check if noise directory exists
        noise_dir = self.config.get('noise_dir')
        if noise_dir and os.path.exists(noise_dir):
            # Load noise samples
            for file in os.listdir(noise_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    try:
                        if LIBROSA_AVAILABLE:
                            noise_path = os.path.join(noise_dir, file)
                            noise, _ = librosa.load(noise_path, sr=self.sample_rate, mono=True)
                            self.noise_samples[file] = noise
                    except Exception as e:
                        logger.error(f"Error loading noise sample {file}: {e}")
            
            logger.info(f"Loaded {len(self.noise_samples)} noise samples")
        else:
            # Create default noise samples
            if NUMPY_AVAILABLE:
                # White noise
                self.noise_samples['white_noise'] = np.random.normal(0, 0.01, self.sample_rate * 5)
                
                # Pink noise (approximate)
                if SCIPY_AVAILABLE:
                    pink_noise = np.random.normal(0, 1, self.sample_rate * 5)
                    b, a = signal.butter(1, 0.05, btype='lowpass')
                    self.noise_samples['pink_noise'] = signal.lfilter(b, a, pink_noise) * 0.01
                
                logger.info("Created default noise samples")
    
    def _init_rir_samples(self) -> None:
        """Initialize room impulse response samples for augmentation."""
        # Check if RIR directory exists
        rir_dir = self.config.get('rir_dir')
        if rir_dir and os.path.exists(rir_dir):
            # Load RIR samples
            for file in os.listdir(rir_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    try:
                        if LIBROSA_AVAILABLE:
                            rir_path = os.path.join(rir_dir, file)
                            rir, _ = librosa.load(rir_path, sr=self.sample_rate, mono=True)
                            self.rir_samples[file] = rir
                    except Exception as e:
                        logger.error(f"Error loading RIR sample {file}: {e}")
            
            logger.info(f"Loaded {len(self.rir_samples)} RIR samples")
        else:
            # Create default RIR samples
            if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
                # Simple room impulse response
                rir = np.zeros(int(self.sample_rate * 0.5))
                rir[0] = 1.0
                
                # Add some reflections
                for i in range(5):
                    delay = int(self.sample_rate * random.uniform(0.01, 0.3))
                    if delay < len(rir):
                        rir[delay] = random.uniform(0.1, 0.5)
                
                # Apply exponential decay
                decay = np.exp(-np.arange(len(rir)) / (self.sample_rate * 0.1))
                rir = rir * decay
                
                # Normalize
                rir = rir / np.sum(np.abs(rir))
                
                self.rir_samples['simple_room'] = rir
                
                logger.info("Created default RIR samples")
    
    def add_noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add noise to audio.
        
        Args:
            audio: Input audio as numpy array
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Augmented audio
        """
        if not NUMPY_AVAILABLE:
            logger.warning("numpy not installed, cannot add noise")
            return audio
        
        try:
            # Select noise sample
            if self.noise_samples:
                noise_name = random.choice(list(self.noise_samples.keys()))
                noise = self.noise_samples[noise_name]
            else:
                # Generate white noise
                noise = np.random.normal(0, 0.01, len(audio))
            
            # Adjust noise length to match audio
            if len(noise) < len(audio):
                # Repeat noise if needed
                repeats = int(np.ceil(len(audio) / len(noise)))
                noise = np.tile(noise, repeats)
            
            # Trim or pad noise to match audio length
            noise = noise[:len(audio)]
            
            # Calculate signal and noise power
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            
            # Calculate noise scale factor
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
            
            # Add scaled noise to audio
            noisy_audio = audio + noise_scale * noise
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val
            
            return noisy_audio
            
        except Exception as e:
            logger.error(f"Error adding noise: {e}")
            return audio
    
    def change_speed(self, audio: np.ndarray, speed_factor: float = 1.0) -> np.ndarray:
        """
        Change audio speed.
        
        Args:
            audio: Input audio as numpy array
            speed_factor: Speed factor (1.0 = original speed)
            
        Returns:
            Augmented audio
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not installed, cannot change speed")
            return audio
        
        try:
            # Change speed using librosa
            return librosa.effects.time_stretch(audio, rate=speed_factor)
            
        except Exception as e:
            logger.error(f"Error changing speed: {e}")
            return audio
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """
        Shift audio pitch.
        
        Args:
            audio: Input audio as numpy array
            n_steps: Number of semitones to shift
            
        Returns:
            Augmented audio
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not installed, cannot shift pitch")
            return audio
        
        try:
            # Shift pitch using librosa
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
        except Exception as e:
            logger.error(f"Error shifting pitch: {e}")
            return audio
    
    def apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply room impulse response to audio.
        
        Args:
            audio: Input audio as numpy array
            
        Returns:
            Augmented audio
        """
        if not SCIPY_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("scipy or numpy not installed, cannot apply RIR")
            return audio
        
        try:
            # Select RIR sample
            if self.rir_samples:
                rir_name = random.choice(list(self.rir_samples.keys()))
                rir = self.rir_samples[rir_name]
            else:
                # Use delta function (no effect)
                rir = np.zeros(1000)
                rir[0] = 1.0
            
            # Apply RIR using convolution
            reverb_audio = signal.convolve(audio, rir, mode='full')
            
            # Trim to original length
            reverb_audio = reverb_audio[:len(audio)]
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(reverb_audio))
            if max_val > 1.0:
                reverb_audio = reverb_audio / max_val
            
            return reverb_audio
            
        except Exception as e:
            logger.error(f"Error applying RIR: {e}")
            return audio
    
    def time_mask(self, audio: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """
        Apply time masking to audio.
        
        Args:
            audio: Input audio as numpy array
            mask_ratio: Ratio of audio to mask
            
        Returns:
            Augmented audio
        """
        if not NUMPY_AVAILABLE:
            logger.warning("numpy not installed, cannot apply time masking")
            return audio
        
        try:
            # Create a copy of the audio
            masked_audio = np.copy(audio)
            
            # Calculate mask length
            mask_length = int(len(audio) * mask_ratio)
            
            # Select random position for mask
            mask_start = random.randint(0, len(audio) - mask_length)
            
            # Apply mask
            masked_audio[mask_start:mask_start + mask_length] = 0
            
            return masked_audio
            
        except Exception as e:
            logger.error(f"Error applying time masking: {e}")
            return audio
    
    def augment_audio(self, audio_file: str, output_file: str, augmentations: List[str] = None) -> bool:
        """
        Augment audio file.
        
        Args:
            audio_file: Input audio filename
            output_file: Output audio filename
            augmentations: List of augmentation techniques to apply
            
        Returns:
            True if successful, False otherwise
        """
        if not LIBROSA_AVAILABLE:
            logger.error("librosa not installed, cannot augment audio")
            return False
        
        try:
            # Load audio
            audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            # Select augmentations
            if augmentations is None:
                augmentations = random.sample(self.augmentations, 
                                             k=min(len(self.augmentations), random.randint(1, 3)))
            
            # Apply augmentations
            augmented_audio = audio
            
            for aug in augmentations:
                if aug == 'noise':
                    snr_db = self.augmentation_params.get('snr_db', random.uniform(10.0, 30.0))
                    augmented_audio = self.add_noise(augmented_audio, snr_db)
                
                elif aug == 'speed':
                    speed_factor = self.augmentation_params.get('speed_factor', random.uniform(0.9, 1.1))
                    augmented_audio = self.change_speed(augmented_audio, speed_factor)
                
                elif aug == 'pitch':
                    n_steps = self.augmentation_params.get('n_steps', random.uniform(-2.0, 2.0))
                    augmented_audio = self.pitch_shift(augmented_audio, n_steps)
                
                elif aug == 'rir':
                    augmented_audio = self.apply_rir(augmented_audio)
                
                elif aug == 'time_mask':
                    mask_ratio = self.augmentation_params.get('mask_ratio', random.uniform(0.05, 0.15))
                    augmented_audio = self.time_mask(augmented_audio, mask_ratio)
            
            # Save augmented audio
            sf.write(output_file, augmented_audio, self.sample_rate)
            
            logger.debug(f"Augmented audio saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error augmenting audio: {e}")
            return False
    
    def augment_batch(self, audio_files: List[str], output_dir: str = None) -> List[str]:
        """
        Augment a batch of audio files.
        
        Args:
            audio_files: List of input audio filenames
            output_dir: Output directory (default: self.output_dir)
            
        Returns:
            List of output audio filenames
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for audio_file in audio_files:
                for i in range(self.num_augmentations):
                    # Create output filename
                    basename = os.path.basename(audio_file)
                    name, ext = os.path.splitext(basename)
                    output_file = os.path.join(output_dir, f"{name}_aug{i}{ext}")
                    
                    # Select augmentations
                    augmentations = random.sample(self.augmentations, 
                                                k=min(len(self.augmentations), random.randint(1, 3)))
                    
                    # Submit task
                    future = executor.submit(self.augment_audio, audio_file, output_file, augmentations)
                    futures.append((future, output_file))
            
            # Process results
            for future, output_file in futures:
                try:
                    success = future.result()
                    if success:
                        results.append(output_file)
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
        
        logger.info(f"Augmented {len(audio_files)} files, created {len(results)} augmented files")
        return results
    
    def process_dataset(self, dataset_path: str, output_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a dataset generated by TTS pipeline.
        
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
            
            # Load metadata
            metadata_path = os.path.join(split_dir, "metadata.csv")
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}")
                continue
            
            # Load data
            items = []
            audio_files = []
            
            import csv
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    items.append(row)
                    audio_file = os.path.join(split_dir, "wavs", row["audio_file"])
                    audio_files.append(audio_file)
            
            # Only augment training data
            if split == 'train':
                # Augment audio
                augmented_files = self.augment_batch(audio_files, wavs_dir)
                
                # Create augmented items
                augmented_items = []
                
                for i, output_file in enumerate(augmented_files):
                    # Get original item
                    orig_idx = i // self.num_augmentations
                    orig_item = items[orig_idx]
                    
                    # Create augmented item
                    aug_item = orig_item.copy()
                    aug_item["id"] = f"{orig_item['id']}_aug{i % self.num_augmentations}"
                    aug_item["audio_file"] = os.path.basename(output_file)
                    
                    augmented_items.append(aug_item)
                
                # Combine original and augmented items
                all_items = items + augmented_items
                
                # Copy original files
                for item in items:
                    src_file = os.path.join(split_dir, "wavs", item["audio_file"])
                    dst_file = os.path.join(wavs_dir, item["audio_file"])
                    if not os.path.exists(dst_file):
                        try:
                            import shutil
                            shutil.copy2(src_file, dst_file)
                        except Exception as e:
                            logger.error(f"Error copying file: {e}")
                
                results[split] = all_items
            else:
                # For validation and test, just copy the files
                for item in items:
                    src_file = os.path.join(split_dir, "wavs", item["audio_file"])
                    dst_file = os.path.join(wavs_dir, item["audio_file"])
                    if not os.path.exists(dst_file):
                        try:
                            import shutil
                            shutil.copy2(src_file, dst_file)
                        except Exception as e:
                            logger.error(f"Error copying file: {e}")
                
                results[split] = items
            
            # Update metadata
            metadata_path = os.path.join(split_output_dir, "metadata.csv")
            with open(metadata_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[split][0].keys())
                writer.writeheader()
                writer.writerows(results[split])
            
            # Update Kaldi files
            wav_scp_path = os.path.join(split_output_dir, "wav.scp")
            with open(wav_scp_path, 'w') as f:
                for item in results[split]:
                    audio_file = os.path.join(wavs_dir, item["audio_file"])
                    f.write(f"{item['id']} {audio_file}\n")
            
            # Update text file
            text_path = os.path.join(split_output_dir, "text")
            with open(text_path, 'w') as f:
                for item in results[split]:
                    f.write(f"{item['id']} {item['utterance']}\n")
            
            # Update utt2spk file
            utt2spk_path = os.path.join(split_output_dir, "utt2spk")
            with open(utt2spk_path, 'w') as f:
                for item in results[split]:
                    f.write(f"{item['id']} {item['id'].split('_')[0]}\n")
            
            # Update spk2utt file
            spk2utt_path = os.path.join(split_output_dir, "spk2utt")
            speakers = {}
            for item in results[split]:
                speaker = item['id'].split('_')[0]
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(item['id'])
            
            with open(spk2utt_path, 'w') as f:
                for speaker, utts in speakers.items():
                    f.write(f"{speaker} {' '.join(utts)}\n")
        
        logger.info(f"Processed dataset: {dataset_path}")
        return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Acoustic Data Augmentation")
    parser.add_argument("--output-dir", type=str, default="augmented_data",
                        help="Output directory for augmented audio")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate for audio")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for parallel processing")
    parser.add_argument("--num-augmentations", type=int, default=5,
                        help="Number of augmented versions to generate per file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--audio-file", type=str, default=None,
                        help="Path to audio file to augment")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Directory containing noise samples")
    parser.add_argument("--rir-dir", type=str, default=None,
                        help="Directory containing RIR samples")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "sample_rate": args.sample_rate,
        "num_workers": args.num_workers,
        "num_augmentations": args.num_augmentations,
        "noise_dir": args.noise_dir,
        "rir_dir": args.rir_dir
    }
    
    # Create data augmentation
    augmentation = DataAugmentation(config)
    
    # Process dataset
    if args.dataset:
        augmentation.process_dataset(args.dataset)
    
    # Augment audio file
    if args.audio_file:
        output_file = os.path.join(args.output_dir, "augmented.wav")
        success = augmentation.augment_audio(args.audio_file, output_file)
        if success:
            print(f"Augmented audio saved to {output_file}")
        else:
            print("Failed to augment audio")




















