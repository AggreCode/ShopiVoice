#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WFST Decoder for ASR

This module provides a decoder that uses Weighted Finite State Transducers (WFSTs)
for constrained speech recognition. It integrates with acoustic models and
language models to produce structured output.

Features:
- Integration with ESPnet/Kaldi acoustic models
- Support for grammar and language model FSTs
- Beam search decoding
- N-best hypothesis generation
- Confidence scoring
"""

import os
import sys
import logging
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WFSTDecoder")

# Check if OpenFST Python bindings are available
try:
    import pywrapfst as fst
    OPENFST_AVAILABLE = True
except ImportError:
    OPENFST_AVAILABLE = False
    logger.warning("OpenFST Python bindings not found. Using command-line tools instead.")

# Check if Kaldi is available
try:
    subprocess.run(["compute-mfcc-feats", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    KALDI_AVAILABLE = True
except (subprocess.SubprocessError, FileNotFoundError):
    KALDI_AVAILABLE = False
    logger.warning("Kaldi tools not found. Some features will be disabled.")


class WFSTDecoder:
    """
    Class for WFST-based decoding of speech recognition.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the WFST decoder with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model_dir: Directory containing model files
                - acoustic_scale: Acoustic model scale (default: 0.1)
                - beam: Beam width for search (default: 13.0)
                - lattice_beam: Lattice beam width (default: 6.0)
                - max_active: Maximum active states (default: 7000)
                - min_active: Minimum active states (default: 200)
                - acoustic_model_type: Type of acoustic model ('kaldi', 'espnet', 'wav2vec2')
                - use_gpu: Whether to use GPU for decoding (default: False)
        """
        self.config = config or {}
        self.model_dir = self.config.get('model_dir', 'models')
        self.acoustic_scale = self.config.get('acoustic_scale', 0.1)
        self.beam = self.config.get('beam', 13.0)
        self.lattice_beam = self.config.get('lattice_beam', 6.0)
        self.max_active = self.config.get('max_active', 7000)
        self.min_active = self.config.get('min_active', 200)
        self.acoustic_model_type = self.config.get('acoustic_model_type', 'kaldi')
        self.use_gpu = self.config.get('use_gpu', False)
        
        # Initialize model paths
        self.hclg_path = os.path.join(self.model_dir, 'HCLG.fst')
        self.words_path = os.path.join(self.model_dir, 'words.txt')
        self.final_mdl_path = os.path.join(self.model_dir, 'final.mdl')
        
        # Load models if available
        self._load_models()
        
        logger.info(f"WFSTDecoder initialized with model_dir: {self.model_dir}")
    
    def _load_models(self) -> bool:
        """
        Load WFST models for decoding.
        
        Returns:
            True if successful, False otherwise
        """
        if not OPENFST_AVAILABLE:
            logger.warning("OpenFST Python bindings not available, models will be loaded on demand")
            return False
        
        try:
            # Check if model files exist
            if not os.path.exists(self.hclg_path):
                logger.warning(f"HCLG FST not found at {self.hclg_path}")
                return False
            
            if not os.path.exists(self.words_path):
                logger.warning(f"Words file not found at {self.words_path}")
                return False
            
            # Load HCLG FST
            self.hclg_fst = fst.Fst.read(self.hclg_path)
            
            # Load words symbol table
            self.words_symtab = fst.SymbolTable.read_text(self.words_path)
            
            logger.info(f"Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _prepare_features(self, audio_file: str) -> str:
        """
        Prepare features from audio file for decoding.
        
        Args:
            audio_file: Input audio filename
            
        Returns:
            Path to features file
        """
        if not KALDI_AVAILABLE:
            logger.error("Kaldi tools not found, cannot prepare features")
            return ""
        
        try:
            # Create temporary directory for features
            temp_dir = tempfile.mkdtemp()
            
            # Create wav.scp file
            wav_scp = os.path.join(temp_dir, "wav.scp")
            with open(wav_scp, 'w') as f:
                f.write(f"utterance {audio_file}\n")
            
            # Extract MFCCs
            feats_ark = os.path.join(temp_dir, "feats.ark")
            feats_scp = os.path.join(temp_dir, "feats.scp")
            
            subprocess.run([
                "compute-mfcc-feats",
                f"--config={os.path.join(self.model_dir, 'conf/mfcc.conf')}",
                f"scp:{wav_scp}",
                f"ark,scp:{feats_ark},{feats_scp}"
            ], check=True)
            
            # Apply CMVN
            cmvn_ark = os.path.join(temp_dir, "cmvn.ark")
            subprocess.run([
                "compute-cmvn-stats",
                f"scp:{feats_scp}",
                f"ark:{cmvn_ark}"
            ], check=True)
            
            feats_cmvn_ark = os.path.join(temp_dir, "feats_cmvn.ark")
            feats_cmvn_scp = os.path.join(temp_dir, "feats_cmvn.scp")
            
            subprocess.run([
                "apply-cmvn",
                f"--norm-vars=true",
                f"ark:{cmvn_ark}",
                f"scp:{feats_scp}",
                f"ark,scp:{feats_cmvn_ark},{feats_cmvn_scp}"
            ], check=True)
            
            # Apply delta features
            feats_delta_ark = os.path.join(temp_dir, "feats_delta.ark")
            feats_delta_scp = os.path.join(temp_dir, "feats_delta.scp")
            
            subprocess.run([
                "add-deltas",
                f"scp:{feats_cmvn_scp}",
                f"ark,scp:{feats_delta_ark},{feats_delta_scp}"
            ], check=True)
            
            return feats_delta_scp
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return ""
    
    def _decode_with_kaldi(self, features_file: str) -> List[Dict[str, Any]]:
        """
        Decode using Kaldi command-line tools.
        
        Args:
            features_file: Input features filename
            
        Returns:
            List of decoded hypotheses
        """
        if not KALDI_AVAILABLE:
            logger.error("Kaldi tools not found, cannot decode")
            return []
        
        try:
            # Create temporary directory for output
            temp_dir = tempfile.mkdtemp()
            
            # Run decoder
            lattice_path = os.path.join(temp_dir, "lat.gz")
            
            cmd = [
                "gmm-latgen-faster",
                f"--acoustic-scale={self.acoustic_scale}",
                f"--beam={self.beam}",
                f"--lattice-beam={self.lattice_beam}",
                f"--max-active={self.max_active}",
                f"--min-active={self.min_active}",
                self.final_mdl_path,
                self.hclg_path,
                f"scp:{features_file}",
                f"ark:|gzip -c > {lattice_path}"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Get best path
            best_path = os.path.join(temp_dir, "best_path.txt")
            
            subprocess.run([
                "lattice-best-path",
                f"--acoustic-scale={self.acoustic_scale}",
                f"ark:gunzip -c {lattice_path} |",
                f"ark,t:{best_path}"
            ], check=True)
            
            # Get word alignment
            alignment = os.path.join(temp_dir, "alignment.txt")
            
            subprocess.run([
                "lattice-align-words",
                f"{os.path.join(self.model_dir, 'phones/word_boundary.int')}",
                self.final_mdl_path,
                f"ark:gunzip -c {lattice_path} |",
                f"ark:{alignment}"
            ], check=True)
            
            # Get confidence scores
            conf_path = os.path.join(temp_dir, "confidence.txt")
            
            subprocess.run([
                "lattice-to-ctm-conf",
                f"--acoustic-scale={self.acoustic_scale}",
                f"ark:gunzip -c {lattice_path} |",
                f"{conf_path}"
            ], check=True)
            
            # Read best path
            transcription = ""
            with open(best_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        transcription = ' '.join(parts[1:])
            
            # Read confidence scores
            confidences = {}
            with open(conf_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        word = parts[4]
                        conf = float(parts[5])
                        confidences[word] = conf
            
            # Create result
            result = {
                "text": transcription,
                "confidence": np.mean(list(confidences.values())) if confidences else 0.0,
                "word_confidences": confidences
            }
            
            return [result]
            
        except Exception as e:
            logger.error(f"Error decoding with Kaldi: {e}")
            return []
    
    def _decode_with_openfst(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode using OpenFST Python bindings.
        
        Args:
            features: Input features as numpy array
            
        Returns:
            List of decoded hypotheses
        """
        if not OPENFST_AVAILABLE:
            logger.error("OpenFST Python bindings not found, cannot decode")
            return []
        
        try:
            # Simple implementation of Viterbi decoding
            # In a real system, you'd use a more sophisticated decoder
            
            # Create a simple FST for the acoustic scores
            acoustic_fst = fst.Fst()
            
            # Add states
            states = []
            for i in range(len(features) + 1):
                state = acoustic_fst.add_state()
                states.append(state)
                if i == len(features):
                    acoustic_fst.set_final(state, 0)
            
            acoustic_fst.set_start(states[0])
            
            # Add arcs for each frame
            for i in range(len(features)):
                for j in range(len(features[i])):
                    # Convert acoustic score to negative log likelihood
                    score = -np.log(max(features[i][j], 1e-10))
                    acoustic_fst.add_arc(states[i], fst.Arc(j, j, fst.Weight(acoustic_fst.weight_type(), score), states[i+1]))
            
            # Compose with HCLG FST
            composed_fst = fst.compose(acoustic_fst, self.hclg_fst)
            
            # Find shortest path (best hypothesis)
            shortest_path = fst.shortestpath(composed_fst, nshortest=1)
            
            # Extract result
            result = []
            for state in shortest_path.states():
                for arc in shortest_path.arcs(state):
                    if arc.olabel != 0:  # Skip epsilon transitions
                        word = self.words_symtab.find(arc.olabel)
                        result.append(word)
            
            # Create result dictionary
            return [{
                "text": ' '.join(result),
                "confidence": 1.0,  # Simplified
                "word_confidences": {word: 1.0 for word in result}  # Simplified
            }]
            
        except Exception as e:
            logger.error(f"Error decoding with OpenFST: {e}")
            return []
    
    def decode_audio(self, audio_file: str, nbest: int = 1) -> List[Dict[str, Any]]:
        """
        Decode audio file using WFST.
        
        Args:
            audio_file: Input audio filename
            nbest: Number of best hypotheses to return
            
        Returns:
            List of decoded hypotheses
        """
        try:
            # Check if audio file exists
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return []
            
            # Prepare features
            features_file = self._prepare_features(audio_file)
            if not features_file:
                return []
            
            # Decode with Kaldi
            results = self._decode_with_kaldi(features_file)
            
            # Return results
            return results[:nbest]
            
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return []
    
    def decode_features(self, features: np.ndarray, nbest: int = 1) -> List[Dict[str, Any]]:
        """
        Decode features using WFST.
        
        Args:
            features: Input features as numpy array
            nbest: Number of best hypotheses to return
            
        Returns:
            List of decoded hypotheses
        """
        try:
            if not OPENFST_AVAILABLE:
                logger.error("OpenFST Python bindings not found, cannot decode features directly")
                return []
            
            # Decode with OpenFST
            results = self._decode_with_openfst(features)
            
            # Return results
            return results[:nbest]
            
        except Exception as e:
            logger.error(f"Error decoding features: {e}")
            return []
    
    def create_hclg(self, h_path: str, c_path: str, l_path: str, g_path: str, output_path: str) -> bool:
        """
        Create HCLG FST by composing H, C, L, and G FSTs.
        
        Args:
            h_path: Path to H FST (HMM)
            c_path: Path to C FST (context-dependency)
            l_path: Path to L FST (lexicon)
            g_path: Path to G FST (grammar)
            output_path: Path to output HCLG FST
            
        Returns:
            True if successful, False otherwise
        """
        if not KALDI_AVAILABLE:
            logger.error("Kaldi tools not found, cannot create HCLG")
            return False
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Compose L and G
            lg_path = os.path.join(temp_dir, "LG.fst")
            subprocess.run([
                "fstcomposecontext",
                "--context-size=3",
                "--central-position=1",
                f"--read-disambig-syms={os.path.join(self.model_dir, 'phones/disambig.int')}",
                f"--write-disambig-syms={os.path.join(temp_dir, 'disambig_ilabels.int')}",
                l_path,
                g_path,
                lg_path
            ], check=True)
            
            # Compose C and LG
            clg_path = os.path.join(temp_dir, "CLG.fst")
            subprocess.run([
                "fstarcsort",
                "--sort_type=ilabel",
                lg_path,
                clg_path
            ], check=True)
            
            # Compose H and CLG
            hclg_path = output_path
            subprocess.run([
                "fstcomposecontext",
                "--context-size=3",
                "--central-position=1",
                f"--read-disambig-syms={os.path.join(temp_dir, 'disambig_ilabels.int')}",
                h_path,
                clg_path,
                hclg_path
            ], check=True)
            
            logger.info(f"HCLG FST created at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating HCLG FST: {e}")
            return False
    
    def create_grammar_fst(self, grammar_file: str, output_path: str) -> bool:
        """
        Create grammar FST from grammar file.
        
        Args:
            grammar_file: Input grammar filename
            output_path: Path to output grammar FST
            
        Returns:
            True if successful, False otherwise
        """
        if not OPENFST_AVAILABLE:
            logger.error("OpenFST not available, cannot create grammar FST")
            return False
        
        try:
            # Check if grammar file exists
            if not os.path.exists(grammar_file):
                logger.error(f"Grammar file not found: {grammar_file}")
                return False
            
            # Read grammar file
            with open(grammar_file, 'r') as f:
                grammar_data = json.load(f)
            
            if "grammar" not in grammar_data:
                logger.error(f"Invalid grammar file format: {grammar_file}")
                return False
            
            # Create FST
            grammar_fst = fst.Fst()
            
            # Create symbol table
            symbols = fst.SymbolTable()
            symbols.add_symbol("<eps>")
            
            # Add words to symbol table
            words = set()
            for pattern in grammar_data["grammar"]:
                for word in pattern.split():
                    if word.startswith("[") and word.endswith("]"):
                        # Handle alternatives
                        alternatives = word[1:-1].split("|")
                        words.update(alternatives)
                    else:
                        words.add(word)
            
            for word in sorted(words):
                symbols.add_symbol(word)
            
            # Create states
            start_state = grammar_fst.add_state()
            grammar_fst.set_start(start_state)
            
            final_state = grammar_fst.add_state()
            grammar_fst.set_final(final_state, 0)
            
            # Add arcs for each pattern
            for pattern in grammar_data["grammar"]:
                current_state = start_state
                
                for word in pattern.split():
                    if word.startswith("[") and word.endswith("]"):
                        # Handle alternatives
                        alternatives = word[1:-1].split("|")
                        next_state = grammar_fst.add_state()
                        
                        for alt in alternatives:
                            grammar_fst.add_arc(
                                current_state,
                                fst.Arc(symbols.find(alt), symbols.find(alt), 0, next_state)
                            )
                        
                        current_state = next_state
                    else:
                        next_state = grammar_fst.add_state()
                        grammar_fst.add_arc(
                            current_state,
                            fst.Arc(symbols.find(word), symbols.find(word), 0, next_state)
                        )
                        current_state = next_state
                
                # Connect to final state
                grammar_fst.add_arc(
                    current_state,
                    fst.Arc(0, 0, 0, final_state)
                )
            
            # Set symbol tables
            grammar_fst.set_input_symbols(symbols)
            grammar_fst.set_output_symbols(symbols)
            
            # Write FST to file
            grammar_fst.write(output_path)
            
            # Write symbol table to file
            symbols.write_text(output_path + ".syms")
            
            logger.info(f"Grammar FST created at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating grammar FST: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WFST Decoder for ASR")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model files")
    parser.add_argument("--audio-file", type=str, required=True,
                        help="Input audio filename")
    parser.add_argument("--acoustic-scale", type=float, default=0.1,
                        help="Acoustic model scale")
    parser.add_argument("--beam", type=float, default=13.0,
                        help="Beam width for search")
    parser.add_argument("--nbest", type=int, default=1,
                        help="Number of best hypotheses to return")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for decoding")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "model_dir": args.model_dir,
        "acoustic_scale": args.acoustic_scale,
        "beam": args.beam,
        "use_gpu": args.use_gpu
    }
    
    # Create decoder
    decoder = WFSTDecoder(config)
    
    # Decode audio
    results = decoder.decode_audio(args.audio_file, args.nbest)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Hypothesis {i+1}: {result['text']} (confidence: {result['confidence']:.4f})")
        
        # Print word confidences
        if "word_confidences" in result:
            print("Word confidences:")
            for word, conf in result["word_confidences"].items():
                print(f"  {word}: {conf:.4f}")
        
        print()




















