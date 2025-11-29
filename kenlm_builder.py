#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KenLM N-gram Language Model Builder

This module provides tools for building n-gram language models using KenLM.
It generates training text from templates, builds the language model,
and provides utilities for integrating with ASR systems.

Features:
- Template-based corpus generation
- KenLM model building with configurable parameters
- Integration with WFST decoders
- Support for domain-specific vocabulary
- Perplexity evaluation
"""

import os
import re
import random
import subprocess
import logging
import tempfile
import itertools
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KenLMBuilder")

# Check if KenLM is installed
try:
    subprocess.run(["lmplz", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    KENLM_AVAILABLE = True
except (subprocess.SubprocessError, FileNotFoundError):
    KENLM_AVAILABLE = False
    logger.warning("KenLM tools not found. Model building will be disabled.")

# Default glossaries (can be overridden)
DEFAULT_QUANTITY_GLOSSARY = {
    # English Numbers
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", 
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "half": "0.5", "quarter": "0.25"
}

DEFAULT_UNIT_GLOSSARY = {
    "kilogram": "kg", "kilo": "kg", "kg": "kg",
    "gram": "g", "g": "g", 
    "milligram": "mg", "mg": "mg",
    "milliliter": "ml", "ml": "ml",
    "liter": "l", "litre": "l", "l": "l",
    "packet": "pcs", "piece": "pcs", "pieces": "pcs", "pcs": "pcs",
    "strip": "strip", "strips": "strip", 
    "tablet": "tablet", "tablets": "tablet", "tab": "tablet",
    "bottle": "bottle", "bottles": "bottle",
    "box": "box", "boxes": "box"
}

DEFAULT_PRODUCT_GLOSSARY = {
    # Paracetamol and brands
    "paracetamol": "paracetamol", "para": "paracetamol",
    "crocin": "crocin", "crosin": "crocin", "crossing": "crocin", "cross in": "crocin",
    "dolo": "dolo 650", "dolo650": "dolo 650", "650": "dolo 650", "dolo 6 50": "dolo 650",
    "dollar": "dolo 650", "dollars": "dolo 650", "dollar six": "dolo 650",
    "calpol": "calpol", "cal pol": "calpol",
    
    # Allergy medicines
    "cetirizine": "cetirizine", "cetzine": "cetirizine", "cetrizine": "cetirizine", 
    "allergy": "cetirizine", "allergy medicine": "cetirizine",
    
    # Stomach medicines
    "omez": "omez", "omeprazole": "omez", "gas medicine": "omez",
    
    # Cough medicines
    "benadryl": "benadryl", "benedryl": "benadryl", "benadryl syrup": "benadryl",
    "cough syrup": "cough syrup", "cough medicine": "cough syrup",
    
    # First aid
    "bandaid": "band-aid", "plaster": "band-aid", "band aid": "band-aid",
    "dettol": "dettol", "detol": "dettol", "antiseptic": "dettol",
    
    # Others
    "vicks": "vicks vaporub", "vaporub": "vicks vaporub", 
    "becosules": "becosules", "becosule": "becosules", "bikasul": "becosules",
    "vitamin": "multivitamin", "multivitamin": "multivitamin"
}


class KenLMBuilder:
    """
    Class for building KenLM n-gram language models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KenLMBuilder with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - quantity_glossary: Dictionary mapping quantity words to values
                - unit_glossary: Dictionary mapping unit words to canonical forms
                - product_glossary: Dictionary mapping product words to canonical forms
                - output_dir: Directory to save model files
                - order: N-gram order (default: 3)
                - prune: Whether to prune n-grams (default: True)
                - discount_fallback: Whether to use discount fallback (default: True)
                - corpus_size: Number of sentences to generate (default: 10000)
        """
        self.config = config or {}
        self.quantity_glossary = self.config.get('quantity_glossary', DEFAULT_QUANTITY_GLOSSARY)
        self.unit_glossary = self.config.get('unit_glossary', DEFAULT_UNIT_GLOSSARY)
        self.product_glossary = self.config.get('product_glossary', DEFAULT_PRODUCT_GLOSSARY)
        self.output_dir = self.config.get('output_dir', 'kenlm_model')
        self.order = self.config.get('order', 3)
        self.prune = self.config.get('prune', True)
        self.discount_fallback = self.config.get('discount_fallback', True)
        self.corpus_size = self.config.get('corpus_size', 10000)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize templates
        self.templates = [
            "{quantity} {unit} {product}",
            "{product} {quantity} {unit}",
        ]
        
        logger.info(f"KenLMBuilder initialized with {len(self.quantity_glossary)} quantities, "
                   f"{len(self.unit_glossary)} units, and {len(self.product_glossary)} products")
    
    def generate_corpus(self, output_file: str, size: int = None) -> bool:
        """
        Generate a corpus of sentences from templates.
        
        Args:
            output_file: Output corpus filename
            size: Number of sentences to generate (default: self.corpus_size)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            size = size or self.corpus_size
            
            # Get all possible values
            quantities = list(self.quantity_glossary.keys())
            units = list(self.unit_glossary.keys())
            products = list(self.product_glossary.keys())
            
            # Generate sentences
            sentences = set()
            
            # Generate sentences from templates
            while len(sentences) < size:
                template = random.choice(self.templates)
                quantity = random.choice(quantities)
                unit = random.choice(units)
                product = random.choice(products)
                
                sentence = template.format(
                    quantity=quantity,
                    unit=unit,
                    product=product
                )
                
                sentences.add(sentence.lower())
            
            # Write corpus to file
            with open(output_file, 'w') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            
            logger.info(f"Generated corpus with {len(sentences)} sentences")
            return True
            
        except Exception as e:
            logger.error(f"Error generating corpus: {e}")
            return False
    
    def generate_all_combinations(self, output_file: str, max_size: int = None) -> bool:
        """
        Generate all possible combinations of templates.
        
        Args:
            output_file: Output corpus filename
            max_size: Maximum number of sentences to generate (default: None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all possible values
            quantities = list(self.quantity_glossary.keys())
            units = list(self.unit_glossary.keys())
            products = list(self.product_glossary.keys())
            
            # Calculate total number of combinations
            total_combinations = len(quantities) * len(units) * len(products) * len(self.templates)
            logger.info(f"Total possible combinations: {total_combinations}")
            
            # Limit the number of combinations if needed
            if max_size and max_size < total_combinations:
                logger.info(f"Limiting to {max_size} combinations")
                
                # Sample quantities, units, and products
                if len(quantities) > 10:
                    quantities = random.sample(quantities, 10)
                if len(units) > 5:
                    units = random.sample(units, 5)
                if len(products) > 20:
                    products = random.sample(products, 20)
            
            # Generate all combinations
            sentences = set()
            
            for template in self.templates:
                for quantity in quantities:
                    for unit in units:
                        for product in products:
                            sentence = template.format(
                                quantity=quantity,
                                unit=unit,
                                product=product
                            )
                            sentences.add(sentence.lower())
                            
                            # Check if we've reached the maximum size
                            if max_size and len(sentences) >= max_size:
                                break
                        
                        if max_size and len(sentences) >= max_size:
                            break
                    
                    if max_size and len(sentences) >= max_size:
                        break
                
                if max_size and len(sentences) >= max_size:
                    break
            
            # Write corpus to file
            with open(output_file, 'w') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            
            logger.info(f"Generated corpus with {len(sentences)} sentences")
            return True
            
        except Exception as e:
            logger.error(f"Error generating all combinations: {e}")
            return False
    
    def build_model(self, corpus_file: str, output_file: str) -> bool:
        """
        Build a KenLM n-gram language model.
        
        Args:
            corpus_file: Input corpus filename
            output_file: Output model filename
            
        Returns:
            True if successful, False otherwise
        """
        if not KENLM_AVAILABLE:
            logger.error("KenLM tools not found, cannot build model")
            return False
        
        try:
            # Build arpa file
            arpa_file = output_file + ".arpa"
            
            cmd = [
                "lmplz",
                f"--order={self.order}",
                "--text=" + corpus_file,
                "--arpa=" + arpa_file
            ]
            
            if self.prune:
                cmd.append("--prune=0|0|1")
            
            if self.discount_fallback:
                cmd.append("--discount_fallback")
            
            subprocess.run(cmd, check=True)
            
            # Build binary file
            subprocess.run([
                "build_binary",
                arpa_file,
                output_file
            ], check=True)
            
            logger.info(f"KenLM model built: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error building KenLM model: {e}")
            return False
    
    def evaluate_model(self, model_file: str, test_file: str) -> float:
        """
        Evaluate a KenLM model using perplexity.
        
        Args:
            model_file: KenLM model filename
            test_file: Test corpus filename
            
        Returns:
            Perplexity score (lower is better)
        """
        if not KENLM_AVAILABLE:
            logger.error("KenLM tools not found, cannot evaluate model")
            return float('inf')
        
        try:
            # Run query command
            result = subprocess.run([
                "query",
                model_file,
                "-v", "summary",
                test_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            
            # Parse perplexity from output
            for line in result.stderr.split('\n'):
                if "Perplexity" in line:
                    perplexity = float(line.split('=')[1].strip())
                    logger.info(f"Model perplexity: {perplexity}")
                    return perplexity
            
            logger.warning("Could not parse perplexity from output")
            return float('inf')
            
        except Exception as e:
            logger.error(f"Error evaluating KenLM model: {e}")
            return float('inf')
    
    def convert_to_fst(self, model_file: str, output_file: str) -> bool:
        """
        Convert a KenLM model to an FST for use with WFST decoders.
        
        Args:
            model_file: KenLM model filename
            output_file: Output FST filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a vocabulary file
            vocab_file = os.path.join(self.output_dir, "vocab.txt")
            
            # Collect all words
            words = set()
            words.update(self.quantity_glossary.keys())
            words.update(self.unit_glossary.keys())
            words.update(self.product_glossary.keys())
            
            # Write vocabulary to file
            with open(vocab_file, 'w') as f:
                for word in sorted(words):
                    f.write(word.lower() + '\n')
            
            # Convert KenLM to FST
            # This requires kaldi-lm tools
            try:
                subprocess.run([
                    "arpa2fst",
                    "--disambig-symbol=#0",
                    "--read-symbol-table=" + vocab_file,
                    model_file + ".arpa",
                    output_file
                ], check=True)
                
                logger.info(f"KenLM model converted to FST: {output_file}")
                return True
                
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.error("arpa2fst command not found, cannot convert model to FST")
                return False
            
        except Exception as e:
            logger.error(f"Error converting KenLM model to FST: {e}")
            return False
    
    def load_glossaries_from_file(self, filename: str) -> bool:
        """
        Load glossaries from a JSON file.
        
        Args:
            filename: Input JSON filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if "quantity_glossary" in data:
                self.quantity_glossary = data["quantity_glossary"]
            
            if "unit_glossary" in data:
                self.unit_glossary = data["unit_glossary"]
            
            if "product_glossary" in data:
                self.product_glossary = data["product_glossary"]
            
            logger.info(f"Glossaries loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading glossaries: {e}")
            return False
    
    def save_glossaries_to_file(self, filename: str) -> bool:
        """
        Save glossaries to a JSON file.
        
        Args:
            filename: Output JSON filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "quantity_glossary": self.quantity_glossary,
                "unit_glossary": self.unit_glossary,
                "product_glossary": self.product_glossary
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Glossaries saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving glossaries: {e}")
            return False
    
    def add_templates(self, templates: List[str]) -> None:
        """
        Add templates for corpus generation.
        
        Args:
            templates: List of templates to add
        """
        self.templates.extend(templates)
        logger.info(f"Added {len(templates)} templates, total: {len(self.templates)}")
    
    def build_pipeline(self) -> bool:
        """
        Run the complete model building pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate corpus
            corpus_file = os.path.join(self.output_dir, "corpus.txt")
            if not self.generate_corpus(corpus_file):
                return False
            
            # Split corpus into train and test sets
            train_file = os.path.join(self.output_dir, "train.txt")
            test_file = os.path.join(self.output_dir, "test.txt")
            
            with open(corpus_file, 'r') as f:
                lines = f.readlines()
            
            # Shuffle lines
            random.shuffle(lines)
            
            # Split 90/10
            split_idx = int(len(lines) * 0.9)
            train_lines = lines[:split_idx]
            test_lines = lines[split_idx:]
            
            with open(train_file, 'w') as f:
                f.writelines(train_lines)
            
            with open(test_file, 'w') as f:
                f.writelines(test_lines)
            
            # Build model
            model_file = os.path.join(self.output_dir, "lm.binary")
            if not self.build_model(train_file, model_file):
                return False
            
            # Evaluate model
            perplexity = self.evaluate_model(model_file, test_file)
            
            # Convert to FST
            fst_file = os.path.join(self.output_dir, "G.fst")
            self.convert_to_fst(model_file, fst_file)
            
            logger.info(f"Model building pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in model building pipeline: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KenLM N-gram Language Model Builder")
    parser.add_argument("--glossary-file", type=str, default=None,
                        help="JSON file containing glossaries")
    parser.add_argument("--output-dir", type=str, default="kenlm_model",
                        help="Output directory for model files")
    parser.add_argument("--order", type=int, default=3,
                        help="N-gram order")
    parser.add_argument("--corpus-size", type=int, default=10000,
                        help="Number of sentences to generate")
    parser.add_argument("--no-prune", action="store_true",
                        help="Disable n-gram pruning")
    parser.add_argument("--no-discount-fallback", action="store_true",
                        help="Disable discount fallback")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "order": args.order,
        "prune": not args.no_prune,
        "discount_fallback": not args.no_discount_fallback,
        "corpus_size": args.corpus_size
    }
    
    # Create KenLM builder
    kenlm_builder = KenLMBuilder(config)
    
    # Load glossaries if specified
    if args.glossary_file:
        kenlm_builder.load_glossaries_from_file(args.glossary_file)
    
    # Run the pipeline
    if kenlm_builder.build_pipeline():
        print(f"KenLM model built successfully in {args.output_dir}")
    else:
        print("Failed to build KenLM model")




















