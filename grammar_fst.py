#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grammar FST Generator for Constrained Decoding

This module provides tools for creating Finite State Transducers (FSTs)
that represent grammar constraints for ASR decoding. The grammar is designed
specifically for pharmaceutical/medicine orders with structured output.

Features:
- Create FSTs for [quantity unit product] patterns
- Support for both OpenFST and Kaldi-compatible formats
- Dynamic grammar generation from glossaries
- Visualization of grammar FSTs
- Integration with WFST decoders
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GrammarFST")

# Check if OpenFST is installed
try:
    subprocess.run(["fstinfo", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    OPENFST_AVAILABLE = True
except (subprocess.SubprocessError, FileNotFoundError):
    OPENFST_AVAILABLE = False
    logger.warning("OpenFST command-line tools not found. Some features will be disabled.")

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


class GrammarFST:
    """
    Class for creating and managing grammar FSTs for constrained decoding.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the GrammarFST with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - quantity_glossary: Dictionary mapping quantity words to values
                - unit_glossary: Dictionary mapping unit words to canonical forms
                - product_glossary: Dictionary mapping product words to canonical forms
                - output_dir: Directory to save FST files
                - use_openfst: Whether to use OpenFST command-line tools
        """
        self.config = config or {}
        self.quantity_glossary = self.config.get('quantity_glossary', DEFAULT_QUANTITY_GLOSSARY)
        self.unit_glossary = self.config.get('unit_glossary', DEFAULT_UNIT_GLOSSARY)
        self.product_glossary = self.config.get('product_glossary', DEFAULT_PRODUCT_GLOSSARY)
        self.output_dir = self.config.get('output_dir', 'grammar_fst')
        self.use_openfst = self.config.get('use_openfst', OPENFST_AVAILABLE)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize symbol tables
        self.word_symbols = set()
        self.phone_symbols = set()
        
        logger.info(f"GrammarFST initialized with {len(self.quantity_glossary)} quantities, "
                   f"{len(self.unit_glossary)} units, and {len(self.product_glossary)} products")
    
    def _add_to_symbols(self, words: List[str]) -> None:
        """
        Add words to the symbol table.
        
        Args:
            words: List of words to add
        """
        for word in words:
            self.word_symbols.add(word.lower())
            # Add phones (simplified for now, in a real system you'd use a G2P)
            for char in word.lower():
                if char.isalnum():
                    self.phone_symbols.add(char)
    
    def _write_symbol_table(self, symbols: Set[str], filename: str) -> None:
        """
        Write a symbol table to a file.
        
        Args:
            symbols: Set of symbols
            filename: Output filename
        """
        with open(filename, 'w') as f:
            # Write epsilon (empty string) symbol
            f.write("<eps> 0\n")
            
            # Write all other symbols
            for i, symbol in enumerate(sorted(symbols), 1):
                f.write(f"{symbol} {i}\n")
    
    def _create_lexicon_fst(self, output_file: str) -> bool:
        """
        Create a lexicon FST mapping words to phones.
        
        Args:
            output_file: Output FST filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_openfst:
            logger.warning("OpenFST not available, skipping lexicon FST creation")
            return False
        
        try:
            # Create a temporary lexicon file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                lexicon_file = f.name
                
                # Write word -> phones entries (simplified)
                for word in sorted(self.word_symbols):
                    # Simple phone mapping (in a real system, use a G2P)
                    phones = ' '.join(char for char in word if char.isalnum())
                    f.write(f"{word} {phones}\n")
            
            # Create symbol tables
            word_syms_file = os.path.join(self.output_dir, "words.txt")
            phone_syms_file = os.path.join(self.output_dir, "phones.txt")
            
            self._write_symbol_table(self.word_symbols, word_syms_file)
            self._write_symbol_table(self.phone_symbols, phone_syms_file)
            
            # Create the lexicon FST
            subprocess.run([
                "fstcompile",
                "--isymbols=" + word_syms_file,
                "--osymbols=" + phone_syms_file,
                "--keep_isymbols=true",
                "--keep_osymbols=true",
                lexicon_file,
                output_file
            ], check=True)
            
            # Clean up
            os.unlink(lexicon_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating lexicon FST: {e}")
            return False
    
    def _create_grammar_fst(self, output_file: str) -> bool:
        """
        Create a grammar FST for constrained decoding.
        
        Args:
            output_file: Output FST filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_openfst:
            logger.warning("OpenFST not available, skipping grammar FST creation")
            return False
        
        try:
            # Create a temporary grammar file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                grammar_file = f.name
                
                # Define states
                START_STATE = 0
                QUANTITY_STATE = 1
                UNIT_STATE = 2
                PRODUCT_STATE = 3
                FINAL_STATE = 4
                
                # Pattern 1: [quantity] [unit] [product]
                # START -> QUANTITY
                for qty_word in self.quantity_glossary:
                    f.write(f"{START_STATE} {QUANTITY_STATE} {qty_word} {qty_word}\n")
                    self._add_to_symbols([qty_word])
                
                # QUANTITY -> UNIT
                for unit_word in self.unit_glossary:
                    f.write(f"{QUANTITY_STATE} {UNIT_STATE} {unit_word} {unit_word}\n")
                    self._add_to_symbols([unit_word])
                
                # UNIT -> PRODUCT
                for product_word in self.product_glossary:
                    f.write(f"{UNIT_STATE} {PRODUCT_STATE} {product_word} {product_word}\n")
                    self._add_to_symbols([product_word])
                
                # PRODUCT -> FINAL (end of utterance)
                f.write(f"{PRODUCT_STATE} {FINAL_STATE} <eps> <eps>\n")
                
                # Pattern 2: [product] [quantity] [unit]
                # START -> PRODUCT
                for product_word in self.product_glossary:
                    f.write(f"{START_STATE} {PRODUCT_STATE} {product_word} {product_word}\n")
                
                # PRODUCT -> QUANTITY
                for qty_word in self.quantity_glossary:
                    f.write(f"{PRODUCT_STATE} {QUANTITY_STATE} {qty_word} {qty_word}\n")
                
                # QUANTITY -> UNIT -> FINAL already defined above
                
                # Mark final state
                f.write(f"{FINAL_STATE}\n")
            
            # Create symbol tables
            word_syms_file = os.path.join(self.output_dir, "words.txt")
            
            self._write_symbol_table(self.word_symbols, word_syms_file)
            
            # Create the grammar FST
            subprocess.run([
                "fstcompile",
                "--isymbols=" + word_syms_file,
                "--osymbols=" + word_syms_file,
                "--keep_isymbols=true",
                "--keep_osymbols=true",
                grammar_file,
                output_file
            ], check=True)
            
            # Clean up
            os.unlink(grammar_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating grammar FST: {e}")
            return False
    
    def _create_vosk_grammar(self, output_file: str) -> bool:
        """
        Create a Vosk-compatible grammar JSON file.
        
        Args:
            output_file: Output JSON filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create grammar patterns
            grammar = {"grammar": []}
            
            # Pattern 1: [quantity] [unit] [product]
            qty_patterns = []
            unit_patterns = []
            product_patterns = []
            
            # Group similar terms for quantities
            qty_groups = {}
            for qty_word, value in self.quantity_glossary.items():
                if value not in qty_groups:
                    qty_groups[value] = []
                qty_groups[value].append(qty_word)
            
            # Create quantity patterns
            for value, words in qty_groups.items():
                qty_patterns.append(f"[{value}|{('|'.join(words))}]")
            
            # Group similar terms for units
            unit_groups = {}
            for unit_word, canonical in self.unit_glossary.items():
                if canonical not in unit_groups:
                    unit_groups[canonical] = []
                unit_groups[canonical].append(unit_word)
            
            # Create unit patterns
            for canonical, words in unit_groups.items():
                unit_patterns.append(f"[{canonical}|{('|'.join(words))}]")
            
            # Group similar terms for products
            product_groups = {}
            for product_word, canonical in self.product_glossary.items():
                if canonical not in product_groups:
                    product_groups[canonical] = []
                product_groups[canonical].append(product_word)
            
            # Create product patterns
            for canonical, words in product_groups.items():
                product_patterns.append(f"[{canonical}|{('|'.join(words))}]")
            
            # Create combined patterns
            # Pattern 1: [quantity] [unit] [product]
            for qty_pattern in qty_patterns:
                for unit_pattern in unit_patterns:
                    for product_pattern in product_patterns:
                        grammar["grammar"].append(f"{qty_pattern} {unit_pattern} {product_pattern}")
            
            # Pattern 2: [product] [quantity] [unit]
            for product_pattern in product_patterns:
                for qty_pattern in qty_patterns:
                    for unit_pattern in unit_patterns:
                        grammar["grammar"].append(f"{product_pattern} {qty_pattern} {unit_pattern}")
            
            # Write grammar to file
            with open(output_file, 'w') as f:
                json.dump(grammar, f, indent=2)
            
            logger.info(f"Vosk grammar created with {len(grammar['grammar'])} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Vosk grammar: {e}")
            return False
    
    def _create_kaldi_grammar(self, output_file: str) -> bool:
        """
        Create a Kaldi-compatible grammar file.
        
        Args:
            output_file: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary grammar file
            with open(output_file, 'w') as f:
                # Define non-terminals
                f.write("nonterminal QUANTITY;\n")
                f.write("nonterminal UNIT;\n")
                f.write("nonterminal PRODUCT;\n")
                f.write("nonterminal ORDER;\n")
                f.write("\n")
                
                # Define rules for quantities
                f.write("# Quantity rules\n")
                for qty_word, value in self.quantity_glossary.items():
                    f.write(f"QUANTITY -> {qty_word};\n")
                f.write("\n")
                
                # Define rules for units
                f.write("# Unit rules\n")
                for unit_word in self.unit_glossary:
                    f.write(f"UNIT -> {unit_word};\n")
                f.write("\n")
                
                # Define rules for products
                f.write("# Product rules\n")
                for product_word in self.product_glossary:
                    f.write(f"PRODUCT -> {product_word};\n")
                f.write("\n")
                
                # Define order patterns
                f.write("# Order patterns\n")
                f.write("ORDER -> QUANTITY UNIT PRODUCT;\n")
                f.write("ORDER -> PRODUCT QUANTITY UNIT;\n")
            
            logger.info(f"Kaldi grammar created at {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Kaldi grammar: {e}")
            return False
    
    def visualize_fst(self, fst_file: str, output_file: str) -> bool:
        """
        Visualize an FST using GraphViz.
        
        Args:
            fst_file: Input FST filename
            output_file: Output image filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_openfst:
            logger.warning("OpenFST not available, skipping FST visualization")
            return False
        
        try:
            # Convert FST to dot format
            dot_file = output_file + ".dot"
            subprocess.run([
                "fstdraw",
                "--portrait=true",
                fst_file,
                dot_file
            ], check=True)
            
            # Convert dot to image
            subprocess.run([
                "dot",
                "-Tpng",
                "-o" + output_file,
                dot_file
            ], check=True)
            
            # Clean up
            os.unlink(dot_file)
            
            logger.info(f"FST visualization saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing FST: {e}")
            return False
    
    def create_grammar(self, format_type: str = "vosk") -> str:
        """
        Create a grammar in the specified format.
        
        Args:
            format_type: Grammar format type ('vosk', 'openfst', 'kaldi')
            
        Returns:
            Path to the created grammar file
        """
        try:
            if format_type == "vosk":
                # Create Vosk grammar
                output_file = os.path.join(self.output_dir, "vosk_grammar.json")
                if self._create_vosk_grammar(output_file):
                    return output_file
            
            elif format_type == "openfst":
                # Create OpenFST grammar
                if not self.use_openfst:
                    logger.error("OpenFST not available, cannot create OpenFST grammar")
                    return ""
                
                # Create grammar FST
                grammar_fst = os.path.join(self.output_dir, "G.fst")
                if self._create_grammar_fst(grammar_fst):
                    # Create lexicon FST
                    lexicon_fst = os.path.join(self.output_dir, "L.fst")
                    if self._create_lexicon_fst(lexicon_fst):
                        # Compose lexicon and grammar
                        lg_fst = os.path.join(self.output_dir, "LG.fst")
                        subprocess.run([
                            "fstcompose",
                            lexicon_fst,
                            grammar_fst,
                            lg_fst
                        ], check=True)
                        
                        # Visualize if requested
                        if self.config.get('visualize', False):
                            self.visualize_fst(lg_fst, os.path.join(self.output_dir, "LG.png"))
                        
                        return lg_fst
            
            elif format_type == "kaldi":
                # Create Kaldi grammar
                output_file = os.path.join(self.output_dir, "kaldi_grammar.txt")
                if self._create_kaldi_grammar(output_file):
                    return output_file
            
            else:
                logger.error(f"Unsupported grammar format: {format_type}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error creating grammar: {e}")
            return ""
    
    def create_all_grammars(self) -> Dict[str, str]:
        """
        Create grammars in all supported formats.
        
        Returns:
            Dictionary mapping format types to grammar file paths
        """
        result = {}
        
        for format_type in ["vosk", "openfst", "kaldi"]:
            grammar_file = self.create_grammar(format_type)
            if grammar_file:
                result[format_type] = grammar_file
        
        return result
    
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


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grammar FST Generator")
    parser.add_argument("--glossary-file", type=str, default=None,
                        help="JSON file containing glossaries")
    parser.add_argument("--output-dir", type=str, default="grammar_fst",
                        help="Output directory for FST files")
    parser.add_argument("--format", type=str, default="vosk",
                        choices=["vosk", "openfst", "kaldi", "all"],
                        help="Grammar format type")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize FSTs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "visualize": args.visualize
    }
    
    # Create grammar FST
    grammar_fst = GrammarFST(config)
    
    # Load glossaries if specified
    if args.glossary_file:
        grammar_fst.load_glossaries_from_file(args.glossary_file)
    
    # Create grammar
    if args.format == "all":
        grammar_files = grammar_fst.create_all_grammars()
        for format_type, grammar_file in grammar_files.items():
            print(f"{format_type.capitalize()} grammar created: {grammar_file}")
    else:
        grammar_file = grammar_fst.create_grammar(args.format)
        if grammar_file:
            print(f"{args.format.capitalize()} grammar created: {grammar_file}")
        else:
            print(f"Failed to create {args.format} grammar")




















