#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Slot Parser and Entity Extraction

This module provides advanced slot parsing and entity extraction for
structured output from ASR transcriptions. It uses a combination of
rule-based parsing, fuzzy matching, and phonetic algorithms to extract
entities like quantities, units, and products.

Features:
- Robust entity extraction with phonetic matching
- Support for multiple languages and dialects
- Contextual disambiguation
- Confidence scoring
- Structured JSON output
"""

import os
import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SlotParser")

# Try to import optional dependencies
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed. Fuzzy matching will be limited.")

try:
    from phonemizer import phonemize
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    logger.warning("phonemizer not installed. Phonetic matching will be disabled.")

try:
    from word2number import w2n
    WORD2NUMBER_AVAILABLE = True
except ImportError:
    WORD2NUMBER_AVAILABLE = False
    logger.warning("word2number not installed. Word-to-number conversion will be limited.")

# Default glossaries (can be overridden)
DEFAULT_QUANTITY_GLOSSARY = {
    # English Numbers
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", 
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "half": "0.5", "quarter": "0.25",
    
    # Odia Numbers
    "shuna": "0", "adha": "0.5", "eka": "1", "gote": "1", "dui": "2", "tini": "3", 
    "chari": "4", "pancha": "5", "chha": "6", "sata": "7", "atha": "8", "naa": "9", 
    "dasa": "10", "egaara": "11", "baro": "12", "terah": "13", "choudah": "14", 
    "pandara": "15", "pombar": "15", "sohala": "16", "satara": "17", "athara": "18", 
    "annishi": "19", "kodie": "20"
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
    "box": "box", "boxes": "box",
    
    # Odia Units
    "keji": "kg", "gram": "g", "milli": "ml", "taa": "pcs"
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
    "vitamin": "multivitamin", "multivitamin": "multivitamin",
    
    # Odia terms
    "jara dawa": "paracetamol", "gas dawa": "omez", "kasa dawa": "cough syrup",
    "patti": "band-aid"
}

# Default inventory
DEFAULT_INVENTORY = {
    "paracetamol": { "brands": {"crocin": 30, "dolo 650": 32, "calpol": 25}, "unit": "strip" },
    "cetirizine": { "brands": {}, "unit": "strip", "price": 20 },
    "omez": { "brands": {}, "unit": "strip", "price": 55 },
    "cough syrup": { "brands": {"benadryl": 120, "grilinctus": 110}, "unit": "bottle" },
    "dettol": { "brands": {}, "unit": "bottle", "price": 80 },
    "band-aid": { "brands": {"hansaplast": 5}, "unit": "pcs", "price": 5 },
    "vicks vaporub": { "brands": {}, "unit": "bottle", "price": 45 },
    "multivitamin": { "brands": {"supradyn": 50, "a-to-z": 60}, "unit": "strip" },
    "becosules": { "brands": {}, "unit": "strip", "price": 48 }
}


class SlotParser:
    """
    Enhanced slot parser for extracting structured information from ASR transcriptions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the slot parser with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - quantity_glossary: Dictionary mapping quantity words to values
                - unit_glossary: Dictionary mapping unit words to canonical forms
                - product_glossary: Dictionary mapping product words to canonical forms
                - inventory: Dictionary containing product inventory information
                - language: Language code (default: 'en')
                - fuzzy_threshold: Threshold for fuzzy matching (default: 70)
                - phonetic_threshold: Threshold for phonetic matching (default: 70)
        """
        self.config = config or {}
        self.quantity_glossary = self.config.get('quantity_glossary', DEFAULT_QUANTITY_GLOSSARY)
        self.unit_glossary = self.config.get('unit_glossary', DEFAULT_UNIT_GLOSSARY)
        self.product_glossary = self.config.get('product_glossary', DEFAULT_PRODUCT_GLOSSARY)
        self.inventory = self.config.get('inventory', DEFAULT_INVENTORY)
        self.language = self.config.get('language', 'en')
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 70)
        self.phonetic_threshold = self.config.get('phonetic_threshold', 70)
        
        # Build search space for products
        self.search_space = self._build_search_space()
        
        # Initialize phonetic caches
        self._phonetic_cache = {}
        self._phonetic_inventory = None
        self._phonetic_choices = None
        
        logger.info(f"SlotParser initialized with {len(self.quantity_glossary)} quantities, "
                   f"{len(self.unit_glossary)} units, and {len(self.product_glossary)} products")
    
    def _build_search_space(self) -> Dict[str, str]:
        """
        Build a search space for product matching.
        
        Returns:
            Dictionary mapping product variations to canonical forms
        """
        search_map = {}
        
        # Add inventory products
        for product, info in self.inventory.items():
            search_map[product] = product
            for brand in info.get("brands", {}):
                search_map[brand] = product
        
        # Add product glossary items
        for key, value in self.product_glossary.items():
            if value in search_map:
                search_map[key] = search_map[value]
            elif value in self.inventory:
                search_map[key] = value
        
        return search_map
    
    def get_phonetic_representation(self, text: str) -> str:
        """
        Get phonetic representation of text.
        
        Args:
            text: Input text
            
        Returns:
            Phonetic representation of text
        """
        if not PHONEMIZER_AVAILABLE:
            return text.lower()
        
        # Check cache
        if text in self._phonetic_cache:
            return self._phonetic_cache[text]
        
        try:
            phonetic = phonemize([text], language=self.language, backend='espeak', preserve_punctuation=False)
            result = phonetic[0].strip()
            self._phonetic_cache[text] = result
            return result
        except Exception as e:
            logger.warning(f"Phonemizer error for '{text}': {e}")
            result = text.lower()
            self._phonetic_cache[text] = result
            return result
    
    def create_phonetic_mapping(self, glossary: Dict[str, str]) -> Dict[str, str]:
        """
        Create a phonetic mapping for a glossary.
        
        Args:
            glossary: Dictionary mapping words to canonical forms
            
        Returns:
            Dictionary mapping phonetic representations to canonical forms
        """
        phonetic_mapping = {}
        for key, value in glossary.items():
            phonetic_key = self.get_phonetic_representation(key)
            phonetic_mapping[phonetic_key] = value
            phonetic_mapping[key.lower()] = value
        return phonetic_mapping
    
    def fuzzy_match_with_phonetics(self, word: str, glossary: Dict[str, str], component_type: str) -> Optional[str]:
        """
        Match a word against a glossary using fuzzy and phonetic matching.
        
        Args:
            word: Input word
            glossary: Dictionary mapping words to canonical forms
            component_type: Type of component ('quantity', 'unit', 'product')
            
        Returns:
            Matched canonical form or None if no match
        """
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback to exact matching
            return glossary.get(word.lower())
        
        # Create phonetic mapping if not already cached
        cache_attr = f'_phonetic_mapping_{component_type}'
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, self.create_phonetic_mapping(glossary))
        
        phonetic_mapping = getattr(self, cache_attr)
        
        # Direct match
        if word.lower() in phonetic_mapping:
            return phonetic_mapping[word.lower()]
        
        # Phonetic match
        input_phonetic = self.get_phonetic_representation(word)
        
        phonetic_choices = [key for key, value in phonetic_mapping.items() 
                           if key != value and not key.isdigit() and len(key) > 2]
        
        if phonetic_choices:
            best_match_ph, score_ph, _ = process.extractOne(input_phonetic, phonetic_choices, scorer=fuzz.ratio)
            if score_ph >= self.phonetic_threshold:
                return phonetic_mapping[best_match_ph]
        
        # Fuzzy text match
        best_match_txt, score_txt, _ = process.extractOne(word.lower(), list(phonetic_mapping.keys()), scorer=fuzz.ratio)
        if score_txt >= self.fuzzy_threshold:
            return phonetic_mapping[best_match_txt]
        
        return None
    
    def match_inventory_phonetic(self, product_name: str) -> Optional[str]:
        """
        Match a product name against the inventory using phonetic and fuzzy matching.
        
        Args:
            product_name: Input product name
            
        Returns:
            Matched canonical product name or None if no match
        """
        # Direct match (fastest)
        processed_name = product_name.lower().strip()
        if processed_name in self.search_space:
            return self.search_space[processed_name]
        
        if not RAPIDFUZZ_AVAILABLE:
            return None
        
        choices = list(self.search_space.keys())
        
        # Token Set Ratio (Good for "dolo 650" vs "650" or "dolo")
        best_match_token, score_token, _ = process.extractOne(processed_name, choices, scorer=fuzz.token_set_ratio)
        if score_token >= self.fuzzy_threshold:
            return self.search_space[best_match_token]
        
        # Phonetic Match
        if not PHONEMIZER_AVAILABLE:
            return None
        
        # Create phonetic inventory if not already cached
        if self._phonetic_inventory is None:
            # Build a {phonetic_key: original_key} map
            phonetic_map = {}
            for key in choices:
                phonetic_map[self.get_phonetic_representation(key)] = key
            self._phonetic_inventory = phonetic_map
            self._phonetic_choices = list(phonetic_map.keys())
        
        input_phonetic = self.get_phonetic_representation(processed_name)
        best_match_ph, score_ph, _ = process.extractOne(input_phonetic, self._phonetic_choices, scorer=fuzz.ratio)
        
        if score_ph >= self.phonetic_threshold:
            original_key = self._phonetic_inventory[best_match_ph]
            return self.search_space[original_key]
        
        # Fallback: simple fuzzy ratio
        best_match_fuzz, score_fuzz, _ = process.extractOne(processed_name, choices, scorer=fuzz.ratio)
        if score_fuzz >= self.fuzzy_threshold:
            return self.search_space[best_match_fuzz]
        
        return None
    
    def convert_word_numbers_to_digits(self, text: str) -> str:
        """
        Convert word representations of numbers to digits.
        
        Args:
            text: Input text
            
        Returns:
            Text with word numbers converted to digits
        """
        # Handle specific patterns for medicine names
        medicine_patterns = [
            (r'\b(dolo\s+)?(six\s+hundr?ed\s+fifty|six\s+fifty)\b', 'dolo 650'),
            (r'\b(dolo\s+)?(six\s+five\s+zero|six\s+five\s+o)\b', 'dolo 650'),
            (r'\bdolo\s+six\b', 'dolo 650'),
        ]
        
        for pattern, replacement in medicine_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if not WORD2NUMBER_AVAILABLE:
            return text
        
        # Find all potential number words in the text
        number_word_pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s+(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|\s+)*\b'
        
        # Find all matches
        matches = re.finditer(number_word_pattern, text, re.IGNORECASE)
        
        # Process each match
        for match in matches:
            number_words = match.group(0)
            try:
                # Try to convert the words to a number
                number = w2n.word_to_num(number_words)
                # Replace the words with the digit
                text = text.replace(number_words, str(number))
            except ValueError:
                # If conversion fails, just continue
                continue
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for parsing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except spaces, dots (for 0.5), and commas (for item separation)
        text = re.sub(r"[^\w\s\.,]", "", text)
        
        # Ensure commas are properly spaced for splitting
        text = re.sub(r"\s*,\s*", ", ", text)
        
        # Convert word numbers to digits
        text = self.convert_word_numbers_to_digits(text)
        
        return text
    
    def process_text(self, text: str) -> str:
        """
        Process text by mapping quantities and units, leaving products alone.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        items = [x.strip() for x in text.split(",") if x.strip()]
        processed_items = []
        
        for item in items:
            words = item.split()
            processed_words = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if not clean_word:
                    continue
                
                # Check if it's a product first
                if self.match_inventory_phonetic(clean_word) is not None:
                    processed_words.append(word)  # Append the original word
                    continue
                
                # Check quantities
                qty_match = self.fuzzy_match_with_phonetics(clean_word, self.quantity_glossary, "quantity")
                if qty_match is not None:
                    processed_words.append(str(qty_match))
                    continue
                
                # Check units
                unit_match = self.fuzzy_match_with_phonetics(clean_word, self.unit_glossary, "unit")
                if unit_match is not None:
                    processed_words.append(unit_match)
                    continue
                
                # Append the original word
                processed_words.append(word)
            
            processed_items.append(" ".join(processed_words))
        
        return ", ".join(processed_items)
    
    def finalize_item(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build the final JSON for an item.
        
        Args:
            item_data: Dictionary containing item data
            
        Returns:
            Finalized item data or None if invalid
        """
        if item_data["qty"] is None or not item_data["product_words"]:
            return None
        
        product_key = " ".join(item_data["product_words"]).strip().lower()
        
        # Get valid quantities and units
        valid_quantities = set(self.quantity_glossary.values())
        valid_units = set(self.unit_glossary.keys()) | set(self.unit_glossary.values())
        
        # Clean product words
        clean_product_words = []
        for w in product_key.split():
            if w not in valid_quantities and w not in valid_units:
                clean_product_words.append(w)
        
        product_key = " ".join(clean_product_words)
        if not product_key:
            return None
        
        # Match product against inventory
        matched = self.match_inventory_phonetic(product_key)
        
        if matched:
            inv_info = self.inventory[matched]
            parsed_unit = inv_info["unit"]
            brand, price = None, None
            
            if inv_info.get("brands"):
                brand_choices = list(inv_info["brands"].keys())
                # Use token_set_ratio to find the specific brand
                best_brand, score, _ = process.extractOne(product_key, brand_choices, scorer=fuzz.token_set_ratio)
                
                if score >= self.fuzzy_threshold:
                    brand = best_brand
                    price = inv_info["brands"][brand]
                else:
                    brand, price = list(inv_info["brands"].items())[0]  # Default
            else:
                price = inv_info.get("price")
            
            return {
                "product_name": matched.title(),
                "brand_name": brand.title() if brand else None,
                "quantity": item_data["qty"],
                "unit": item_data["unit"] or parsed_unit,
                "price_per_unit": price,
                "total_price": price * item_data["qty"] if price else None,
                "confidence": 0.9  # High confidence for inventory match
            }
        else:
            return {
                "product_name": product_key.title(),
                "brand_name": None,
                "quantity": item_data["qty"],
                "unit": item_data["unit"] or "pcs",
                "price_per_unit": None,
                "total_price": None,
                "confidence": 0.5  # Lower confidence for unknown product
            }
    
    def parse_order(self, transcription: str) -> List[Dict[str, Any]]:
        """
        Parse a transcription into structured order items.
        
        Args:
            transcription: Input transcription
            
        Returns:
            List of parsed order items
        """
        results = []
        items = [x.strip() for x in transcription.split(",") if x.strip()]
        
        # Get valid quantities and units
        valid_quantities = set(self.quantity_glossary.values())
        valid_units = set(self.unit_glossary.keys()) | set(self.unit_glossary.values())
        
        for item_string in items:
            words = item_string.split()
            current_item = {"qty": None, "unit": None, "product_words": []}
            
            for word in words:
                clean_word = word.lower()
                
                is_qty = clean_word in valid_quantities
                is_unit = clean_word in valid_units
                is_known_product = self.match_inventory_phonetic(clean_word) is not None
                
                if is_qty:
                    if current_item["product_words"] and current_item["qty"] is None:
                        current_item["qty"] = float(clean_word)
                    else:
                        finalized = self.finalize_item(current_item)
                        if finalized:
                            results.append(finalized)
                        current_item = {"qty": float(clean_word), "unit": None, "product_words": []}
                
                elif is_unit:
                    if current_item["qty"] is not None:
                        if current_item["unit"] is None:
                            current_item["unit"] = clean_word
                        else:
                            current_item["product_words"].append(word)
                    else:
                        current_item["product_words"].append(word)
                
                elif is_known_product:
                    if current_item["product_words"]:
                        # Check if this new product is different from what we have
                        if current_item["qty"] is not None:
                            finalized = self.finalize_item(current_item)
                            if finalized:
                                results.append(finalized)
                            current_item = {"qty": None, "unit": None, "product_words": [word]}
                        else:
                            current_item["product_words"].append(word)
                    else:
                        current_item["product_words"].append(word)
                
                else:
                    current_item["product_words"].append(word)
            
            # After the loop finishes, finalize the last item
            finalized = self.finalize_item(current_item)
            if finalized:
                results.append(finalized)
        
        return results
    
    def parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse text into structured data.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing parsed data
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Process text
        processed_text = self.process_text(preprocessed_text)
        
        # Parse order
        parsed_order = self.parse_order(processed_text)
        
        # Calculate total price
        total_price = sum(item.get('total_price', 0) for item in parsed_order if item.get('total_price'))
        
        return {
            "original_transcription": text,
            "processed_text": processed_text,
            "parsed_order": parsed_order,
            "total_price": total_price
        }
    
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
            
            if "inventory" in data:
                self.inventory = data["inventory"]
            
            # Rebuild search space
            self.search_space = self._build_search_space()
            
            # Clear caches
            self._phonetic_cache = {}
            self._phonetic_inventory = None
            self._phonetic_choices = None
            
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
                "product_glossary": self.product_glossary,
                "inventory": self.inventory
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
    
    parser = argparse.ArgumentParser(description="Enhanced Slot Parser")
    parser.add_argument("--glossary-file", type=str, default=None,
                        help="JSON file containing glossaries")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to parse")
    parser.add_argument("--file", type=str, default=None,
                        help="File containing text to parse")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--fuzzy-threshold", type=int, default=70,
                        help="Threshold for fuzzy matching")
    parser.add_argument("--phonetic-threshold", type=int, default=70,
                        help="Threshold for phonetic matching")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "language": args.language,
        "fuzzy_threshold": args.fuzzy_threshold,
        "phonetic_threshold": args.phonetic_threshold
    }
    
    # Create slot parser
    slot_parser = SlotParser(config)
    
    # Load glossaries if specified
    if args.glossary_file:
        slot_parser.load_glossaries_from_file(args.glossary_file)
    
    # Parse text
    if args.text:
        result = slot_parser.parse_text(args.text)
        print(json.dumps(result, indent=2))
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        result = slot_parser.parse_text(text)
        print(json.dumps(result, indent=2))
    else:
        print("Please provide text to parse using --text or --file")




















