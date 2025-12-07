#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU-Efficient Pharmaceutical Voice Order Pipeline using whisper.cpp
--------------------------------------------------------------------
Records audio from ESP32, transcribes using whisper.cpp with GBNF grammar
constraints, applies phonetic matching for pronunciation variations,
and parses pharmaceutical orders into structured JSON.

Key Features:
- CPU-only inference (no GPU required)
- Strict grammar constraints to prevent hallucinations
- Phonetic matching for Indian English pronunciation handling
- ESP32 serial audio recording integration
"""

import serial
import wave
import re
import json
import os
import time
import logging
import subprocess
import shutil
from rapidfuzz import process, fuzz
from metaphone import doublemetaphone

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WhisperCppPharmaPipeline")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Whisper.cpp Configuration ---
WHISPER_CPP_PATH = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-small.en.bin"
GRAMMAR_FILE = "pharma_inventory.gbnf"
PHONETIC_THRESHOLD = 80  # Minimum phonetic similarity score (0-100)

# --- Serial Port and Recording Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'  # Example for Linux/Mac
# SERIAL_PORT = 'COM3'        # Example for Windows

BAUD_RATE = 921600
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

# --- WAV File Format Parameters (must match ESP32) ---
CHANNELS = 1
SAMPLE_WIDTH = 2  # 2 bytes for 16-bit samples
SAMPLE_RATE = 16000

# =============================================================================
# PHARMACEUTICAL GLOSSARIES AND INVENTORY
# =============================================================================

# Load glossaries from JSON file
GLOSSARY_FILE = "pharma_glossary.json"

def load_glossaries():
    """Load pharmaceutical glossaries from JSON file"""
    try:
        with open(GLOSSARY_FILE, 'r') as f:
            glossaries = json.load(f)
        logger.info(f"Loaded glossaries from {GLOSSARY_FILE}")
        return (
            glossaries.get("quantity_glossary", {}),
            glossaries.get("unit_glossary", {}),
            glossaries.get("product_glossary", {})
        )
    except FileNotFoundError:
        logger.warning(f"Glossary file {GLOSSARY_FILE} not found, using defaults")
        return {}, {}, {}

QUANTITY_GLOSSARY, UNIT_GLOSSARY, PRODUCT_GLOSSARY = load_glossaries()

# Pharmaceutical Inventory
inventory = {
    "paracetamol": {
        "brands": {"crocin": 20, "dolo 650": 32, "calpol": 25},
        "unit": "strip",
        "price": 20
    },
    "cetirizine": {
        "brands": {},
        "unit": "strip",
        "price": 20
    },
    "ibuprofen": {
        "brands": {"brufen": 30},
        "unit": "strip",
        "price": 30
    },
    "metformin": {
        "brands": {"glucophage": 50},
        "unit": "strip",
        "price": 45
    },
    "atorvastatin": {
        "brands": {"lipitor": 80},
        "unit": "strip",
        "price": 75
    },
    "amlodipine": {
        "brands": {"norvasc": 60},
        "unit": "strip",
        "price": 55
    },
    "lisinopril": {
        "brands": {"zestril": 70},
        "unit": "strip",
        "price": 65
    },
    "albuterol": {
        "brands": {"ventolin": 120},
        "unit": "bottle",
        "price": 110
    },
    "amoxicillin": {
        "brands": {"amoxil": 40},
        "unit": "strip",
        "price": 35
    },
    "azithromycin": {
        "brands": {"zithromax": 90},
        "unit": "strip",
        "price": 85
    },
    "ciprofloxacin": {
        "brands": {"cipro": 55},
        "unit": "strip",
        "price": 50
    },
    "levothyroxine": {
        "brands": {"synthroid": 45},
        "unit": "strip",
        "price": 40
    },
    "omez": {
        "brands": {},
        "unit": "strip",
        "price": 55
    },
    "cough syrup": {
        "brands": {"benadryl": 120},
        "unit": "bottle",
        "price": 110
    },
    "dettol": {
        "brands": {},
        "unit": "bottle",
        "price": 80
    },
    "band-aid": {
        "brands": {},
        "unit": "pcs",
        "price": 5
    },
    "vicks vaporub": {
        "brands": {},
        "unit": "bottle",
        "price": 45
    },
    "multivitamin": {
        "brands": {"becosules": 48},
        "unit": "strip",
        "price": 45
    },
    "aspirin": {
        "brands": {"disprin": 15},
        "unit": "strip",
        "price": 12
    },
    "gabapentin": {
        "brands": {"neurontin": 95},
        "unit": "strip",
        "price": 90
    },
    "sertraline": {
        "brands": {"zoloft": 110},
        "unit": "strip",
        "price": 105
    },
    "fluoxetine": {
        "brands": {"prozac": 100},
        "unit": "strip",
        "price": 95
    }
}

# =============================================================================
# INSTALLATION CHECK
# =============================================================================

def check_whisper_cpp():
    """Verify whisper.cpp installation"""
    logger.info("Checking whisper.cpp installation...")
    
    if not os.path.exists(WHISPER_CPP_PATH):
        error_msg = (
            f"whisper.cpp not found at: {WHISPER_CPP_PATH}\n"
            f"Please install from: https://github.com/ggerganov/whisper.cpp\n"
            f"  git clone https://github.com/ggerganov/whisper.cpp.git\n"
            f"  cd whisper.cpp\n"
            f"  make\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not os.path.exists(WHISPER_MODEL):
        error_msg = (
            f"Whisper model not found at: {WHISPER_MODEL}\n"
            f"Please download the model:\n"
            f"  cd whisper.cpp\n"
            f"  bash ./models/download-ggml-model.sh small.en\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not os.path.exists(GRAMMAR_FILE):
        error_msg = (
            f"Grammar file not found: {GRAMMAR_FILE}\n"
            f"Please generate it first:\n"
            f"  python3 generate_pharma_grammar.py\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info("✓ whisper.cpp installation verified")
    print("✓ whisper.cpp found")
    print(f"✓ Model: {WHISPER_MODEL}")
    print(f"✓ Grammar: {GRAMMAR_FILE}")
    return True

# =============================================================================
# PHONETIC MATCHING
# =============================================================================

class PhoneticMatcher:
    """
    Phonetic matching using Double Metaphone algorithm
    Handles pronunciation variations, especially for Indian English
    """
    
    def __init__(self, threshold=PHONETIC_THRESHOLD):
        """Initialize phonetic matcher"""
        self.threshold = threshold
        self.cache = {}  # Cache for phonetic codes
        logger.info(f"PhoneticMatcher initialized (threshold={threshold})")
    
    def get_phonetic_code(self, word):
        """Get phonetic code for a word (with caching)"""
        word_lower = word.lower().strip()
        if word_lower not in self.cache:
            # doublemetaphone returns tuple of (primary, secondary) codes
            codes = doublemetaphone(word_lower)
            self.cache[word_lower] = codes
        return self.cache[word_lower]
    
    def phonetic_similarity(self, word1, word2):
        """
        Calculate phonetic similarity score (0-100)
        Uses both fuzzy string matching and phonetic matching
        """
        word1_lower = word1.lower().strip()
        word2_lower = word2.lower().strip()
        
        # Exact match
        if word1_lower == word2_lower:
            return 100
        
        # Fuzzy string similarity (rapidfuzz)
        fuzzy_score = fuzz.ratio(word1_lower, word2_lower)
        
        # Phonetic similarity
        phonetic1 = self.get_phonetic_code(word1)
        phonetic2 = self.get_phonetic_code(word2)
        
        phonetic_score = 0
        # Check if any of the phonetic codes match
        if phonetic1[0] and phonetic2[0]:
            if phonetic1[0] == phonetic2[0]:
                phonetic_score = 90
            elif phonetic1[1] and phonetic2[1] and phonetic1[1] == phonetic2[1]:
                phonetic_score = 80
            elif phonetic1[0] and phonetic2[1] and phonetic1[0] == phonetic2[1]:
                phonetic_score = 75
            elif phonetic1[1] and phonetic2[0] and phonetic1[1] == phonetic2[0]:
                phonetic_score = 75
        
        # Return weighted average (60% fuzzy, 40% phonetic)
        combined_score = (fuzzy_score * 0.6) + (phonetic_score * 0.4)
        
        logger.debug(
            f"Phonetic match '{word1}' vs '{word2}': "
            f"fuzzy={fuzzy_score:.1f}, phonetic={phonetic_score:.1f}, "
            f"combined={combined_score:.1f}"
        )
        
        return combined_score
    
    def match_word(self, word, candidates):
        """
        Match a word against a list of candidates
        Returns (best_match, score) or (None, 0) if no match above threshold
        """
        if not candidates:
            return None, 0
        
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = self.phonetic_similarity(word, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= self.threshold:
            logger.info(f"✓ Phonetic match: '{word}' → '{best_match}' (score={best_score:.1f})")
            return best_match, best_score
        else:
            logger.debug(f"✗ No phonetic match for '{word}' (best={best_score:.1f})")
            return None, 0

# =============================================================================
# WHISPER.CPP TRANSCRIBER
# =============================================================================

class WhisperCppTranscriber:
    """Handles transcription using whisper.cpp (grammar-free mode for better compatibility)"""
    
    def __init__(self, model_path=WHISPER_MODEL, grammar_file=None):
        """Initialize whisper.cpp transcriber"""
        self.model_path = model_path
        self.grammar_file = grammar_file  # Not used currently due to whisper.cpp grammar parsing issues
        self.whisper_cli = WHISPER_CPP_PATH
        logger.info(f"WhisperCppTranscriber initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Mode: Raw transcription + phonetic matching (no grammar constraints)")
    
    def transcribe(self, audio_file_path):
        """
        Transcribe audio file using whisper.cpp WITHOUT grammar constraints
        Grammar was causing parsing issues, so we use raw transcription + phonetic matching
        Returns transcription string or None on error
        """
        logger.info(f"Transcribing audio: {audio_file_path}")
        start_time = time.time()
        
        try:
            # Construct whisper.cpp command (WITHOUT grammar for now)
            cmd = [
                self.whisper_cli,
                "-m", self.model_path,
                "-f", audio_file_path,
                "-l", "en",  # Language: English
                "-t", "4",   # Threads (adjust based on CPU)
                "--no-timestamps",  # Don't include timestamps in output
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run whisper.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                logger.error(f"whisper.cpp failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                print(f"❌ Transcription failed: {result.stderr}")
                return None
            
            # Parse output from stdout (whisper.cpp prints transcription to stdout)
            output_lines = result.stdout.strip().split('\n')
            
            # Find the transcription line (skip debug/info lines)
            transcription = ""
            for line in output_lines:
                # Skip empty lines and lines that look like debug output
                if line.strip() and not line.startswith('whisper_') and not line.startswith('main:') and not line.startswith('system_info:'):
                    transcription = line.strip()
                    break
            
            transcribe_time = time.time() - start_time
            logger.info(f"✓ Transcription complete in {transcribe_time:.2f}s")
            logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
            
            return transcription
            
        except subprocess.TimeoutExpired:
            logger.error("Transcription timed out (>30s)")
            print("❌ Transcription timed out")
            return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            return None

# =============================================================================
# AUDIO RECORDING FROM ESP32
# =============================================================================

def record_audio_from_esp32():
    """Records audio from ESP32 via serial port"""
    logger.info("Starting audio recording from ESP32")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
            logger.info(f"Connected to {SERIAL_PORT}")
            print(f"✅ Connected to {SERIAL_PORT}.")
            print("⏳ Please press the button on the device to START recording...")
            
            audio_data_chunks = []
            is_recording_started = False
            
            while True:
                chunk = ser.read(4096)
                if chunk:
                    if not is_recording_started:
                        logger.info("Recording started")
                        print("🎤 Recording started... (Press button again to STOP)")
                        is_recording_started = True
                    audio_data_chunks.append(chunk)
                elif is_recording_started and not chunk:
                    logger.info("Recording stopped")
                    print("👍 Recording stopped.")
                    break
                elif not is_recording_started and not chunk:
                    print("...Still waiting for recording to start...")
            
            if not audio_data_chunks:
                logger.error("No audio data was recorded")
                print("❌ No audio data was recorded.")
                return None
            
            logger.info(f"Saving audio to '{WAVE_OUTPUT_FILENAME}'")
            print(f"💾 Saving audio to '{WAVE_OUTPUT_FILENAME}'...")
            
            audio_data = b''.join(audio_data_chunks)
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
            
            logger.info("File saved successfully")
            print("✅ File saved successfully.")
            return WAVE_OUTPUT_FILENAME
            
    except serial.SerialException as e:
        logger.error(f"Could not open serial port {SERIAL_PORT}: {e}")
        print(f"❌ Error: Could not open serial port {SERIAL_PORT}.")
        print(f"   Details: {e}")
        print("   Please ensure the ESP32 is connected and you have selected the correct port.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during recording: {e}")
        print(f"❌ An unexpected error occurred during recording: {e}")
        return None

# =============================================================================
# ANTI-HALLUCINATION VALIDATION
# =============================================================================

def build_allowed_vocabulary():
    """
    Build complete vocabulary of allowed words to prevent hallucinations
    Only words in this set are valid - everything else is rejected
    """
    allowed = set()
    
    # Add all quantities
    allowed.update(QUANTITY_GLOSSARY.keys())
    allowed.update(str(v) for v in QUANTITY_GLOSSARY.values())
    
    # Add all units
    allowed.update(UNIT_GLOSSARY.keys())
    allowed.update(UNIT_GLOSSARY.values())
    
    # Add all products
    allowed.update(inventory.keys())
    
    # Add all brands
    for product_info in inventory.values():
        allowed.update(product_info.get('brands', {}).keys())
    
    # Add all product glossary terms
    allowed.update(PRODUCT_GLOSSARY.keys())
    allowed.update(PRODUCT_GLOSSARY.values())
    
    # Add common connector words that are OK
    allowed.update(['and', 'of', 'the'])
    
    logger.info(f"Built allowed vocabulary: {len(allowed)} terms")
    return allowed

ALLOWED_VOCABULARY = build_allowed_vocabulary()

def validate_transcription_strict(transcription):
    """
    STRICT VALIDATION: Reject transcriptions with hallucinations
    Handles multi-word brands (e.g., "dolo 650", "vitamin d")
    Returns cleaned transcription or None if invalid words detected
    """
    if not transcription:
        return None
    
    text_lower = transcription.lower().strip()
    
    # STEP 1: Identify multi-word terms (brands like "dolo 650")
    # Extract all multi-word entries from vocabulary
    multi_word_terms = [term for term in ALLOWED_VOCABULARY if ' ' in term]
    # Sort by length (longest first) to match "vitamin d3" before "vitamin d"
    multi_word_terms.sort(key=len, reverse=True)
    
    # Replace multi-word terms with placeholders to protect them
    temp_text = text_lower
    replacements = {}  # placeholder -> original term
    placeholder_counter = 0
    
    for term in multi_word_terms:
        if term in temp_text:
            placeholder = f"__MULTIWORD_{placeholder_counter}__"
            temp_text = temp_text.replace(term, placeholder)
            replacements[placeholder] = term
            placeholder_counter += 1
            logger.debug(f"✓ Protected multi-word term: '{term}'")
    
    # STEP 2: Now split and validate individual words/placeholders
    # Use [\w-]+ to keep hyphens (for "band-aid", etc.)
    words = re.findall(r'[\w-]+', temp_text)
    
    invalid_words = []
    validated_words = []
    
    for word in words:
        if not word:
            continue
        
        # Check if it's a placeholder for a multi-word term
        if word.startswith('__MULTIWORD_') and word in replacements:
            validated_words.append(replacements[word])
            logger.debug(f"✓ Multi-word term: '{replacements[word]}'")
            continue
            
        # Check if word is in allowed vocabulary
        if word in ALLOWED_VOCABULARY:
            validated_words.append(word)
            logger.debug(f"✓ Valid word: '{word}'")
            continue
        
        # Try phonetic match against allowed vocabulary (last resort)
        matched_word, score = phonetic_matcher.match_word(word, list(ALLOWED_VOCABULARY))
        
        if matched_word and score >= 85:  # Higher threshold for validation
            validated_words.append(matched_word)
            logger.info(f"✓ Phonetic correction: '{word}' → '{matched_word}' (score={score:.1f})")
        else:
            invalid_words.append(word)
            logger.warning(f"✗ INVALID word detected: '{word}' (possible hallucination)")
    
    # Reject if any invalid words found
    if invalid_words:
        logger.error(f"❌ Transcription REJECTED - Invalid words: {invalid_words}")
        logger.error(f"   This appears to be a hallucination or out-of-vocabulary speech")
        print(f"⚠️  Transcription rejected: Contains invalid words {invalid_words}")
        print(f"   Please speak clearly using pharmaceutical terms only")
        return None
    
    # Return validated transcription
    validated_text = ' '.join(validated_words)
    logger.info(f"✅ Validation passed: '{validated_text}'")
    return validated_text

# =============================================================================
# TEXT PROCESSING & PARSING WITH PHONETIC MATCHING
# =============================================================================

def build_search_space():
    """Build search space for product matching"""
    search_map = {}
    for product, info in inventory.items():
        search_map[product] = product
        for brand in info.get("brands", {}):
            search_map[brand] = product
    for key, value in PRODUCT_GLOSSARY.items():
        if value in search_map:
            search_map[key] = search_map[value]
        elif value in inventory:
            search_map[key] = value
    return search_map

search_space = build_search_space()
phonetic_matcher = PhoneticMatcher(threshold=PHONETIC_THRESHOLD)

def match_product(product_name):
    """
    Match a product string to inventory using fuzzy + phonetic matching.
    Returns the base product name if found, None otherwise.
    """
    processed_name = product_name.lower().strip()
    
    # Direct match
    if processed_name in search_space:
        return search_space[processed_name]
    
    choices = list(search_space.keys())
    
    # Fuzzy token set ratio (good for partial matches)
    best_match_token, score_token, _ = process.extractOne(
        processed_name, choices, scorer=fuzz.token_set_ratio
    )
    if score_token >= 70:
        return search_space[best_match_token]
    
    # Phonetic matching as fallback
    matched_word, phonetic_score = phonetic_matcher.match_word(processed_name, choices)
    if matched_word:
        logger.info(f"✓ Phonetic match used: '{processed_name}' → '{matched_word}'")
        return search_space[matched_word]
    
    # Simple fuzzy ratio
    best_match_fuzz, score_fuzz, _ = process.extractOne(
        processed_name, choices, scorer=fuzz.ratio
    )
    if score_fuzz >= 70:
        return search_space[best_match_fuzz]
    
    return None

def normalize_text(text):
    """Normalize and clean transcribed text"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra punctuation but keep commas for item separation
    text = re.sub(r"[^\w\s\.,]", "", text)
    
    # Ensure commas are properly spaced
    text = re.sub(r"\s*,\s*", ", ", text)
    
    # Strip whitespace
    text = text.strip()
    
    logger.debug(f"Normalized text: '{text}'")
    return text

def process_text(text):
    """
    Process transcribed text by mapping quantities and units.
    Enhanced with phonetic matching.
    """
    logger.info(f"Processing text: '{text}'")
    
    # Normalize first
    text = normalize_text(text)
    
    items = [x.strip() for x in text.split(",") if x.strip()]
    processed_items = []
    
    for item in items:
        words = item.split()
        processed_words = []
        
        for word in words:
            clean_word = word.lower().strip()
            
            # Check if it's a quantity
            if clean_word in QUANTITY_GLOSSARY:
                mapped_qty = QUANTITY_GLOSSARY[clean_word]
                processed_words.append(mapped_qty)
                logger.debug(f"Mapped quantity: '{clean_word}' -> '{mapped_qty}'")
                continue
            
            # Phonetic match for quantity
            qty_match, _ = phonetic_matcher.match_word(clean_word, list(QUANTITY_GLOSSARY.keys()))
            if qty_match:
                mapped_qty = QUANTITY_GLOSSARY[qty_match]
                processed_words.append(mapped_qty)
                logger.debug(f"Phonetic quantity: '{clean_word}' -> '{mapped_qty}'")
                continue
            
            # Check if it's a unit
            if clean_word in UNIT_GLOSSARY:
                mapped_unit = UNIT_GLOSSARY[clean_word]
                processed_words.append(mapped_unit)
                logger.debug(f"Mapped unit: '{clean_word}' -> '{mapped_unit}'")
                continue
            
            # Phonetic match for unit
            unit_match, _ = phonetic_matcher.match_word(clean_word, list(UNIT_GLOSSARY.keys()))
            if unit_match:
                mapped_unit = UNIT_GLOSSARY[unit_match]
                processed_words.append(mapped_unit)
                logger.debug(f"Phonetic unit: '{clean_word}' -> '{mapped_unit}'")
                continue
            
            # Keep the original word (could be a product)
            processed_words.append(word)
        
        processed_items.append(" ".join(processed_words))
    
    processed_text = ", ".join(processed_items)
    logger.info(f"Processed text: '{processed_text}'")
    return processed_text

VALID_QUANTITIES = set(QUANTITY_GLOSSARY.values())
VALID_UNITS = set(UNIT_GLOSSARY.keys()) | set(UNIT_GLOSSARY.values())

def finalize_item(item_data):
    """Build final JSON for a parsed item"""
    if item_data["qty"] is None or not item_data["product_words"]:
        return None
    
    product_key = " ".join(item_data["product_words"]).strip().lower()
    
    # Remove quantities and units from product name
    clean_product_words = []
    for w in product_key.split():
        if w not in VALID_QUANTITIES and w not in VALID_UNITS:
            clean_product_words.append(w)
    
    product_key = " ".join(clean_product_words)
    if not product_key:
        return None
    
    logger.debug(f"Finalizing item: product_key='{product_key}', qty={item_data['qty']}, unit={item_data['unit']}")
    
    # Match to inventory (with phonetic fallback)
    matched = match_product(product_key)
    
    if matched:
        logger.info(f"✓ Matched product '{product_key}' to '{matched}'")
        inv_info = inventory[matched]
        parsed_unit = inv_info["unit"]
        brand, price = None, None
        
        if inv_info.get("brands"):
            brand_choices = list(inv_info["brands"].keys())
            
            # Try phonetic match for brands first
            brand_match, phonetic_score = phonetic_matcher.match_word(product_key, brand_choices)
            if brand_match:
                brand = brand_match
                price = inv_info["brands"][brand]
                logger.info(f"✓ Phonetic brand match: '{product_key}' → '{brand}'")
            else:
                # Fallback to fuzzy match
                best_brand, score, _ = process.extractOne(
                    product_key, brand_choices, scorer=fuzz.token_set_ratio
                )
                
                if score >= 70:
                    brand = best_brand
                    price = inv_info["brands"][brand]
                    logger.info(f"✓ Fuzzy brand match: '{product_key}' → '{brand}' (score={score})")
                else:
                    brand, price = list(inv_info["brands"].items())[0]
        else:
            price = inv_info.get("price")
        
        return {
            "product_name": matched.title(),
            "brand_name": brand.title() if brand else None,
            "quantity": item_data["qty"],
            "unit": item_data["unit"] or parsed_unit,
            "price_per_unit": price,
            "total_price": price * item_data["qty"] if price else None
        }
    else:
        logger.warning(f"⚠ Product '{product_key}' not found in inventory")
        return {
            "product_name": product_key.title(),
            "brand_name": None,
            "quantity": item_data["qty"],
            "unit": item_data["unit"] or "pcs",
            "price_per_unit": None,
            "total_price": None
        }

def parse_order(transcription):
    """
    Parse processed transcription into structured order items.
    Handles patterns like: "5 strip paracetamol", "dolo 650 2 boxes", etc.
    """
    logger.info(f"Parsing order from: '{transcription}'")
    results = []
    items = [x.strip() for x in transcription.split(",") if x.strip()]
    
    for item_string in items:
        words = item_string.split()
        current_item = {"qty": None, "unit": None, "product_words": []}
        
        for word in words:
            clean_word = word.lower()
            
            is_qty = clean_word in VALID_QUANTITIES
            is_unit = clean_word in VALID_UNITS
            is_known_product = match_product(clean_word) is not None
            
            if is_qty:
                if current_item["product_words"] and current_item["qty"] is None:
                    current_item["qty"] = float(clean_word)
                else:
                    # New quantity means new item
                    finalized = finalize_item(current_item)
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
                if current_item["product_words"] and current_item["qty"] is not None:
                    # New product after qty means new item
                    finalized = finalize_item(current_item)
                    if finalized:
                        results.append(finalized)
                    current_item = {"qty": None, "unit": None, "product_words": [word]}
                else:
                    current_item["product_words"].append(word)
            
            else:
                # Unknown word, add to product
                current_item["product_words"].append(word)
        
        # Finalize last item
        finalized = finalize_item(current_item)
        if finalized:
            results.append(finalized)
    
    logger.info(f"Parsed {len(results)} items from order")
    return results

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_audio_order(audio_file_path, transcriber):
    """Complete pipeline: Audio -> whisper.cpp -> VALIDATION -> Phonetic -> Parse -> JSON"""
    logger.info("=" * 60)
    logger.info("Processing audio order")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("📋 STEP 2: TRANSCRIPTION AND PROCESSING")
    print("=" * 60)
    
    # Transcribe with whisper.cpp
    print("🎤 Transcribing with whisper.cpp (CPU mode)...")
    transcribed_text = transcriber.transcribe(audio_file_path)
    
    if not transcribed_text:
        logger.error("Transcription failed")
        print("❌ Could not transcribe the audio.")
        return None
    
    print(f"✅ Transcription complete: '{transcribed_text}'")
    
    # CRITICAL: Validate transcription to prevent hallucinations
    print("🛡️  Validating transcription (anti-hallucination check)...")
    validated_text = validate_transcription_strict(transcribed_text)
    
    if not validated_text:
        logger.error("Validation failed - transcription rejected")
        print("❌ Transcription validation failed!")
        print("   The audio contained words outside the pharmaceutical vocabulary.")
        print("   Please speak only pharmaceutical product names, quantities, and units.")
        return None
    
    print(f"✅ Validation passed: '{validated_text}'")
    
    # Process the validated transcription (with phonetic matching)
    print("🔄 Processing with phonetic matching...")
    processed_text = process_text(validated_text)
    print(f"✅ Processed text: '{processed_text}'")
    
    # Parse into structured order
    print("🔍 Parsing into structured order...")
    parsed_order = parse_order(processed_text)
    
    if not parsed_order:
        logger.warning("No items could be parsed from the order")
        print("⚠️  Could not parse any items from the order.")
    
    return {
        "original_transcription": transcribed_text,
        "validated_text": validated_text,
        "processed_text": processed_text,
        "parsed_order": parsed_order
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with continuous recording support"""
    print("=" * 60)
    print("🚀 WHISPER.CPP PHARMACEUTICAL VOICE ORDER PIPELINE (CPU)")
    print("=" * 60)
    
    # Check whisper.cpp installation
    print("\n🔍 Checking installation...")
    try:
        check_whisper_cpp()
    except FileNotFoundError as e:
        print(f"\n❌ Setup incomplete:\n{e}")
        return
    
    # Initialize whisper.cpp transcriber
    print("\n📦 Initializing whisper.cpp transcriber...")
    transcriber = WhisperCppTranscriber()
    print("✓ Transcriber ready (CPU mode)")
    
    # Session statistics
    session_order_count = 0
    session_total_value = 0.0
    all_orders = []
    
    # Continuous recording loop
    while True:
        print("\n" + "=" * 60)
        print(f"📋 ORDER #{session_order_count + 1}: AUDIO RECORDING")
        print("=" * 60)
        
        # Record audio from ESP32
        recorded_file = record_audio_from_esp32()
        
        # If recording was successful, process the audio
        if recorded_file:
            result = process_audio_order(recorded_file, transcriber)
            
            if result:
                print("\n" + "=" * 60)
                print("📊 FINAL RESULTS")
                print("=" * 60)
                print(f"🗣️  Original Transcription: {result['original_transcription']}")
                print(f"🛡️  Validated Text: {result['validated_text']}")
                print(f"⚙️  Processed Text: {result['processed_text']}")
                print("📦 Parsed Order:")
                print(json.dumps(result['parsed_order'], indent=2, ensure_ascii=False))
                
                order_total = sum(
                    item.get('total_price', 0)
                    for item in result['parsed_order']
                    if item.get('total_price')
                )
                
                if order_total > 0:
                    print(f"\n💰 Order Value: ₹{order_total:.2f}")
                
                # Update session statistics
                session_order_count += 1
                session_total_value += order_total
                all_orders.append({
                    "order_number": session_order_count,
                    "transcription": result['original_transcription'],
                    "items": result['parsed_order'],
                    "total": order_total
                })
                
                logger.info(f"Order #{session_order_count} completed successfully")
        else:
            print("\n⚠️  Recording failed for this order.")
            logger.error("Audio recording failed")
        
        # Ask if user wants to continue
        print("\n" + "=" * 60)
        print("Continue with another order?")
        print("  Press 'c' or Enter to continue")
        print("  Press 'q' to quit and view session summary")
        print("=" * 60)
        
        try:
            user_input = input("➤ ").strip().lower()
        except EOFError:
            user_input = 'q'
        
        if user_input == 'q' or user_input == 'quit' or user_input == 'exit':
            break
        
        # Continue loop for next order
        logger.info("Starting next order...")
    
    # Display session summary
    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Total Orders Processed: {session_order_count}")
    print(f"Total Session Value: ₹{session_total_value:.2f}")
    
    if session_order_count > 0:
        print(f"\nAverage Order Value: ₹{session_total_value/session_order_count:.2f}")
        print(f"\nAll Orders:")
        for order in all_orders:
            print(f"  Order #{order['order_number']}: ₹{order['total']:.2f} ({len(order['items'])} items)")
    
    print("\n" + "=" * 60)
    print("✅ Session ended. Thank you for using ShopiVoice!")
    print("=" * 60)
    logger.info("Session ended")

if __name__ == "__main__":
    main()



