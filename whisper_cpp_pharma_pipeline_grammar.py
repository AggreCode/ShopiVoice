#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grammar-Constrained Pharmaceutical Voice Order Pipeline using whisper.cpp
--------------------------------------------------------------------------
Focus: GRAMMAR-BASED TRANSCRIPTION with minimal post-processing

Records audio from ESP32, transcribes using whisper.cpp with inline GBNF 
grammar constraints to prevent hallucinations at the source.

Key Features:
- Inline GBNF grammar passed directly to whisper.cpp
- Grammar enforces: quantity + unit + product patterns
- CPU-only inference (no GPU required)
- Minimal phonetic post-processing (only for known variations)
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
from rapidfuzz import process, fuzz
from metaphone import doublemetaphone

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WhisperGrammarPipeline")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Whisper.cpp Configuration ---
WHISPER_CPP_PATH = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./whisper.cpp/models/ggml-small.en.bin"
PHONETIC_THRESHOLD = 85  # High threshold for minimal corrections

# --- Serial Port and Recording Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 921600
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

# --- WAV File Format Parameters ---
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000

# =============================================================================
# PHARMACEUTICAL GLOSSARIES AND INVENTORY
# =============================================================================

# Load glossaries
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
    "amoxicillin": {
        "brands": {"amoxil": 40},
        "unit": "strip",
        "price": 40
    },
    "azithromycin": {
        "brands": {"azithral": 90},
        "unit": "strip",
        "price": 90
    },
    "omeprazole": {
        "brands": {},
        "unit": "strip",
        "price": 35
    },
    "metformin": {
        "brands": {},
        "unit": "strip",
        "price": 25
    },
    "amlodipine": {
        "brands": {},
        "unit": "strip",
        "price": 30
    },
    "atorvastatin": {
        "brands": {},
        "unit": "strip",
        "price": 35
    },
    "losartan": {
        "brands": {},
        "unit": "strip",
        "price": 40
    },
    "levothyroxine": {
        "brands": {"synthroid": 50, "eltroxin": 45},
        "unit": "strip",
        "price": 45
    },
    "aspirin": {
        "brands": {},
        "unit": "strip",
        "price": 10
    },
    "fluoxetine": {
        "brands": {"prozac": 80},
        "unit": "strip",
        "price": 80
    },
    "sertraline": {
        "brands": {"zoloft": 85},
        "unit": "strip",
        "price": 85
    },
    "multivitamin": {
        "brands": {"centrum": 150},
        "unit": "bottle",
        "price": 150
    },
    "vitamin d": {
        "brands": {},
        "unit": "strip",
        "price": 60
    },
    "calcium": {
        "brands": {"shelcal": 120},
        "unit": "strip",
        "price": 120
    },
    "dettol": {
        "brands": {},
        "unit": "bottle",
        "price": 80
    },
    "band-aid": {
        "brands": {},
        "unit": "box",
        "price": 50
    },
    "cough syrup": {
        "brands": {},
        "unit": "bottle",
        "price": 100
    }
}

# =============================================================================
# INLINE GBNF GRAMMAR GENERATION
# =============================================================================

def escape_gbnf_string(s):
    """Escape special characters for GBNF grammar"""
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    return s

def generate_inventory_grammar():
    """
    Generates inline GBNF grammar string from inventory.
    Uses the 'Leading Space' fix: root ::= [ ]? command [ ]?
    
    This is the CRITICAL function that prevents hallucinations.
    """
    # 1. Collect all products (sorted by length descending for proper matching)
    products = []
    
    # Add all product names
    for product_name in inventory.keys():
        products.append(product_name.lower())
    
    # Add all brands (WITH SPACES - important for multi-word brands)
    for product_info in inventory.values():
        for brand_name in product_info.get('brands', {}).keys():
            # Add brand as-is
            products.append(brand_name.lower())
            # Also add without spaces for matching
            products.append(brand_name.lower().replace(" ", ""))
    
    # Add product glossary terms
    for alias, product in PRODUCT_GLOSSARY.items():
        products.append(alias.lower())
    
    # Remove duplicates and sort by length (descending)
    products = sorted(set(products), key=len, reverse=True)
    
    # 2. Build product rules
    product_rules = " | ".join([f'"{escape_gbnf_string(p)}"' for p in products])
    
    # 3. Collect all units
    units = []
    for unit in set(UNIT_GLOSSARY.keys()) | set(UNIT_GLOSSARY.values()):
        units.append(unit.lower())
    # Add standard units
    units.extend(["kg", "g", "mg", "ml", "liter", "litre", "packet", "pcs", "piece", 
                  "box", "strip", "strips", "bottle", "bottles", "tablet", "tablets", "tab"])
    units = sorted(set(units))
    unit_rules = " | ".join([f'"{u}"' for u in units])
    
    # 4. Build quantity rules
    # Numeric quantities 0-100
    numeric_rules = [f'"{i}"' for i in range(0, 101)]
    
    # Word quantities from glossary
    word_quantities = list(QUANTITY_GLOSSARY.keys())
    word_rules = [f'"{word}"' for word in word_quantities]
    
    quantity_rules = " | ".join(numeric_rules + word_rules)
    
    # 5. Define the Grammar with Leading Space fix
    # CRITICAL: "root ::= [ ]? order [ ]?" allows leading/trailing spaces
    gbnf_grammar = f"""root      ::= [ ]? order [ ]?
order     ::= item | item [ ]? "," [ ]? order | item " and " order
item      ::= quantity [ ]+ unit [ ]+ product | product [ ]+ quantity [ ]+ unit
quantity  ::= {quantity_rules}
unit      ::= {unit_rules}
product   ::= {product_rules}
"""
    
    logger.info(f"✓ Generated GBNF grammar")
    logger.info(f"  Products: {len(products)}")
    logger.info(f"  Units: {len(units)}")
    logger.info(f"  Quantities: {len(numeric_rules) + len(word_rules)}")
    logger.info(f"  Grammar size: {len(gbnf_grammar)} bytes")
    
    return gbnf_grammar

# =============================================================================
# WHISPER.CPP TRANSCRIBER WITH INLINE GRAMMAR
# =============================================================================

class WhisperCppTranscriber:
    """
    Handles whisper.cpp transcription with inline GBNF grammar constraints.
    Grammar is passed directly to whisper.cpp to constrain output at SOURCE.
    """
    
    def __init__(self, model_path=None, whisper_cpp_path=None):
        self.model_path = model_path or WHISPER_MODEL
        self.whisper_cpp_path = whisper_cpp_path or WHISPER_CPP_PATH
        
        # Verify installation
        if not os.path.exists(self.whisper_cpp_path):
            raise FileNotFoundError(
                f"whisper.cpp not found at {self.whisper_cpp_path}\n"
                f"Please install from: https://github.com/ggerganov/whisper.cpp"
            )
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                f"Run: cd whisper.cpp && bash ./models/download-ggml-model.sh small.en"
            )
        
        # Generate grammar once during initialization
        logger.info("Initializing WhisperCppTranscriber with GRAMMAR constraints...")
        self.grammar = generate_inventory_grammar()
        
        logger.info("✅ WhisperCppTranscriber ready")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Mode: GRAMMAR-CONSTRAINED (inline GBNF)")
    
    def transcribe(self, audio_file):
        """
        Transcribe audio using whisper.cpp with inline grammar.
        Grammar is enforced DURING transcription, not after.
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None
        
        logger.info(f"🎤 Transcribing: {audio_file}")
        logger.info(f"🔒 Grammar constraints: ACTIVE")
        
        # Build command with inline grammar
        cmd = [
            self.whisper_cpp_path,
            "-m", self.model_path,
            "-f", audio_file,
            "--grammar", self.grammar,  # ← INLINE GRAMMAR STRING
            "-l", "en",
            "-t", "4",
            "--no-timestamps",
            "--output-txt",  # Cleaner output format
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"❌ whisper.cpp failed: {result.stderr}")
                return None
            
            # Parse output - extract transcription
            lines = result.stdout.strip().split('\n')
            transcription = None
            
            # Look for the actual transcription line
            for line in lines:
                line = line.strip()
                # Skip debug/info lines
                if not line:
                    continue
                if any(skip in line.lower() for skip in [
                    'whisper_', 'system_info', 'processing', 'load time',
                    'sample time', 'encode time', 'decode time', 'main:'
                ]):
                    continue
                
                # This should be the transcription
                transcription = line
                logger.debug(f"Found transcription line: '{line}'")
                break
            
            if not transcription:
                logger.warning("⚠️  Empty transcription")
                logger.info("   Possible causes: poor audio quality, no speech, or grammar mismatch")
                return None
            
            logger.info(f"✅ Transcription complete ({elapsed:.2f}s)")
            logger.info(f"   GRAMMAR OUTPUT: '{transcription}'")
            
            return transcription
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Transcription timeout (30s)")
            return None
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None

# =============================================================================
# MINIMAL PHONETIC MATCHER (FOR KNOWN VARIATIONS ONLY)
# =============================================================================

class MinimalPhoneticMatcher:
    """
    Lightweight phonetic matcher for known pronunciation variations.
    Uses Double Metaphone algorithm.
    """
    
    def __init__(self, threshold=85):
        self.threshold = threshold
        logger.info(f"Phonetic matcher initialized (threshold={threshold})")
    
    def phonetic_score(self, word1, word2):
        """Calculate phonetic similarity score using Double Metaphone"""
        if word1 == word2:
            return 100.0
        
        # Get metaphone codes
        code1_primary, code1_secondary = doublemetaphone(word1)
        code2_primary, code2_secondary = doublemetaphone(word2)
        
        # Check for exact metaphone match
        if code1_primary and (code1_primary == code2_primary or code1_primary == code2_secondary):
            return 95.0
        if code1_secondary and (code1_secondary == code2_primary or code1_secondary == code2_secondary):
            return 90.0
        
        # Fallback to fuzzy string match
        ratio = fuzz.ratio(word1.lower(), word2.lower())
        return ratio
    
    def match_product(self, transcribed_word, inventory_list):
        """Match transcribed word to inventory product using phonetics"""
        best_match = None
        best_score = 0
        
        for product in inventory_list:
            score = self.phonetic_score(transcribed_word, product)
            if score > best_score:
                best_score = score
                best_match = product
        
        if best_score >= self.threshold:
            if best_match != transcribed_word:
                logger.info(f"   Phonetic: '{transcribed_word}' → '{best_match}' (score={best_score:.1f})")
            return best_match, best_score
        
        return None, 0

phonetic_matcher = MinimalPhoneticMatcher(threshold=PHONETIC_THRESHOLD)

# =============================================================================
# TEXT PROCESSING (MINIMAL - Grammar does most work)
# =============================================================================

def normalize_text(text):
    """Minimal normalization - grammar already constrains output"""
    text = text.lower().strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def process_text(text):
    """
    Minimal processing - grammar already enforced correct structure.
    Only apply known corrections from glossaries.
    """
    text = normalize_text(text)
    
    # Apply unit glossary (singular/plural normalization)
    for variant, standard in UNIT_GLOSSARY.items():
        text = re.sub(r'\b' + variant + r'\b', standard, text)
    
    # Apply product glossary (known aliases)
    for alias, standard in PRODUCT_GLOSSARY.items():
        text = re.sub(r'\b' + alias + r'\b', standard, text)
    
    logger.info(f"   Processed: '{text}'")
    return text

# =============================================================================
# ORDER PARSING
# =============================================================================

def parse_quantity(quantity_str):
    """Parse quantity from string (number or word)"""
    quantity_str = quantity_str.strip().lower()
    
    # Check if it's in word glossary
    if quantity_str in QUANTITY_GLOSSARY:
        return float(QUANTITY_GLOSSARY[quantity_str])
    
    # Try to parse as number
    try:
        return float(quantity_str)
    except ValueError:
        return None

def parse_order(text):
    """
    Parse order from grammar-constrained text.
    Pattern: quantity unit product OR product quantity unit
    """
    text = text.lower().strip()
    # Normalize hyphens to spaces (e.g., "three-strip" → "three strip")
    text = text.replace('-', ' ')
    logger.info(f"Parsing order: '{text}'")
    
    # Split by common separators
    items_text = re.split(r',|\band\b', text)
    
    parsed_items = []
    
    for item_text in items_text:
        item_text = item_text.strip()
        if not item_text:
            continue
        
        logger.debug(f"  Parsing item: '{item_text}'")
        
        # Build search spaces
        all_products = list(inventory.keys())
        all_brands = []
        brand_to_product = {}  # Map brand → product
        
        for product_name, product_info in inventory.items():
            for brand_name in product_info.get('brands', {}).keys():
                all_brands.append(brand_name)
                brand_to_product[brand_name] = product_name
        
        all_units = list(set(UNIT_GLOSSARY.values()))
        
        # Try to match product or brand
        matched_product = None
        matched_brand = None
        
        # First, try exact product match
        for product in all_products:
            if product in item_text:
                matched_product = product
                logger.debug(f"    Matched product: '{product}'")
                break
        
        # If no product, try brand match
        if not matched_product:
            # First try exact match
            for brand in all_brands:
                if brand in item_text:
                    matched_brand = brand
                    matched_product = brand_to_product[brand]
                    logger.debug(f"    Matched brand: '{brand}' → product: '{matched_product}'")
                    break
            
            # If no exact match, try fuzzy match
            if not matched_product:
                best_brand = None
                best_score = 0
                
                # Try matching each brand against the full text
                for brand in all_brands:
                    similarity = fuzz.partial_ratio(brand, item_text)
                    if similarity > best_score:
                        best_score = similarity
                        best_brand = brand
                
                if best_score >= 75:  # 75% similarity threshold
                    matched_brand = best_brand
                    matched_product = brand_to_product[best_brand]
                    logger.debug(f"    Fuzzy matched brand: '{best_brand}' in '{item_text}' (sim={best_score}) → product: '{matched_product}'")

        
        # Try phonetic match if no exact match
        if not matched_product:
            words = item_text.split()
            for word in words:
                # Try matching against products
                match, score = phonetic_matcher.match_product(word, all_products)
                if match:
                    matched_product = match
                    logger.debug(f"    Phonetic matched product: '{word}' → '{match}'")
                    break
                
                # Try matching against brands
                match, score = phonetic_matcher.match_product(word, all_brands)
                if match:
                    matched_brand = match
                    matched_product = brand_to_product[match]
                    logger.debug(f"    Phonetic matched brand: '{word}' → '{match}' → '{matched_product}'")
                    break
        
        if not matched_product:
            logger.warning(f"    No product/brand found in: '{item_text}'")
            continue
        
        # Try to match unit
        matched_unit = None
        for unit in all_units:
            if unit in item_text:
                matched_unit = unit
                logger.debug(f"    Matched unit: '{unit}'")
                break
        
        if not matched_unit:
            matched_unit = inventory[matched_product]['unit']
            logger.debug(f"    Using default unit: '{matched_unit}'")
        
        # Try to extract quantity (exclude numbers from matched brand/product name)
        quantity = None
        words = item_text.split()
        
        # Filter out words that are part of the matched brand or product
        brand_words = matched_brand.split() if matched_brand else []
        product_words = matched_product.split()
        
        for word in words:
            # Skip if this word is part of the brand or product name
            if word in brand_words or word in product_words:
                continue
            
            qty = parse_quantity(word)
            if qty is not None:
                quantity = qty
                logger.debug(f"    Matched quantity: {qty} from '{word}'")
                break
        
        # If still no quantity, look for standalone numbers (not in brand/product)
        if quantity is None:
            # Remove brand and product from text
            temp_text = item_text
            if matched_brand:
                temp_text = temp_text.replace(matched_brand, "")
            temp_text = temp_text.replace(matched_product, "")
            
            number_match = re.search(r'\b(\d+)\b', temp_text)
            if number_match:
                quantity = float(number_match.group(1))
                logger.debug(f"    Extracted quantity: {quantity} (regex, after removing brand/product)")
        
        if quantity is None:
            quantity = 1.0
            logger.debug(f"    Using default quantity: 1")
        
        # Build item
        product_info = inventory[matched_product]
        
        # Use matched brand if available, otherwise default
        if matched_brand:
            brand = matched_brand
            price_per_unit = product_info['brands'].get(matched_brand, product_info['price'])
        else:
            brand = list(product_info['brands'].keys())[0] if product_info['brands'] else matched_product.title()
            price_per_unit = product_info['brands'].get(brand, product_info['price'])
        
        item = {
            "product_name": matched_product.title(),
            "brand_name": brand.title(),
            "quantity": quantity,
            "unit": matched_unit,
            "price_per_unit": price_per_unit,
            "total_price": quantity * price_per_unit
        }
        
        parsed_items.append(item)
        logger.info(f"   ✓ Parsed: {quantity} {matched_unit} {matched_product}")
    
    return parsed_items

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_audio_order(audio_file_path, transcriber):
    """
    Complete pipeline: Audio → whisper.cpp (GRAMMAR) → Minimal Processing → Parse → JSON
    """
    logger.info("=" * 60)
    logger.info("Processing audio order with GRAMMAR constraints")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("📋 STEP 2: GRAMMAR-CONSTRAINED TRANSCRIPTION")
    print("=" * 60)
    
    # Transcribe with grammar constraints
    print("🎤 Transcribing with whisper.cpp + GBNF grammar...")
    transcribed_text = transcriber.transcribe(audio_file_path)
    
    if not transcribed_text:
        logger.error("Transcription failed or returned empty")
        print("❌ Transcription failed.")
        print("   Possible causes:")
        print("   - Audio too short or no speech detected")
        print("   - Speech doesn't match any grammar patterns")
        print("   - Audio quality issues")
        return None
    
    print(f"✅ Grammar-constrained output: '{transcribed_text}'")
    
    # Minimal processing (mostly just glossary normalization)
    print("🔄 Applying minimal normalization...")
    processed_text = process_text(transcribed_text)
    print(f"✅ Normalized: '{processed_text}'")
    
    # Parse into structured order
    print("🔍 Parsing into structured order...")
    parsed_order = parse_order(processed_text)
    
    if not parsed_order:
        logger.warning("No items could be parsed")
        print("⚠️  Could not parse any items from the transcription.")
        return None
    
    print(f"✅ Parsed {len(parsed_order)} item(s)")
    
    return {
        "grammar_transcription": transcribed_text,
        "normalized_text": processed_text,
        "parsed_order": parsed_order
    }

# =============================================================================
# ESP32 SERIAL RECORDING (Unchanged)
# =============================================================================

def record_audio_from_esp32(serial_port=SERIAL_PORT, baud_rate=BAUD_RATE, 
                            output_filename=WAVE_OUTPUT_FILENAME):
    """Record audio from ESP32 via serial port"""
    print("\n" + "=" * 60)
    print("📋 STEP 1: AUDIO RECORDING FROM ESP32")
    print("=" * 60)
    
    try:
        print(f"📡 Connecting to ESP32 on {serial_port}...")
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        print("✅ Connected to ESP32")
        
        print("🎙️  Say your pharmaceutical order clearly...")
        print("   Example: 'paracetamol 3 strips'")
        print("   Waiting for audio data...")
        
        audio_data = bytearray()
        start_marker = b'START'
        end_marker = b'END'
        recording = False
        
        while True:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                
                if start_marker in chunk:
                    recording = True
                    print("📝 Recording started...")
                    start_idx = chunk.find(start_marker) + len(start_marker)
                    chunk = chunk[start_idx:]
                
                if recording:
                    end_idx = chunk.find(end_marker)
                    if end_idx != -1:
                        audio_data.extend(chunk[:end_idx])
                        print("✅ Recording completed")
                        break
                    else:
                        audio_data.extend(chunk)
        
        ser.close()
        
        # Save as WAV
        print(f"💾 Saving audio to {output_filename}...")
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        
        duration = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        print(f"✅ Audio saved: {len(audio_data)} bytes ({duration:.2f}s)")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        print(f"❌ Recording failed: {e}")
        return None

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main pipeline with ESP32 recording"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "PHARMACEUTICAL VOICE ORDER PIPELINE" + " " * 23 + "║")
    print("║" + " " * 15 + "Grammar-Constrained Mode" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Initialize transcriber (loads grammar)
    print("\n🔧 Initializing transcriber with GBNF grammar...")
    transcriber = WhisperCppTranscriber()
    
    print("\n🚀 Pipeline ready!")
    print("=" * 60)
    
    try:
        while True:
            # Record audio from ESP32
            audio_file = record_audio_from_esp32()
            
            if not audio_file:
                print("❌ Recording failed. Exiting...")
                break
            
            # Process the audio order
            result = process_audio_order(audio_file, transcriber)
            
            if result:
                print("\n" + "=" * 60)
                print("📊 FINAL RESULTS")
                print("=" * 60)
                print(f"🗣️  Grammar Output: {result['grammar_transcription']}")
                print(f"⚙️  Normalized: {result['normalized_text']}")
                print("\n📦 Parsed Order:")
                print(json.dumps(result['parsed_order'], indent=2, ensure_ascii=False))
                
                # Show summary
                print("\n💰 Order Summary:")
                total = 0
                for item in result['parsed_order']:
                    print(f"   • {item['quantity']} {item['unit']} {item['product_name']} → ₹{item['total_price']:.2f}")
                    total += item['total_price']
                print(f"   📊 TOTAL: ₹{total:.2f}")
            else:
                print("\n❌ Could not process the order.")
            
            # Continue or exit
            print("\n" + "=" * 60)
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("👋 Exiting pipeline. Thank you!")
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        print(f"\n❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()

