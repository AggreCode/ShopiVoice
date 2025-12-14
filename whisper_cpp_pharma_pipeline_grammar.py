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
WHISPER_CPP_PATH = "./whisper.cpp/build/bin/whisper-cli"  # Updated: now uses whisper-cli (not main)
WHISPER_MODEL = "./whisper.cpp/models/ggml-small.en.bin"
PHONETIC_THRESHOLD = 50  # Aggressive matching - always returns best score

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

# No transcription fixes - rely purely on grammar + aggressive phonetic matching

# Pharmaceutical Inventory (Flattened - all products independent)
inventory = {
    # Generic medicines
    "paracetamol": {"unit": "strip", "price": 20},
    "cetirizine": {"unit": "strip", "price": 20},
    "ibuprofen": {"unit": "strip", "price": 30},
    "amoxicillin": {"unit": "strip", "price": 40},
    "azithromycin": {"unit": "strip", "price": 90},
    "omeprazole": {"unit": "strip", "price": 35},
    "metformin": {"unit": "strip", "price": 25},
    "amlodipine": {"unit": "strip", "price": 30},
    "atorvastatin": {"unit": "strip", "price": 35},
    "losartan": {"unit": "strip", "price": 40},
    "levothyroxine": {"unit": "strip", "price": 45},
    "aspirin": {"unit": "strip", "price": 10},
    "fluoxetine": {"unit": "strip", "price": 80},
    "sertraline": {"unit": "strip", "price": 85},
    "multivitamin": {"unit": "bottle", "price": 150},
    "vitamind": {"unit": "strip", "price": 60},
    "calcium": {"unit": "strip", "price": 120},
    
    # Brands (all independent products now)
    "crocin": {"unit": "strip", "price": 20},
    "dolo 650": {"unit": "strip", "price": 32},
    "calpol": {"unit": "strip", "price": 25},
    "brufen": {"unit": "strip", "price": 30},
    "amoxil": {"unit": "strip", "price": 40},
    "azithral": {"unit": "strip", "price": 90},
    "synthroid": {"unit": "strip", "price": 50},
    "eltroxin": {"unit": "strip", "price": 45},
    "prozac": {"unit": "strip", "price": 80},
    "zoloft": {"unit": "strip", "price": 85},
    "centrum": {"unit": "bottle", "price": 150},
    "shelcal": {"unit": "strip", "price": 120},
    "ranidom": {"unit": "bottle", "price": 131},
    
    # Other products
    "dettol": {"unit": "bottle", "price": 80},
    "bandaid": {"unit": "box", "price": 50},
    "cough syrup": {"unit": "bottle", "price": 100},
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
    Simplified - no boundary guards (not supported in whisper.cpp GBNF).
    
    This function constrains vocabulary to reduce hallucinations.
    """
    # 1. Collect all products and sort by length (longer first for better matching)
    products = []
    
    for product_name in inventory.keys():
        products.append(product_name.lower())
    
    # Sort by length descending (longer matches first)
    products.sort(key=len, reverse=True)
    
    # 2. Build product rules (all products in one list)
    product_rules = " | ".join([f'"{escape_gbnf_string(p)}"' for p in products])
    
    # 3. Build units (simple list, no variants)
    units = ["strip", "bottle", "box", "tablet", "kg", "gram", "mg", "ml", "liter"]
    unit_rules = " | ".join([f'"{u}"' for u in units])
    
    # 4. Build quantity rules (numeric only: 0-100)
    numeric_rules = [f'"{i}"' for i in range(0, 101)]
    quantity_rules = " | ".join(numeric_rules)
    
    # 5. Define the Grammar
    # Single pattern: quantity + unit + product
    gbnf_grammar = f"""root     ::= [ ]? order [ ]?
order    ::= item | item [ ]? "," [ ]? order
item     ::= quantity [ ]+ unit [ ]+ product
quantity ::= {quantity_rules}
unit     ::= {unit_rules}
product  ::= {product_rules}
"""
    
    logger.info(f"✓ Generated GBNF grammar")
    logger.info(f"  Products: {len(products)}")
    logger.info(f"  Units: {len(units)}")
    logger.info(f"  Quantities: 0-100 (numeric only)")
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
        self.grammar_file = "pharma_inventory.gbnf"  # Grammar file path
        
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
        
        # Generate grammar and write to file
        logger.info("Initializing WhisperCppTranscriber with GRAMMAR constraints...")
        grammar_content = generate_inventory_grammar()
        
        # Write grammar to file (CRITICAL: whisper.cpp needs file-based grammar)
        with open(self.grammar_file, 'w') as f:
            f.write(grammar_content)
        logger.info(f"✓ Grammar written to {self.grammar_file}")
        
        logger.info("✅ WhisperCppTranscriber ready")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Mode: GRAMMAR-CONSTRAINED (file-based GBNF)")
    
    def transcribe(self, audio_file):
        """
        Transcribe audio using whisper.cpp with inline grammar.
        Grammar is enforced DURING transcription, not after.
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None
        
        logger.info(f"🎤 Transcribing: {audio_file}")
        logger.info(f"🔒 Grammar constraints: ACTIVE (strict mode)")
        logger.info(f"📄 Grammar file: {self.grammar_file}")
        
        # Build command - pass filename to --grammar (whisper-cli will read it)
        cmd = [
            self.whisper_cpp_path,
            "-m", self.model_path,
            "-f", audio_file,
            "--grammar", self.grammar_file,  # ← Pass filename (whisper-cli reads file)
            "--grammar-rule", "root",  # ← CRITICAL: Specify top-level GBNF rule
            "--grammar-penalty", "7.0",  # ← OPTIMAL: Captures all items while guiding toward grammar
            "--entropy-thold", "10.0",  # ← Allow high confusion without fallback
            "--logprob-thold", "-100.0",  # ← Don't give up even if probability is low
            "--temperature", "0.4",  # ← Allow exploration to find valid grammar paths
            "--max-len", "0",  # ← No length limit (let it transcribe full audio)
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
    Aggressive phonetic matcher for pronunciation variations.
    Uses Double Metaphone + Fuzzy matching.
    ALWAYS returns the best match, even if score is low.
    """
    
    def __init__(self, threshold=50):
        self.threshold = threshold
        logger.info(f"🔊 Aggressive Phonetic Matcher initialized (threshold={threshold}%, always returns best match)")
    
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
        """
        Match transcribed word to inventory product using phonetics.
        ALWAYS returns the best match, even if score is below threshold.
        """
        best_match = None
        best_score = 0
        
        for product in inventory_list:
            score = self.phonetic_score(transcribed_word, product)
            if score > best_score:
                best_score = score
                best_match = product
        
        # ALWAYS return best match (no None return)
        if best_match:
            if best_score >= self.threshold:
                logger.info(f"   ✅ Phonetic: '{transcribed_word}' → '{best_match}' (score={best_score:.1f}%)")
            else:
                logger.warning(f"   ⚠️  Low-confidence match: '{transcribed_word}' → '{best_match}' (score={best_score:.1f}%)")
            return best_match, best_score
        
        # Fallback (should never happen if inventory_list is not empty)
        return inventory_list[0] if inventory_list else None, 0

phonetic_matcher = MinimalPhoneticMatcher(threshold=PHONETIC_THRESHOLD)

# =============================================================================
# TEXT PROCESSING (MINIMAL - Grammar does most work)
# =============================================================================

def normalize_text(text):
    """
    Minimal normalization - no glossary fixes.
    Pure reliance on grammar + aggressive phonetic matching.
    """
    text = text.lower().strip()
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

# =============================================================================
# ORDER PARSING
# =============================================================================

def parse_order(text):
    """
    Simplified parser for grammar-constrained text.
    Pattern: quantity unit product (e.g., "5 bottle fluoxetine")
    
    Strategy:
    1. Split by comma
    2. For each item, try regex pattern matching
    3. Exact product match first
    4. Phonetic fallback if needed
    """
    text = normalize_text(text)
    logger.info(f"Parsing: '{text}'")
    
    parsed_items = []
    all_products = list(inventory.keys())
    valid_units = ["strip", "bottle", "box", "tablet", "kg", "gram", "mg", "ml", "liter"]
    
    # Build regex pattern to match: quantity unit product_name
    # Product names can be multi-word (e.g., "dolo 650", "cough syrup")
    # We need to match until we hit the next quantity or end of string
    units_pattern = "|".join(valid_units)
    
    # Pattern: number + unit + everything until next number+unit or end
    pattern = r'(\d+(?:\.\d+)?)\s+(' + units_pattern + r')\s+(.+?)(?=\s+\d+\s+(?:' + units_pattern + r')|$)'
    
    matches = re.finditer(pattern, text)
    
    for match in matches:
        quantity_str, unit, product_name = match.groups()
        quantity = float(quantity_str)
        product_name = product_name.strip().rstrip(',')  # Remove trailing comma if present
        
        logger.debug(f"  Extracted: qty={quantity}, unit={unit}, product='{product_name}'")
        
        # Match product: exact first, then phonetic
        matched_product = None
        
        if product_name in all_products:
            matched_product = product_name
            logger.debug(f"    ✓ Exact product match: '{product_name}'")
        else:
            # Phonetic fallback
            match, score = phonetic_matcher.match_product(product_name, all_products)
            if match:
                matched_product = match
                logger.info(f"    ✓ Phonetic match: '{product_name}' → '{match}' (score={score:.1f})")
            else:
                logger.warning(f"    ✗ No match for: '{product_name}'")
                continue
        
        # Validate unit (use exact or default from inventory)
        if unit not in valid_units:
            unit = inventory[matched_product]['unit']
            logger.debug(f"    Using default unit: '{unit}'")
        
        # Build item (no brand_name field in flat inventory)
        product_info = inventory[matched_product]
        
        item = {
            "product_name": matched_product.title(),
            "quantity": quantity,
            "unit": unit,
            "price_per_unit": product_info['price'],
            "total_price": quantity * product_info['price']
        }
        
        parsed_items.append(item)
        logger.info(f"   ✓ Parsed: {quantity} {unit} {matched_product}")
    
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
    
    # Minimal normalization (lowercase, trim, transcription fixes)
    print("🔄 Applying minimal normalization...")
    processed_text = normalize_text(transcribed_text)
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
    """
    Record audio from ESP32 via serial port (button-controlled)
    Compatible with esp32_button_streamer.ino
    """
    print("\n" + "=" * 60)
    print("📋 STEP 1: AUDIO RECORDING FROM ESP32")
    print("=" * 60)
    
    try:
        print(f"📡 Connecting to ESP32 on {serial_port}...")
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        print("✅ Connected to ESP32")
        
        # Clear any existing data
        ser.reset_input_buffer()
        
        print("\n🎙️  Press the button on ESP32 to start recording")
        print("   Say your pharmaceutical order clearly")
        print("   Example: 'paracetamol 3 strips'")
        print("   Press button again to stop")
        print("\n⏳ Waiting for button press...")
        
        audio_data = bytearray()
        recording = False
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline()
                
                try:
                    # Try to decode as text (for status messages)
                    text = line.decode('utf-8', errors='ignore').strip()
                    
                    if "Recording started" in text:
                        recording = True
                        print("📝 Recording started (LED should be ON)")
                        print("   Speak now...")
                        continue
                    
                    elif "Recording stopped" in text:
                        print("✅ Recording stopped (LED should be OFF)")
                        break
                    
                    elif text and not recording:
                        # Print ESP32 status messages while waiting
                        print(f"   ESP32: {text}")
                        continue
                
                except UnicodeDecodeError:
                    # This is binary audio data, not text
                    pass
                
                # If recording, collect binary audio data
                if recording:
                    audio_data.extend(line)
        
        ser.close()
        
        if len(audio_data) == 0:
            print("⚠️  No audio data received")
            return None
        
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
        
    except serial.SerialException as e:
        logger.error(f"Serial port error: {e}")
        print(f"❌ Serial port error: {e}")
        print(f"   Make sure ESP32 is connected to {serial_port}")
        return None
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

