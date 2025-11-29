#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pharmaceutical Voice Order Pipeline using Whisper ASR
------------------------------------------------------
Records audio from ESP32, transcribes using fine-tuned Whisper model,
and parses pharmaceutical orders into structured JSON.
"""

import serial
import wave
import re
import json
import os
import time
import logging
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa
from rapidfuzz import process, fuzz

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WhisperPharmaPipeline")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Whisper Model Configuration ---
WHISPER_MODEL_PATH = "runpod_backup/whisper_pharma_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# PHARMACEUTICAL GLOSSARIES
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
# WHISPER MODEL INITIALIZATION
# =============================================================================

class WhisperTranscriber:
    """Handles Whisper model loading and transcription"""
    
    def __init__(self, model_path=WHISPER_MODEL_PATH, device=DEVICE):
        """Initialize Whisper model and processor"""
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Whisper model and processor"""
        logger.info(f"Loading Whisper model from {self.model_path}")
        start_time = time.time()
        
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded successfully on {self.device} in {load_time:.2f}s")
            print(f"✓ Whisper model loaded on {self.device} ({load_time:.2f}s)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"❌ Error loading model: {e}")
            return False
    
    def transcribe(self, audio_file_path):
        """Transcribe audio file using Whisper model"""
        if self.model is None or self.processor is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        logger.info(f"Transcribing audio: {audio_file_path}")
        start_time = time.time()
        
        try:
            # Load audio with soundfile
            audio_array, sampling_rate = sf.read(audio_file_path)
            duration = len(audio_array) / sampling_rate
            logger.info(f"Audio loaded: duration={duration:.2f}s, SR={sampling_rate}Hz, shape={audio_array.shape}")
            
            # Resample to 16kHz if needed (Whisper requires 16kHz)
            if sampling_rate != 16000:
                logger.info(f"Resampling from {sampling_rate}Hz to 16000Hz")
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            
            # Prepare input features
            logger.info("Preparing input features for Whisper")
            input_features = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features
            
            # Move to device
            input_features = input_features.to(self.device)
            logger.info(f"Input features shape: {input_features.shape}")
            
            # Generate transcription
            logger.info("Generating transcription...")
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            transcribe_time = time.time() - start_time
            logger.info(f"✓ Transcription complete in {transcribe_time:.2f}s")
            logger.info(f"RAW TRANSCRIPTION: '{transcription}'")
            
            return transcription
            
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
# TEXT PROCESSING & PARSING
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

def match_product(product_name):
    """
    Match a product string to inventory using fuzzy matching.
    Returns the base product name if found, None otherwise.
    """
    processed_name = product_name.lower().strip()
    
    # Direct match
    if processed_name in search_space:
        return search_space[processed_name]
    
    choices = list(search_space.keys())
    
    # Token set ratio (good for partial matches)
    best_match_token, score_token, _ = process.extractOne(
        processed_name, choices, scorer=fuzz.token_set_ratio
    )
    if score_token >= 70:
        return search_space[best_match_token]
    
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
            
            # Check if it's a unit
            if clean_word in UNIT_GLOSSARY:
                mapped_unit = UNIT_GLOSSARY[clean_word]
                processed_words.append(mapped_unit)
                logger.debug(f"Mapped unit: '{clean_word}' -> '{mapped_unit}'")
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
    
    # Match to inventory
    matched = match_product(product_key)
    
    if matched:
        logger.info(f"✓ Matched product '{product_key}' to '{matched}'")
        inv_info = inventory[matched]
        parsed_unit = inv_info["unit"]
        brand, price = None, None
        
        if inv_info.get("brands"):
            brand_choices = list(inv_info["brands"].keys())
            best_brand, score, _ = process.extractOne(
                product_key, brand_choices, scorer=fuzz.token_set_ratio
            )
            
            if score >= 70:
                brand = best_brand
                price = inv_info["brands"][brand]
                logger.info(f"✓ Matched brand '{product_key}' to '{brand}' (score={score})")
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
    """Complete pipeline: Audio -> Whisper -> Parse -> JSON"""
    logger.info("=" * 60)
    logger.info("Processing audio order")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("📋 STEP 2: TRANSCRIPTION AND PROCESSING")
    print("=" * 60)
    
    # Transcribe with Whisper
    print("🎤 Transcribing with Whisper...")
    transcribed_text = transcriber.transcribe(audio_file_path)
    
    if not transcribed_text:
        logger.error("Transcription failed")
        print("❌ Could not transcribe the audio.")
        return None
    
    print(f"✅ Transcription complete: '{transcribed_text}'")
    
    # Process the transcription
    print("🔄 Processing transcription...")
    processed_text = process_text(transcribed_text)
    print(f"✅ Processed text: '{processed_text}'")
    
    # Parse into structured order
    print("🔍 Parsing into structured order...")
    parsed_order = parse_order(processed_text)
    
    if not parsed_order:
        logger.warning("No items could be parsed from the order")
        print("⚠️  Could not parse any items from the order.")
    
    return {
        "original_transcription": transcribed_text,
        "processed_text": processed_text,
        "parsed_order": parsed_order
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 WHISPER PHARMACEUTICAL VOICE ORDER PIPELINE")
    print("=" * 60)
    
    # Initialize Whisper transcriber
    print("\n📦 Loading Whisper model...")
    transcriber = WhisperTranscriber()
    if not transcriber.load_model():
        print("❌ Failed to load Whisper model. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("📋 STEP 1: AUDIO RECORDING")
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
            print(f"⚙️  Processed Text: {result['processed_text']}")
            print("📦 Parsed Order:")
            print(json.dumps(result['parsed_order'], indent=2, ensure_ascii=False))
            
            total_price = sum(
                item.get('total_price', 0)
                for item in result['parsed_order']
                if item.get('total_price')
            )
            if total_price > 0:
                print(f"\n💰 Total Order Value: ₹{total_price:.2f}")
            
            logger.info("Pipeline completed successfully")
    else:
        print("\nPipeline stopped because audio recording failed.")
        logger.error("Audio recording failed")

if __name__ == "__main__":
    main()



