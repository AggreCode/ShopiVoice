#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Updated Smart Dataset Generator

This script generates a dataset that ensures good coverage of all glossary items
while keeping the dataset size manageable. It uses the specified templates for
dataset generation and grammar building.
"""

import os
import json
import random
import logging
import itertools
import csv
import collections
from typing import Dict, List, Set, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UpdatedSmartDatasetGenerator")

# Import the original DatasetGenerator
from backup.dataset_generator import DatasetGenerator

# Define the specific templates to use
SPECIFIED_TEMPLATES = [
    "{quantity} {unit} {product}",
    "{product} {quantity} {unit}"
]

class UpdatedSmartDatasetGenerator(DatasetGenerator):
    """
    Enhanced dataset generator that creates a smart, balanced dataset
    with good coverage of all glossary items using the specified templates.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with parent class constructor"""
        super().__init__(config)
        
        # Override templates with specified ones
        self.templates = SPECIFIED_TEMPLATES
        logger.info(f"Using specified templates: {self.templates}")
        
        # Set minimum occurrences for each glossary item
        self.min_occurrences = self.config.get('min_occurrences', 20)
        
        # Initialize coverage tracking
        self.coverage = {
            'quantity': collections.defaultdict(int),
            'unit': collections.defaultdict(int),
            'product': collections.defaultdict(int),
            'template': collections.defaultdict(int)
        }
        
        logger.info(f"UpdatedSmartDatasetGenerator initialized with min_occurrences={self.min_occurrences}")
    
    def set_language_templates(self, language: str) -> None:
        """
        Override parent method to always use the specified templates.
        
        Args:
            language: Language code ('en', 'or', 'mixed')
        """
        # Ignore the language parameter and always use specified templates
        self.templates = SPECIFIED_TEMPLATES
        self.language = language
        logger.info(f"Set language to {language}, but using specified templates: {self.templates}")
    
    def generate_smart_dataset(self, target_size: int = 30000) -> List[Dict[str, Any]]:
        """
        Generate a smart dataset that ensures good coverage of all glossary items.
        
        Args:
            target_size: Target dataset size
            
        Returns:
            List of utterance data dictionaries
        """
        dataset = []
        
        # Phase 1: Ensure minimum coverage for each glossary item
        logger.info("Phase 1: Ensuring minimum coverage for each glossary item")
        
        # Get all glossary items
        quantities = list(self.quantity_glossary.keys())
        units = list(self.unit_glossary.keys())
        products = list(self.product_glossary.keys())
        
        # Ensure each product appears with different quantities and units
        for product in products:
            # Select a subset of quantities and units to use with this product
            product_quantities = random.sample(
                quantities, 
                min(len(quantities), max(5, self.min_occurrences // 4))
            )
            product_units = random.sample(
                units, 
                min(len(units), max(3, self.min_occurrences // 7))
            )
            
            # Generate combinations
            combinations = list(itertools.product(product_quantities, product_units, [product]))
            
            # Shuffle combinations
            random.shuffle(combinations)
            
            # Generate utterances for each combination
            for i, (quantity, unit, prod) in enumerate(combinations):
                if i >= self.min_occurrences:
                    break
                    
                # Select template
                template = random.choice(self.templates)
                
                # Generate utterance
                utterance = template.format(
                    quantity=quantity,
                    unit=unit,
                    product=prod
                )
                
                # Sanitize utterance (inherited from parent class)
                utterance = self._sanitize_utterance(utterance)
                
                # Get canonical values
                canonical_quantity = self.quantity_glossary[quantity]
                canonical_unit = self.unit_glossary[unit]
                canonical_product = self.product_glossary[prod]
                
                # Create structured data
                structured_data = {
                    "quantity": canonical_quantity,
                    "unit": canonical_unit,
                    "product": canonical_product
                }
                
                # Create utterance data
                utterance_data = {
                    "id": f"utt_{len(dataset):06d}",
                    "utterance": utterance,
                    "structured_data": structured_data,
                    "template": template
                }
                
                # Add to dataset
                dataset.append(utterance_data)
                
                # Update coverage
                self.coverage['quantity'][quantity] += 1
                self.coverage['unit'][unit] += 1
                self.coverage['product'][prod] += 1
                self.coverage['template'][template] += 1
        
        logger.info(f"Phase 1 complete: Generated {len(dataset)} utterances")
        
        # Phase 2: Ensure each quantity appears with different units
        if len(dataset) < target_size:
            logger.info("Phase 2: Ensuring each quantity appears with different units")
            
            for quantity in quantities:
                if self.coverage['quantity'][quantity] < self.min_occurrences:
                    # How many more utterances needed for this quantity
                    needed = self.min_occurrences - self.coverage['quantity'][quantity]
                    
                    # Generate utterances
                    for _ in range(needed):
                        # Select random unit and product
                        unit = random.choice(units)
                        product = random.choice(products)
                        
                        # Select template
                        template = random.choice(self.templates)
                        
                        # Generate utterance
                        utterance = template.format(
                            quantity=quantity,
                            unit=unit,
                            product=product
                        )
                        
                        # Sanitize utterance (inherited from parent class)
                        utterance = self._sanitize_utterance(utterance)
                        
                        # Get canonical values
                        canonical_quantity = self.quantity_glossary[quantity]
                        canonical_unit = self.unit_glossary[unit]
                        canonical_product = self.product_glossary[product]
                        
                        # Create structured data
                        structured_data = {
                            "quantity": canonical_quantity,
                            "unit": canonical_unit,
                            "product": canonical_product
                        }
                        
                        # Create utterance data
                        utterance_data = {
                            "id": f"utt_{len(dataset):06d}",
                            "utterance": utterance,
                            "structured_data": structured_data,
                            "template": template
                        }
                        
                        # Add to dataset
                        dataset.append(utterance_data)
                        
                        # Update coverage
                        self.coverage['quantity'][quantity] += 1
                        self.coverage['unit'][unit] += 1
                        self.coverage['product'][product] += 1
                        self.coverage['template'][template] += 1
                        
                        # Check if target size reached
                        if len(dataset) >= target_size:
                            break
                    
                    # Check if target size reached
                    if len(dataset) >= target_size:
                        break
        
        logger.info(f"Phase 2 complete: Generated {len(dataset)} utterances")
        
        # Phase 3: Ensure each unit appears with different quantities
        if len(dataset) < target_size:
            logger.info("Phase 3: Ensuring each unit appears with different quantities")
            
            for unit in units:
                if self.coverage['unit'][unit] < self.min_occurrences:
                    # How many more utterances needed for this unit
                    needed = self.min_occurrences - self.coverage['unit'][unit]
                    
                    # Generate utterances
                    for _ in range(needed):
                        # Select random quantity and product
                        quantity = random.choice(quantities)
                        product = random.choice(products)
                        
                        # Select template
                        template = random.choice(self.templates)
                        
                        # Generate utterance
                        utterance = template.format(
                            quantity=quantity,
                            unit=unit,
                            product=product
                        )
                        
                        # Sanitize utterance (inherited from parent class)
                        utterance = self._sanitize_utterance(utterance)
                        
                        # Get canonical values
                        canonical_quantity = self.quantity_glossary[quantity]
                        canonical_unit = self.unit_glossary[unit]
                        canonical_product = self.product_glossary[product]
                        
                        # Create structured data
                        structured_data = {
                            "quantity": canonical_quantity,
                            "unit": canonical_unit,
                            "product": canonical_product
                        }
                        
                        # Create utterance data
                        utterance_data = {
                            "id": f"utt_{len(dataset):06d}",
                            "utterance": utterance,
                            "structured_data": structured_data,
                            "template": template
                        }
                        
                        # Add to dataset
                        dataset.append(utterance_data)
                        
                        # Update coverage
                        self.coverage['quantity'][quantity] += 1
                        self.coverage['unit'][unit] += 1
                        self.coverage['product'][product] += 1
                        self.coverage['template'][template] += 1
                        
                        # Check if target size reached
                        if len(dataset) >= target_size:
                            break
                    
                    # Check if target size reached
                    if len(dataset) >= target_size:
                        break
        
        logger.info(f"Phase 3 complete: Generated {len(dataset)} utterances")
        
        # Phase 4: Fill remaining dataset with random utterances
        if len(dataset) < target_size:
            logger.info(f"Phase 4: Filling dataset to target size ({target_size})")
            
            remaining = target_size - len(dataset)
            
            # Generate random utterances
            for _ in range(remaining):
                # Select random quantity, unit, and product
                quantity = random.choice(quantities)
                unit = random.choice(units)
                product = random.choice(products)
                
                # Select template
                template = random.choice(self.templates)
                
                # Generate utterance
                utterance = template.format(
                    quantity=quantity,
                    unit=unit,
                    product=product
                )
                
                # Get canonical values
                canonical_quantity = self.quantity_glossary[quantity]
                canonical_unit = self.unit_glossary[unit]
                canonical_product = self.product_glossary[product]
                
                # Create structured data
                structured_data = {
                    "quantity": canonical_quantity,
                    "unit": canonical_unit,
                    "product": canonical_product
                }
                
                # Create utterance data
                utterance_data = {
                    "id": f"utt_{len(dataset):06d}",
                    "utterance": utterance,
                    "structured_data": structured_data,
                    "template": template
                }
                
                # Add to dataset
                dataset.append(utterance_data)
                
                # Update coverage
                self.coverage['quantity'][quantity] += 1
                self.coverage['unit'][unit] += 1
                self.coverage['product'][product] += 1
                self.coverage['template'][template] += 1
        
        logger.info(f"Phase 4 complete: Generated {len(dataset)} utterances")
        
        # Print coverage statistics
        self._print_coverage_stats()
        
        return dataset
    
    def _print_coverage_stats(self) -> None:
        """Print coverage statistics"""
        logger.info("Coverage statistics:")
        
        # Quantity coverage
        min_qty = min(self.coverage['quantity'].values()) if self.coverage['quantity'] else 0
        max_qty = max(self.coverage['quantity'].values()) if self.coverage['quantity'] else 0
        avg_qty = sum(self.coverage['quantity'].values()) / len(self.coverage['quantity']) if self.coverage['quantity'] else 0
        
        logger.info(f"  Quantities: min={min_qty}, max={max_qty}, avg={avg_qty:.2f}")
        
        # Unit coverage
        min_unit = min(self.coverage['unit'].values()) if self.coverage['unit'] else 0
        max_unit = max(self.coverage['unit'].values()) if self.coverage['unit'] else 0
        avg_unit = sum(self.coverage['unit'].values()) / len(self.coverage['unit']) if self.coverage['unit'] else 0
        
        logger.info(f"  Units: min={min_unit}, max={max_unit}, avg={avg_unit:.2f}")
        
        # Product coverage
        min_prod = min(self.coverage['product'].values()) if self.coverage['product'] else 0
        max_prod = max(self.coverage['product'].values()) if self.coverage['product'] else 0
        avg_prod = sum(self.coverage['product'].values()) / len(self.coverage['product']) if self.coverage['product'] else 0
        
        logger.info(f"  Products: min={min_prod}, max={max_prod}, avg={avg_prod:.2f}")
        
        # Template coverage
        min_tmpl = min(self.coverage['template'].values()) if self.coverage['template'] else 0
        max_tmpl = max(self.coverage['template'].values()) if self.coverage['template'] else 0
        avg_tmpl = sum(self.coverage['template'].values()) / len(self.coverage['template']) if self.coverage['template'] else 0
        
        logger.info(f"  Templates: min={min_tmpl}, max={max_tmpl}, avg={avg_tmpl:.2f}")
    
    def generate_and_save_smart_dataset(self, target_size: int = 30000) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate and save a smart dataset.
        
        Args:
            target_size: Target dataset size
            
        Returns:
            Dictionary containing split datasets
        """
        # Generate dataset
        dataset = self.generate_smart_dataset(target_size)
        
        # Split dataset
        dataset_splits = self.split_dataset(dataset)
        
        # Save dataset
        self.save_dataset(dataset_splits)
        
        # Save coverage statistics
        coverage_path = os.path.join(self.output_dir, "coverage_stats.json")
        with open(coverage_path, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            coverage_dict = {
                k: dict(v) for k, v in self.coverage.items()
            }
            json.dump(coverage_dict, f, indent=2)
        
        return dataset_splits


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Updated Smart Dataset Generator")
    parser.add_argument("--glossary-file", type=str, default=None,
                        help="JSON file containing glossaries")
    parser.add_argument("--output-dir", type=str, default="smart_dataset",
                        help="Output directory for generated dataset")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "or", "mixed"],
                        help="Language code")
    parser.add_argument("--dataset-size", type=int, default=30000,
                        help="Target dataset size")
    parser.add_argument("--min-occurrences", type=int, default=20,
                        help="Minimum occurrences for each glossary item")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--kaldi", action="store_true",
                        help="Create Kaldi-format files")
    parser.add_argument("--espnet", action="store_true",
                        help="Create ESPnet-format files")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "language": args.language,
        "dataset_size": args.dataset_size,
        "min_occurrences": args.min_occurrences,
        "seed": args.seed
    }
    
    # Create dataset generator
    generator = UpdatedSmartDatasetGenerator(config)
    
    # Set language-specific templates
    generator.set_language_templates(args.language)
    
    # Load glossaries if specified
    if args.glossary_file:
        generator.load_glossaries_from_file(args.glossary_file)
    
    # Generate dataset
    dataset_splits = generator.generate_and_save_smart_dataset(args.dataset_size)
    
    # Create additional format files if requested
    if args.kaldi:
        generator.create_kaldi_files(dataset_splits)
    
    if args.espnet:
        generator.create_espnet_files(dataset_splits)
    
    print(f"Smart dataset generated in {args.output_dir}")




