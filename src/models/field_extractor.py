"""
Field Extraction Module

This module provides advanced field extraction capabilities using
Named Entity Recognition (NER) and machine learning approaches.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """Represents an extracted field with metadata."""
    name: str
    value: Any
    confidence: float
    start_pos: int
    end_pos: int
    extraction_method: str
    context: str


class FieldExtractor:
    """Advanced field extractor using multiple extraction methods."""
    
    def __init__(self, use_ner: bool = True, use_ml: bool = True):
        """
        Initialize field extractor.
        
        Args:
            use_ner: Whether to use Named Entity Recognition
            use_ml: Whether to use machine learning for field extraction
        """
        self.use_ner = use_ner
        self.use_ml = use_ml
        
        # Initialize NER if available
        if use_ner:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy NER model loaded successfully")
            except (ImportError, OSError):
                logger.warning("SpaCy not available, NER disabled")
                self.use_ner = False
                self.nlp = None
        else:
            self.nlp = None
        
        # Initialize ML models for field extraction
        if use_ml:
            self.field_models = {}
            self.vectorizers = {}
        
        # Field-specific patterns
        self.field_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            'currency': r'\$[\d,]+\.?\d{0,2}',
            'percentage': r'\d+\.?\d*%',
            'zip_code': r'\b\d{5}(-\d{4})?\b',
            'invoice_number': r'Invoice\s*#?\s*(\w+)',
            'receipt_number': r'Receipt\s*#?\s*(\w+)',
            'contract_id': r'Contract\s*#?\s*(\w+)'
        }
    
    def extract_with_patterns(self, text: str, field_name: str) -> List[ExtractedField]:
        """
        Extract fields using regex patterns.
        
        Args:
            text: Input text
            field_name: Name of the field to extract
            
        Returns:
            List of extracted fields
        """
        fields = []
        
        if field_name in self.field_patterns:
            pattern = self.field_patterns[field_name]
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                field = ExtractedField(
                    name=field_name,
                    value=match.group(0),
                    confidence=0.8,  # Base confidence for pattern matching
                    start_pos=match.start(),
                    end_pos=match.end(),
                    extraction_method="regex",
                    context=text[max(0, match.start()-20):match.end()+20]
                )
                fields.append(field)
        
        return fields
    
    def extract_with_ner(self, text: str) -> List[ExtractedField]:
        """
        Extract fields using Named Entity Recognition.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted fields
        """
        fields = []
        
        if not self.use_ner or self.nlp is None:
            return fields
        
        try:
            doc = self.nlp(text)
            
            # Map SpaCy entities to our field names
            entity_mapping = {
                'PERSON': 'person_name',
                'ORG': 'organization',
                'GPE': 'location',
                'MONEY': 'currency',
                'DATE': 'date',
                'TIME': 'time',
                'EMAIL': 'email',
                'PHONE': 'phone'
            }
            
            for ent in doc.ents:
                if ent.label_ in entity_mapping:
                    field = ExtractedField(
                        name=entity_mapping[ent.label_],
                        value=ent.text,
                        confidence=0.7,  # Base confidence for NER
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        extraction_method="ner",
                        context=text[max(0, ent.start_char-20):ent.end_char+20]
                    )
                    fields.append(field)
        
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
        
        return fields
    
    def extract_with_ml(self, text: str, field_name: str) -> List[ExtractedField]:
        """
        Extract fields using machine learning models.
        
        Args:
            text: Input text
            field_name: Name of the field to extract
            
        Returns:
            List of extracted fields
        """
        fields = []
        
        if not self.use_ml or field_name not in self.field_models:
            return fields
        
        try:
            # Split text into sentences or chunks
            sentences = re.split(r'[.!?]\s+', text)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 5:  # Skip very short sentences
                    continue
                
                # Prepare features
                features = self.vectorizers[field_name].transform([sentence])
                
                # Predict
                model = self.field_models[field_name]
                prediction = model.predict(features)[0]
                confidence = model.predict_proba(features)[0].max()
                
                if prediction == 1 and confidence > 0.5:  # Positive prediction with good confidence
                    field = ExtractedField(
                        name=field_name,
                        value=sentence.strip(),
                        confidence=confidence,
                        start_pos=text.find(sentence),
                        end_pos=text.find(sentence) + len(sentence),
                        extraction_method="ml",
                        context=sentence
                    )
                    fields.append(field)
        
        except Exception as e:
            logger.error(f"ML extraction failed for {field_name}: {e}")
        
        return fields
    
    def extract_field(self, text: str, field_name: str) -> List[ExtractedField]:
        """
        Extract a specific field using all available methods.
        
        Args:
            text: Input text
            field_name: Name of the field to extract
            
        Returns:
            List of extracted fields
        """
        all_fields = []
        
        # Try pattern-based extraction
        pattern_fields = self.extract_with_patterns(text, field_name)
        all_fields.extend(pattern_fields)
        
        # Try NER extraction
        ner_fields = self.extract_with_ner(text)
        # Filter NER fields by name
        relevant_ner_fields = [f for f in ner_fields if f.name == field_name]
        all_fields.extend(relevant_ner_fields)
        
        # Try ML-based extraction
        ml_fields = self.extract_with_ml(text, field_name)
        all_fields.extend(ml_fields)
        
        # Remove duplicates and merge overlapping extractions
        merged_fields = self._merge_overlapping_fields(all_fields)
        
        return merged_fields
    
    def extract_all_fields(self, text: str, field_names: List[str]) -> Dict[str, List[ExtractedField]]:
        """
        Extract multiple fields from text.
        
        Args:
            text: Input text
            field_names: List of field names to extract
            
        Returns:
            Dictionary mapping field names to extracted fields
        """
        results = {}
        
        for field_name in field_names:
            fields = self.extract_field(text, field_name)
            results[field_name] = fields
        
        return results
    
    def _merge_overlapping_fields(self, fields: List[ExtractedField]) -> List[ExtractedField]:
        """
        Merge overlapping field extractions.
        
        Args:
            fields: List of extracted fields
            
        Returns:
            List of merged fields
        """
        if not fields:
            return []
        
        # Sort by start position
        fields = sorted(fields, key=lambda f: f.start_pos)
        
        merged = []
        current = fields[0]
        
        for field in fields[1:]:
            # Check for overlap
            if field.start_pos <= current.end_pos:
                # Merge fields - keep the one with higher confidence
                if field.confidence > current.confidence:
                    current = field
                else:
                    # Update current field's end position
                    current.end_pos = max(current.end_pos, field.end_pos)
            else:
                # No overlap, add current and start new
                merged.append(current)
                current = field
        
        merged.append(current)
        return merged
    
    def train_field_model(self, field_name: str, training_data: List[Tuple[str, int]]) -> None:
        """
        Train a machine learning model for field extraction.
        
        Args:
            field_name: Name of the field
            training_data: List of (text, label) tuples where label is 1 for positive, 0 for negative
        """
        if not self.use_ml:
            return
        
        try:
            texts, labels = zip(*training_data)
            
            # Create vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Prepare features
            X = vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
            
            # Store model and vectorizer
            self.field_models[field_name] = model
            self.vectorizers[field_name] = vectorizer
            
            logger.info(f"Trained ML model for field: {field_name}")
        
        except Exception as e:
            logger.error(f"Failed to train model for {field_name}: {e}")
    
    def save_models(self, filepath: str) -> None:
        """
        Save trained models and vectorizers.
        
        Args:
            filepath: Path to save the models
        """
        if not self.use_ml:
            return
        
        model_data = {
            'field_models': self.field_models,
            'vectorizers': self.vectorizers,
            'field_patterns': self.field_patterns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Field extraction models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """
        Load trained models and vectorizers.
        
        Args:
            filepath: Path to the saved models
        """
        model_data = joblib.load(filepath)
        
        self.field_models = model_data['field_models']
        self.vectorizers = model_data['vectorizers']
        self.field_patterns.update(model_data['field_patterns'])
        
        logger.info(f"Field extraction models loaded from {filepath}")
    
    def get_extraction_summary(self, text: str, field_names: List[str]) -> Dict[str, Any]:
        """
        Get a summary of field extraction results.
        
        Args:
            text: Input text
            field_names: List of field names to extract
            
        Returns:
            Summary dictionary
        """
        results = self.extract_all_fields(text, field_names)
        
        summary = {
            'total_fields_found': sum(len(fields) for fields in results.values()),
            'fields_by_type': {name: len(fields) for name, fields in results.items()},
            'extraction_methods': {},
            'confidence_scores': {}
        }
        
        for field_name, fields in results.items():
            if fields:
                methods = [f.extraction_method for f in fields]
                confidences = [f.confidence for f in fields]
                
                summary['extraction_methods'][field_name] = {
                    'regex': methods.count('regex'),
                    'ner': methods.count('ner'),
                    'ml': methods.count('ml')
                }
                
                summary['confidence_scores'][field_name] = {
                    'mean': np.mean(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
        
        return summary
