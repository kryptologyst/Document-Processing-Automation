"""
Confidence Estimation Module

This module provides confidence estimation for document processing
results to enable human-in-loop validation and quality control.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Represents a confidence score with metadata."""
    value: float
    method: str
    factors: Dict[str, float]
    explanation: str


class ConfidenceEstimator:
    """Estimates confidence scores for document processing results."""
    
    def __init__(self, anomaly_threshold: float = 0.1):
        """
        Initialize confidence estimator.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection
        """
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_detector = IsolationForest(contamination=anomaly_threshold, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Confidence factors and their weights
        self.factor_weights = {
            'pattern_match_quality': 0.3,
            'field_completeness': 0.2,
            'consistency_check': 0.2,
            'anomaly_score': 0.15,
            'context_relevance': 0.15
        }
    
    def estimate_pattern_match_quality(self, text: str, pattern: str, match: str) -> float:
        """
        Estimate quality of pattern matching.
        
        Args:
            text: Original text
            pattern: Regex pattern used
            match: Matched text
            
        Returns:
            Quality score between 0 and 1
        """
        if not match:
            return 0.0
        
        # Factor 1: Match length relative to text length
        length_ratio = len(match) / len(text) if text else 0
        length_score = min(length_ratio * 10, 1.0)  # Normalize
        
        # Factor 2: Pattern complexity (more specific patterns get higher scores)
        pattern_complexity = len(pattern) / 50.0  # Normalize
        complexity_score = min(pattern_complexity, 1.0)
        
        # Factor 3: Match position (earlier matches might be more relevant)
        position = text.find(match) if match in text else len(text)
        position_score = 1.0 - (position / len(text)) if text else 0.0
        
        # Combine factors
        quality_score = (length_score * 0.4 + complexity_score * 0.3 + position_score * 0.3)
        
        return min(quality_score, 1.0)
    
    def estimate_field_completeness(self, extracted_fields: List[Dict[str, Any]], 
                                  expected_fields: List[str]) -> float:
        """
        Estimate completeness of field extraction.
        
        Args:
            extracted_fields: List of extracted field dictionaries
            expected_fields: List of expected field names
            
        Returns:
            Completeness score between 0 and 1
        """
        if not expected_fields:
            return 1.0
        
        extracted_field_names = {field.get('name', '') for field in extracted_fields}
        expected_field_names = set(expected_fields)
        
        # Calculate coverage
        coverage = len(extracted_field_names.intersection(expected_field_names)) / len(expected_field_names)
        
        # Penalize for unexpected fields
        unexpected_ratio = len(extracted_field_names - expected_field_names) / len(expected_field_names)
        unexpected_penalty = min(unexpected_ratio * 0.2, 0.2)
        
        completeness_score = coverage - unexpected_penalty
        
        return max(completeness_score, 0.0)
    
    def estimate_consistency_check(self, extracted_fields: List[Dict[str, Any]]) -> float:
        """
        Estimate consistency of extracted fields.
        
        Args:
            extracted_fields: List of extracted field dictionaries
            
        Returns:
            Consistency score between 0 and 1
        """
        if not extracted_fields:
            return 0.0
        
        consistency_scores = []
        
        # Check date consistency
        dates = [field['value'] for field in extracted_fields if field.get('name') == 'date']
        if len(dates) > 1:
            # Simple date consistency check
            date_consistency = 1.0 if len(set(dates)) == 1 else 0.5
            consistency_scores.append(date_consistency)
        
        # Check amount consistency
        amounts = [field['value'] for field in extracted_fields if field.get('name') in ['total', 'amount']]
        if len(amounts) > 1:
            # Check if amounts are reasonable relative to each other
            try:
                amount_values = [float(str(amt).replace('$', '').replace(',', '')) for amt in amounts]
                if len(amount_values) > 1:
                    amount_ratio = min(amount_values) / max(amount_values)
                    amount_consistency = 1.0 if amount_ratio > 0.1 else 0.5
                    consistency_scores.append(amount_consistency)
            except (ValueError, TypeError):
                consistency_scores.append(0.5)
        
        # Check field name consistency
        field_names = [field.get('name', '') for field in extracted_fields]
        unique_names = len(set(field_names))
        total_fields = len(field_names)
        name_consistency = 1.0 if unique_names == total_fields else 0.8
        
        consistency_scores.append(name_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def estimate_anomaly_score(self, features: np.ndarray) -> float:
        """
        Estimate anomaly score using isolation forest.
        
        Args:
            features: Feature vector for anomaly detection
            
        Returns:
            Anomaly score between 0 and 1 (higher = more normal)
        """
        if not self.is_fitted:
            # Fit on dummy data if not fitted
            dummy_features = np.random.randn(100, features.shape[1])
            self.anomaly_detector.fit(dummy_features)
            self.is_fitted = True
        
        try:
            # Get anomaly score (higher = more normal)
            anomaly_score = self.anomaly_detector.decision_function([features])[0]
            
            # Normalize to 0-1 range
            normalized_score = (anomaly_score + 1) / 2
            return max(0.0, min(1.0, normalized_score))
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return 0.5
    
    def estimate_context_relevance(self, text: str, extracted_fields: List[Dict[str, Any]]) -> float:
        """
        Estimate relevance of extracted fields to document context.
        
        Args:
            text: Original document text
            extracted_fields: List of extracted field dictionaries
            
        Returns:
            Relevance score between 0 and 1
        """
        if not extracted_fields or not text:
            return 0.0
        
        relevance_scores = []
        
        for field in extracted_fields:
            field_value = str(field.get('value', ''))
            field_name = field.get('name', '')
            
            # Check if field value appears in text
            if field_value in text:
                relevance_scores.append(1.0)
            else:
                # Check for partial matches
                partial_match = any(word in text.lower() for word in field_value.lower().split())
                relevance_scores.append(0.7 if partial_match else 0.3)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def estimate_overall_confidence(self, text: str, extracted_fields: List[Dict[str, Any]], 
                                  expected_fields: List[str] = None) -> ConfidenceScore:
        """
        Estimate overall confidence score for document processing results.
        
        Args:
            text: Original document text
            extracted_fields: List of extracted field dictionaries
            expected_fields: List of expected field names
            
        Returns:
            ConfidenceScore object
        """
        if expected_fields is None:
            expected_fields = ['date', 'total', 'customer', 'invoice_number']
        
        # Calculate individual factor scores
        factors = {}
        
        # Pattern match quality (average across fields)
        pattern_scores = []
        for field in extracted_fields:
            if 'pattern' in field.get('extraction_method', ''):
                score = self.estimate_pattern_match_quality(
                    text, field.get('pattern', ''), str(field.get('value', ''))
                )
                pattern_scores.append(score)
        
        factors['pattern_match_quality'] = np.mean(pattern_scores) if pattern_scores else 0.5
        
        # Field completeness
        factors['field_completeness'] = self.estimate_field_completeness(extracted_fields, expected_fields)
        
        # Consistency check
        factors['consistency_check'] = self.estimate_consistency_check(extracted_fields)
        
        # Context relevance
        factors['context_relevance'] = self.estimate_context_relevance(text, extracted_fields)
        
        # Anomaly score (using simple features)
        if extracted_fields:
            # Create simple feature vector
            features = np.array([
                len(extracted_fields),
                len(text),
                factors['field_completeness'],
                factors['consistency_check']
            ])
            factors['anomaly_score'] = self.estimate_anomaly_score(features)
        else:
            factors['anomaly_score'] = 0.0
        
        # Calculate weighted overall confidence
        overall_confidence = sum(
            factors[factor] * weight 
            for factor, weight in self.factor_weights.items()
        )
        
        # Generate explanation
        explanation = self._generate_confidence_explanation(factors, overall_confidence)
        
        return ConfidenceScore(
            value=overall_confidence,
            method="weighted_ensemble",
            factors=factors,
            explanation=explanation
        )
    
    def _generate_confidence_explanation(self, factors: Dict[str, float], 
                                       overall_confidence: float) -> str:
        """
        Generate human-readable explanation of confidence score.
        
        Args:
            factors: Dictionary of factor scores
            overall_confidence: Overall confidence score
            
        Returns:
            Explanation string
        """
        explanations = []
        
        if factors['pattern_match_quality'] > 0.8:
            explanations.append("High-quality pattern matches")
        elif factors['pattern_match_quality'] < 0.4:
            explanations.append("Low-quality pattern matches")
        
        if factors['field_completeness'] > 0.8:
            explanations.append("Complete field extraction")
        elif factors['field_completeness'] < 0.4:
            explanations.append("Incomplete field extraction")
        
        if factors['consistency_check'] > 0.8:
            explanations.append("Consistent field values")
        elif factors['consistency_check'] < 0.4:
            explanations.append("Inconsistent field values")
        
        if factors['context_relevance'] > 0.8:
            explanations.append("High context relevance")
        elif factors['context_relevance'] < 0.4:
            explanations.append("Low context relevance")
        
        if factors['anomaly_score'] < 0.3:
            explanations.append("Anomalous document structure")
        
        if not explanations:
            explanations.append("Average extraction quality")
        
        explanation = f"Confidence: {overall_confidence:.2f}. " + "; ".join(explanations)
        
        return explanation
    
    def batch_estimate_confidence(self, texts: List[str], 
                                extracted_fields_list: List[List[Dict[str, Any]]],
                                expected_fields_list: List[List[str]] = None) -> List[ConfidenceScore]:
        """
        Estimate confidence scores for multiple documents.
        
        Args:
            texts: List of document texts
            extracted_fields_list: List of extracted field lists
            expected_fields_list: List of expected field lists
            
        Returns:
            List of ConfidenceScore objects
        """
        if expected_fields_list is None:
            expected_fields_list = [None] * len(texts)
        
        confidence_scores = []
        
        for text, fields, expected in zip(texts, extracted_fields_list, expected_fields_list):
            score = self.estimate_overall_confidence(text, fields, expected)
            confidence_scores.append(score)
        
        return confidence_scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save the confidence estimation model.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'factor_weights': self.factor_weights,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Confidence estimation model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load the confidence estimation model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.anomaly_detector = model_data['anomaly_detector']
        self.scaler = model_data['scaler']
        self.factor_weights = model_data['factor_weights']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Confidence estimation model loaded from {filepath}")
    
    def get_confidence_threshold_recommendation(self, confidence_scores: List[float]) -> float:
        """
        Recommend confidence threshold based on historical scores.
        
        Args:
            confidence_scores: List of historical confidence scores
            
        Returns:
            Recommended threshold
        """
        if not confidence_scores:
            return 0.7  # Default threshold
        
        # Use 25th percentile as threshold (capture 75% of documents)
        threshold = np.percentile(confidence_scores, 25)
        
        # Ensure threshold is reasonable
        threshold = max(0.3, min(0.9, threshold))
        
        return threshold
