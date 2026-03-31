"""
Explainability Engine Module

This module provides explainability features for document processing
automation, including confidence scoring, feature importance, and
human-in-loop validation capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Represents an explanation for a document processing decision."""
    document_id: str
    decision: str
    confidence: float
    factors: Dict[str, float]
    explanation_text: str
    recommendations: List[str]
    requires_human_review: bool


@dataclass
class HumanReviewRequest:
    """Represents a request for human review."""
    document_id: str
    document_type: str
    extracted_fields: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    explanation: str
    priority: str  # low, medium, high
    created_at: datetime


class ExplainabilityEngine:
    """Engine for providing explainable document processing decisions."""
    
    def __init__(self, confidence_threshold: float = 0.7, 
                 human_review_threshold: float = 0.5):
        """
        Initialize explainability engine.
        
        Args:
            confidence_threshold: Threshold for automatic processing
            human_review_threshold: Threshold for human review requirement
        """
        self.confidence_threshold = confidence_threshold
        self.human_review_threshold = human_review_threshold
        self.explanation_templates = self._load_explanation_templates()
        self.review_requests = []
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different scenarios."""
        return {
            'high_confidence': "Document processed with high confidence ({confidence:.2f}). All key fields extracted successfully.",
            'medium_confidence': "Document processed with moderate confidence ({confidence:.2f}). Some fields may require verification.",
            'low_confidence': "Document processed with low confidence ({confidence:.2f}). Manual review recommended.",
            'field_missing': "Field '{field_name}' not extracted. Pattern matching failed or field not present in document.",
            'field_ambiguous': "Field '{field_name}' extracted with low confidence. Multiple possible values detected.",
            'document_type_uncertain': "Document type classification uncertain. Multiple types possible.",
            'processing_error': "Processing error occurred: {error_message}",
            'human_review_required': "Human review required due to low confidence or ambiguous results."
        }
    
    def explain_document_processing(self, document_result: Dict[str, Any]) -> ExplanationResult:
        """
        Generate explanation for document processing result.
        
        Args:
            document_result: Document processing result dictionary
            
        Returns:
            ExplanationResult object
        """
        document_id = document_result.get('document_id', 'unknown')
        confidence = document_result.get('confidence', 0.0)
        document_type = document_result.get('document_type', 'unknown')
        fields = document_result.get('fields', [])
        
        # Analyze factors contributing to confidence
        factors = self._analyze_confidence_factors(document_result)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(confidence, factors, fields)
        
        # Determine if human review is required
        requires_human_review = self._requires_human_review(confidence, factors, fields)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(confidence, factors, fields)
        
        # Determine decision
        decision = self._determine_decision(confidence, requires_human_review)
        
        return ExplanationResult(
            document_id=document_id,
            decision=decision,
            confidence=confidence,
            factors=factors,
            explanation_text=explanation_text,
            recommendations=recommendations,
            requires_human_review=requires_human_review
        )
    
    def _analyze_confidence_factors(self, document_result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze factors contributing to confidence score."""
        factors = {}
        
        # Document type confidence
        doc_type_confidence = document_result.get('document_type_confidence', 0.8)
        factors['document_type_confidence'] = doc_type_confidence
        
        # Field extraction confidence
        fields = document_result.get('fields', [])
        if fields:
            field_confidences = [field.get('confidence', 0.0) for field in fields]
            factors['field_extraction_confidence'] = np.mean(field_confidences)
            factors['field_completeness'] = len(fields) / 5.0  # Assume 5 expected fields
        else:
            factors['field_extraction_confidence'] = 0.0
            factors['field_completeness'] = 0.0
        
        # Text quality factors
        text = document_result.get('text', '')
        factors['text_length'] = min(len(text) / 1000.0, 1.0)  # Normalize to 0-1
        factors['text_quality'] = self._assess_text_quality(text)
        
        # Pattern matching quality
        factors['pattern_matching_quality'] = self._assess_pattern_matching_quality(fields)
        
        # Overall confidence calculation
        factors['overall_confidence'] = np.mean(list(factors.values()))
        
        return factors
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality for processing."""
        if not text:
            return 0.0
        
        # Check for common text quality indicators
        quality_score = 1.0
        
        # Check for OCR artifacts
        if any(char in text for char in ['', '?', '??']):
            quality_score -= 0.3
        
        # Check for excessive whitespace
        if len(text.split()) < len(text) * 0.1:
            quality_score -= 0.2
        
        # Check for reasonable length
        if len(text) < 50:
            quality_score -= 0.2
        elif len(text) > 10000:
            quality_score -= 0.1
        
        return max(quality_score, 0.0)
    
    def _assess_pattern_matching_quality(self, fields: List[Dict[str, Any]]) -> float:
        """Assess quality of pattern matching results."""
        if not fields:
            return 0.0
        
        # Check for fields with high confidence
        high_confidence_fields = sum(1 for field in fields if field.get('confidence', 0) > 0.8)
        total_fields = len(fields)
        
        return high_confidence_fields / total_fields if total_fields > 0 else 0.0
    
    def _generate_explanation_text(self, confidence: float, factors: Dict[str, float], 
                                 fields: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation text."""
        explanations = []
        
        # Overall confidence explanation
        if confidence >= 0.8:
            explanations.append(self.explanation_templates['high_confidence'].format(confidence=confidence))
        elif confidence >= 0.6:
            explanations.append(self.explanation_templates['medium_confidence'].format(confidence=confidence))
        else:
            explanations.append(self.explanation_templates['low_confidence'].format(confidence=confidence))
        
        # Field-specific explanations
        for field in fields:
            field_name = field.get('name', 'unknown')
            field_confidence = field.get('confidence', 0.0)
            
            if field_confidence < 0.5:
                explanations.append(self.explanation_templates['field_ambiguous'].format(field_name=field_name))
        
        # Factor-based explanations
        if factors.get('text_quality', 1.0) < 0.7:
            explanations.append("Text quality may affect extraction accuracy.")
        
        if factors.get('pattern_matching_quality', 1.0) < 0.7:
            explanations.append("Pattern matching quality is below optimal.")
        
        return " ".join(explanations)
    
    def _requires_human_review(self, confidence: float, factors: Dict[str, float], 
                             fields: List[Dict[str, Any]]) -> bool:
        """Determine if human review is required."""
        # Low overall confidence
        if confidence < self.human_review_threshold:
            return True
        
        # Low field extraction confidence
        if factors.get('field_extraction_confidence', 1.0) < 0.6:
            return True
        
        # Missing critical fields
        critical_fields = ['date', 'total', 'customer', 'invoice_number']
        extracted_field_names = [field.get('name', '') for field in fields]
        missing_critical = sum(1 for field in critical_fields if field not in extracted_field_names)
        
        if missing_critical > 1:
            return True
        
        # Low text quality
        if factors.get('text_quality', 1.0) < 0.5:
            return True
        
        return False
    
    def _generate_recommendations(self, confidence: float, factors: Dict[str, float], 
                                fields: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Consider manual review of extracted fields")
            recommendations.append("Verify document type classification")
        
        # Field-specific recommendations
        low_confidence_fields = [field for field in fields if field.get('confidence', 0) < 0.6]
        if low_confidence_fields:
            recommendations.append("Review low-confidence field extractions")
        
        # Text quality recommendations
        if factors.get('text_quality', 1.0) < 0.7:
            recommendations.append("Improve document image quality or OCR preprocessing")
        
        # Pattern matching recommendations
        if factors.get('pattern_matching_quality', 1.0) < 0.7:
            recommendations.append("Refine extraction patterns for better accuracy")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Processing completed successfully")
        
        return recommendations
    
    def _determine_decision(self, confidence: float, requires_human_review: bool) -> str:
        """Determine the processing decision."""
        if requires_human_review:
            return "human_review_required"
        elif confidence >= self.confidence_threshold:
            return "auto_approved"
        else:
            return "auto_rejected"
    
    def create_human_review_request(self, document_result: Dict[str, Any], 
                                  explanation: ExplanationResult) -> HumanReviewRequest:
        """Create a human review request."""
        # Determine priority
        priority = "high" if explanation.confidence < 0.3 else "medium" if explanation.confidence < 0.6 else "low"
        
        # Extract field information
        fields = document_result.get('fields', [])
        extracted_fields = [{'name': f.get('name', ''), 'value': f.get('value', ''), 'confidence': f.get('confidence', 0.0)} for f in fields]
        
        # Create confidence scores dictionary
        confidence_scores = {field['name']: field['confidence'] for field in extracted_fields}
        
        review_request = HumanReviewRequest(
            document_id=document_result.get('document_id', 'unknown'),
            document_type=document_result.get('document_type', 'unknown'),
            extracted_fields=extracted_fields,
            confidence_scores=confidence_scores,
            explanation=explanation.explanation_text,
            priority=priority,
            created_at=datetime.now()
        )
        
        self.review_requests.append(review_request)
        return review_request
    
    def get_review_requests(self, priority: Optional[str] = None) -> List[HumanReviewRequest]:
        """Get human review requests, optionally filtered by priority."""
        if priority:
            return [req for req in self.review_requests if req.priority == priority]
        return self.review_requests.copy()
    
    def resolve_review_request(self, document_id: str, human_feedback: Dict[str, Any]) -> bool:
        """Resolve a human review request with feedback."""
        for i, request in enumerate(self.review_requests):
            if request.document_id == document_id:
                # Update request with human feedback
                request.human_feedback = human_feedback
                request.resolved_at = datetime.now()
                request.status = "resolved"
                
                # Remove from active requests
                self.review_requests.pop(i)
                return True
        
        return False
    
    def generate_explainability_report(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate a comprehensive explainability report."""
        if not explanations:
            return {}
        
        # Calculate statistics
        total_docs = len(explanations)
        auto_approved = sum(1 for exp in explanations if exp.decision == "auto_approved")
        auto_rejected = sum(1 for exp in explanations if exp.decision == "auto_rejected")
        human_review_required = sum(1 for exp in explanations if exp.decision == "human_review_required")
        
        # Confidence statistics
        confidences = [exp.confidence for exp in explanations]
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Factor analysis
        all_factors = {}
        for exp in explanations:
            for factor, value in exp.factors.items():
                if factor not in all_factors:
                    all_factors[factor] = []
                all_factors[factor].append(value)
        
        factor_stats = {}
        for factor, values in all_factors.items():
            factor_stats[factor] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Common recommendations
        all_recommendations = []
        for exp in explanations:
            all_recommendations.extend(exp.recommendations)
        
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        common_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_documents': total_docs,
            'decision_distribution': {
                'auto_approved': auto_approved,
                'auto_rejected': auto_rejected,
                'human_review_required': human_review_required
            },
            'confidence_statistics': {
                'mean': avg_confidence,
                'std': std_confidence,
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'factor_statistics': factor_stats,
            'common_recommendations': common_recommendations[:10],  # Top 10
            'human_review_rate': human_review_required / total_docs if total_docs > 0 else 0
        }
    
    def save_explainability_data(self, explanations: List[ExplanationResult], 
                               filepath: str) -> None:
        """Save explainability data to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = []
        for exp in explanations:
            data.append({
                'document_id': exp.document_id,
                'decision': exp.decision,
                'confidence': exp.confidence,
                'factors': exp.factors,
                'explanation_text': exp.explanation_text,
                'recommendations': exp.recommendations,
                'requires_human_review': exp.requires_human_review
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Explainability data saved to {filepath}")
    
    def load_explainability_data(self, filepath: str) -> List[ExplanationResult]:
        """Load explainability data from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        explanations = []
        for item in data:
            exp = ExplanationResult(
                document_id=item['document_id'],
                decision=item['decision'],
                confidence=item['confidence'],
                factors=item['factors'],
                explanation_text=item['explanation_text'],
                recommendations=item['recommendations'],
                requires_human_review=item['requires_human_review']
            )
            explanations.append(exp)
        
        logger.info(f"Explainability data loaded from {filepath}")
        return explanations
