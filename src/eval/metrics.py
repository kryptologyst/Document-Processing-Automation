"""
Evaluation Metrics Module

This module provides comprehensive evaluation metrics for document processing
automation, including accuracy, precision, recall, and business-specific metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FieldMetrics:
    """Metrics for individual field extraction."""
    field_name: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_extracted: int
    total_expected: int
    total_correct: int
    false_positives: int
    false_negatives: int


@dataclass
class DocumentMetrics:
    """Metrics for document-level processing."""
    document_type: str
    overall_accuracy: float
    field_completeness: float
    field_accuracy: float
    processing_time: float
    confidence_score: float
    total_documents: int
    successful_extractions: int
    failed_extractions: int


class DocumentEvaluator:
    """Comprehensive evaluator for document processing systems."""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize document evaluator.
        
        Args:
            tolerance: Tolerance for numerical field comparisons
        """
        self.tolerance = tolerance
        self.metrics_history = []
    
    def evaluate_field_extraction(self, predicted_fields: List[Dict[str, Any]], 
                                ground_truth_fields: List[Dict[str, Any]]) -> Dict[str, FieldMetrics]:
        """
        Evaluate field extraction performance.
        
        Args:
            predicted_fields: List of predicted field dictionaries
            ground_truth_fields: List of ground truth field dictionaries
            
        Returns:
            Dictionary mapping field names to FieldMetrics
        """
        # Convert to dictionaries for easier lookup
        pred_dict = {field['name']: field for field in predicted_fields}
        gt_dict = {field['name']: field for field in ground_truth_fields}
        
        # Get all unique field names
        all_fields = set(pred_dict.keys()) | set(gt_dict.keys())
        
        field_metrics = {}
        
        for field_name in all_fields:
            pred_field = pred_dict.get(field_name)
            gt_field = gt_dict.get(field_name)
            
            # Calculate metrics for this field
            metrics = self._calculate_field_metrics(field_name, pred_field, gt_field)
            field_metrics[field_name] = metrics
        
        return field_metrics
    
    def _calculate_field_metrics(self, field_name: str, pred_field: Optional[Dict[str, Any]], 
                               gt_field: Optional[Dict[str, Any]]) -> FieldMetrics:
        """Calculate metrics for a single field."""
        # Handle different cases
        if pred_field is None and gt_field is None:
            # Both missing - perfect match
            return FieldMetrics(
                field_name=field_name,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                accuracy=1.0,
                total_extracted=0,
                total_expected=0,
                total_correct=0,
                false_positives=0,
                false_negatives=0
            )
        
        elif pred_field is None and gt_field is not None:
            # False negative - field expected but not extracted
            return FieldMetrics(
                field_name=field_name,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                total_extracted=0,
                total_expected=1,
                total_correct=0,
                false_positives=0,
                false_negatives=1
            )
        
        elif pred_field is not None and gt_field is None:
            # False positive - field extracted but not expected
            return FieldMetrics(
                field_name=field_name,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                total_extracted=1,
                total_expected=0,
                total_correct=0,
                false_positives=1,
                false_negatives=0
            )
        
        else:
            # Both present - check if values match
            pred_value = pred_field['value']
            gt_value = gt_field['value']
            
            is_correct = self._compare_field_values(pred_value, gt_value, field_name)
            
            if is_correct:
                return FieldMetrics(
                    field_name=field_name,
                    precision=1.0,
                    recall=1.0,
                    f1_score=1.0,
                    accuracy=1.0,
                    total_extracted=1,
                    total_expected=1,
                    total_correct=1,
                    false_positives=0,
                    false_negatives=0
                )
            else:
                return FieldMetrics(
                    field_name=field_name,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    accuracy=0.0,
                    total_extracted=1,
                    total_expected=1,
                    total_correct=0,
                    false_positives=1,
                    false_negatives=1
                )
    
    def _compare_field_values(self, pred_value: Any, gt_value: Any, field_name: str) -> bool:
        """Compare predicted and ground truth field values."""
        # Handle None values
        if pred_value is None and gt_value is None:
            return True
        if pred_value is None or gt_value is None:
            return False
        
        # Convert to strings for comparison
        pred_str = str(pred_value).strip().lower()
        gt_str = str(gt_value).strip().lower()
        
        # Exact match
        if pred_str == gt_str:
            return True
        
        # Numerical comparison with tolerance
        if self._is_numerical_field(field_name):
            try:
                pred_num = float(pred_str.replace(',', '').replace('$', ''))
                gt_num = float(gt_str.replace(',', '').replace('$', ''))
                return abs(pred_num - gt_num) <= self.tolerance
            except (ValueError, TypeError):
                pass
        
        # Date comparison
        if self._is_date_field(field_name):
            return self._compare_dates(pred_str, gt_str)
        
        # Fuzzy string matching for text fields
        if self._is_text_field(field_name):
            return self._fuzzy_string_match(pred_str, gt_str)
        
        return False
    
    def _is_numerical_field(self, field_name: str) -> bool:
        """Check if field is numerical."""
        numerical_fields = ['total', 'amount', 'subtotal', 'tax', 'value', 'price', 'cost']
        return any(num_field in field_name.lower() for num_field in numerical_fields)
    
    def _is_date_field(self, field_name: str) -> bool:
        """Check if field is a date field."""
        date_fields = ['date', 'due_date', 'created_date', 'modified_date']
        return any(date_field in field_name.lower() for date_field in date_fields)
    
    def _is_text_field(self, field_name: str) -> bool:
        """Check if field is a text field."""
        text_fields = ['customer', 'merchant', 'parties', 'items', 'description']
        return any(text_field in field_name.lower() for text_field in text_fields)
    
    def _compare_dates(self, pred_date: str, gt_date: str) -> bool:
        """Compare date strings with various formats."""
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in date_formats:
            try:
                pred_dt = datetime.strptime(pred_date, fmt)
                gt_dt = datetime.strptime(gt_date, fmt)
                return pred_dt.date() == gt_dt.date()
            except ValueError:
                continue
        
        return False
    
    def _fuzzy_string_match(self, pred_str: str, gt_str: str, threshold: float = 0.8) -> bool:
        """Perform fuzzy string matching."""
        # Simple implementation - can be enhanced with more sophisticated algorithms
        if len(pred_str) == 0 or len(gt_str) == 0:
            return pred_str == gt_str
        
        # Calculate Jaccard similarity
        pred_words = set(pred_str.split())
        gt_words = set(gt_str.split())
        
        if len(pred_words) == 0 and len(gt_words) == 0:
            return True
        
        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
    
    def evaluate_document_classification(self, predicted_types: List[str], 
                                       ground_truth_types: List[str]) -> Dict[str, float]:
        """
        Evaluate document type classification performance.
        
        Args:
            predicted_types: List of predicted document types
            ground_truth_types: List of ground truth document types
            
        Returns:
            Dictionary of classification metrics
        """
        if len(predicted_types) != len(ground_truth_types):
            raise ValueError("Predicted and ground truth lists must have the same length")
        
        # Calculate basic metrics
        accuracy = accuracy_score(ground_truth_types, predicted_types)
        
        # Calculate precision, recall, F1 for each class
        unique_types = list(set(ground_truth_types + predicted_types))
        
        precision = precision_score(ground_truth_types, predicted_types, 
                                  labels=unique_types, average='weighted', zero_division=0)
        recall = recall_score(ground_truth_types, predicted_types, 
                            labels=unique_types, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth_types, predicted_types, 
                     labels=unique_types, average='weighted', zero_division=0)
        
        # Detailed classification report
        classification_report_dict = classification_report(
            ground_truth_types, predicted_types, 
            labels=unique_types, output_dict=True, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report_dict,
            'confusion_matrix': confusion_matrix(ground_truth_types, predicted_types, 
                                               labels=unique_types).tolist()
        }
    
    def evaluate_confidence_calibration(self, confidence_scores: List[float], 
                                      correct_predictions: List[bool]) -> Dict[str, float]:
        """
        Evaluate confidence calibration.
        
        Args:
            confidence_scores: List of confidence scores
            correct_predictions: List of boolean values indicating correctness
            
        Returns:
            Dictionary of calibration metrics
        """
        if len(confidence_scores) != len(correct_predictions):
            raise ValueError("Confidence scores and correctness lists must have the same length")
        
        # Convert to numpy arrays
        confidences = np.array(confidence_scores)
        correct = np.array(correct_predictions)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = np.mean((confidences - correct) ** 2)
        
        # Reliability diagram data
        reliability_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                count_in_bin = in_bin.sum()
                reliability_data.append({
                    'bin_center': (bin_lower + bin_upper) / 2,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'count': count_in_bin
                })
        
        return {
            'ece': ece,
            'brier_score': brier_score,
            'reliability_data': reliability_data
        }
    
    def evaluate_processing_performance(self, processing_times: List[float], 
                                      document_sizes: List[int]) -> Dict[str, float]:
        """
        Evaluate processing performance metrics.
        
        Args:
            processing_times: List of processing times in seconds
            document_sizes: List of document sizes (e.g., character count)
            
        Returns:
            Dictionary of performance metrics
        """
        if len(processing_times) != len(document_sizes):
            raise ValueError("Processing times and document sizes must have the same length")
        
        processing_times = np.array(processing_times)
        document_sizes = np.array(document_sizes)
        
        # Basic statistics
        avg_processing_time = np.mean(processing_times)
        median_processing_time = np.median(processing_times)
        std_processing_time = np.std(processing_times)
        
        # Throughput metrics
        documents_per_second = 1.0 / avg_processing_time
        characters_per_second = np.mean(document_sizes / processing_times)
        
        # Performance by document size
        size_performance = {}
        size_bins = np.percentile(document_sizes, [25, 50, 75])
        
        for i, (lower, upper) in enumerate(zip([0] + size_bins.tolist(), size_bins.tolist() + [np.inf])):
            mask = (document_sizes >= lower) & (document_sizes < upper)
            if mask.sum() > 0:
                size_performance[f'bin_{i+1}'] = {
                    'size_range': f'{lower:.0f}-{upper:.0f}',
                    'avg_time': float(np.mean(processing_times[mask])),
                    'documents': int(mask.sum())
                }
        
        return {
            'avg_processing_time': avg_processing_time,
            'median_processing_time': median_processing_time,
            'std_processing_time': std_processing_time,
            'documents_per_second': documents_per_second,
            'characters_per_second': characters_per_second,
            'size_performance': size_performance
        }
    
    def create_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            evaluation_results: Dictionary containing all evaluation results
            
        Returns:
            Formatted evaluation report string
        """
        report = []
        report.append("=" * 80)
        report.append("DOCUMENT PROCESSING AUTOMATION - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Document classification results
        if 'classification' in evaluation_results:
            cls_results = evaluation_results['classification']
            report.append("DOCUMENT CLASSIFICATION METRICS:")
            report.append("-" * 40)
            report.append(f"Accuracy: {cls_results['accuracy']:.4f}")
            report.append(f"Precision: {cls_results['precision']:.4f}")
            report.append(f"Recall: {cls_results['recall']:.4f}")
            report.append(f"F1-Score: {cls_results['f1_score']:.4f}")
            report.append("")
        
        # Field extraction results
        if 'field_extraction' in evaluation_results:
            field_results = evaluation_results['field_extraction']
            report.append("FIELD EXTRACTION METRICS:")
            report.append("-" * 40)
            
            for field_name, metrics in field_results.items():
                report.append(f"{field_name.upper()}:")
                report.append(f"  Precision: {metrics.precision:.4f}")
                report.append(f"  Recall: {metrics.recall:.4f}")
                report.append(f"  F1-Score: {metrics.f1_score:.4f}")
                report.append(f"  Accuracy: {metrics.accuracy:.4f}")
                report.append(f"  Total Extracted: {metrics.total_extracted}")
                report.append(f"  Total Expected: {metrics.total_expected}")
                report.append(f"  Correct: {metrics.total_correct}")
                report.append("")
        
        # Confidence calibration results
        if 'confidence_calibration' in evaluation_results:
            cal_results = evaluation_results['confidence_calibration']
            report.append("CONFIDENCE CALIBRATION METRICS:")
            report.append("-" * 40)
            report.append(f"Expected Calibration Error (ECE): {cal_results['ece']:.4f}")
            report.append(f"Brier Score: {cal_results['brier_score']:.4f}")
            report.append("")
        
        # Processing performance results
        if 'processing_performance' in evaluation_results:
            perf_results = evaluation_results['processing_performance']
            report.append("PROCESSING PERFORMANCE METRICS:")
            report.append("-" * 40)
            report.append(f"Average Processing Time: {perf_results['avg_processing_time']:.4f}s")
            report.append(f"Median Processing Time: {perf_results['median_processing_time']:.4f}s")
            report.append(f"Documents per Second: {perf_results['documents_per_second']:.2f}")
            report.append(f"Characters per Second: {perf_results['characters_per_second']:.0f}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append("This evaluation report provides comprehensive metrics for")
        report.append("document processing automation performance across multiple")
        report.append("dimensions including accuracy, speed, and reliability.")
        report.append("")
        report.append("For production deployment, consider:")
        report.append("- Setting appropriate confidence thresholds")
        report.append("- Implementing human-in-loop validation for low-confidence cases")
        report.append("- Monitoring performance metrics over time")
        report.append("- Regular model retraining with new data")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any], filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            filepath: Path to save the results
        """
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = self._make_serializable(evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclasses and other non-serializable objects to dictionaries."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
