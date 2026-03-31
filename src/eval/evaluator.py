"""
Document Evaluator Module

This module provides the main evaluator class for comprehensive
document processing system evaluation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from .metrics import DocumentEvaluator, FieldMetrics, DocumentMetrics
from ..data.schema import DocumentType, get_schema

logger = logging.getLogger(__name__)


class DocumentProcessingEvaluator:
    """Main evaluator for document processing automation systems."""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize the document processing evaluator.
        
        Args:
            tolerance: Tolerance for numerical field comparisons
        """
        self.tolerance = tolerance
        self.metrics_calculator = DocumentEvaluator(tolerance)
        self.evaluation_history = []
    
    def evaluate_system(self, predictions: List[Dict[str, Any]], 
                       ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the complete document processing system.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            
        Returns:
            Comprehensive evaluation results
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        logger.info(f"Evaluating system on {len(predictions)} documents")
        
        # Extract components for evaluation
        pred_types = [pred.get('document_type', 'unknown') for pred in predictions]
        gt_types = [gt.get('document_type', 'unknown') for gt in ground_truth]
        
        pred_fields = [pred.get('fields', []) for pred in predictions]
        gt_fields = [gt.get('fields', []) for gt in ground_truth]
        
        pred_confidences = [pred.get('confidence', 0.0) for pred in predictions]
        pred_times = [pred.get('processing_time', 0.0) for pred in predictions]
        doc_sizes = [len(pred.get('text', '')) for pred in predictions]
        
        # Evaluate document classification
        classification_results = self.metrics_calculator.evaluate_document_classification(
            pred_types, gt_types
        )
        
        # Evaluate field extraction
        field_extraction_results = self._evaluate_field_extraction_batch(
            pred_fields, gt_fields
        )
        
        # Evaluate confidence calibration
        correct_predictions = self._calculate_correctness(predictions, ground_truth)
        confidence_calibration = self.metrics_calculator.evaluate_confidence_calibration(
            pred_confidences, correct_predictions
        )
        
        # Evaluate processing performance
        processing_performance = self.metrics_calculator.evaluate_processing_performance(
            pred_times, doc_sizes
        )
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(
            predictions, ground_truth, correct_predictions
        )
        
        # Compile results
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_documents': len(predictions),
            'classification': classification_results,
            'field_extraction': field_extraction_results,
            'confidence_calibration': confidence_calibration,
            'processing_performance': processing_performance,
            'business_metrics': business_metrics,
            'summary': self._create_summary(
                classification_results, field_extraction_results, 
                confidence_calibration, processing_performance, business_metrics
            )
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_field_extraction_batch(self, pred_fields_list: List[List[Dict[str, Any]]], 
                                       gt_fields_list: List[List[Dict[str, Any]]]) -> Dict[str, FieldMetrics]:
        """Evaluate field extraction across all documents."""
        # Aggregate all fields across documents
        all_pred_fields = []
        all_gt_fields = []
        
        for pred_fields, gt_fields in zip(pred_fields_list, gt_fields_list):
            all_pred_fields.extend(pred_fields)
            all_gt_fields.extend(gt_fields)
        
        # Calculate metrics for each field type
        field_metrics = self.metrics_calculator.evaluate_field_extraction(
            all_pred_fields, all_gt_fields
        )
        
        return field_metrics
    
    def _calculate_correctness(self, predictions: List[Dict[str, Any]], 
                             ground_truth: List[Dict[str, Any]]) -> List[bool]:
        """Calculate correctness for each document."""
        correct_predictions = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Check document type correctness
            type_correct = pred.get('document_type') == gt.get('document_type')
            
            # Check field extraction correctness
            pred_fields = {f['name']: f['value'] for f in pred.get('fields', [])}
            gt_fields = {f['name']: f['value'] for f in gt.get('fields', [])}
            
            # Calculate field accuracy
            field_accuracy = self._calculate_field_accuracy(pred_fields, gt_fields)
            
            # Overall correctness (weighted combination)
            overall_correct = type_correct and field_accuracy > 0.8
            
            correct_predictions.append(overall_correct)
        
        return correct_predictions
    
    def _calculate_field_accuracy(self, pred_fields: Dict[str, Any], 
                                gt_fields: Dict[str, Any]) -> float:
        """Calculate field extraction accuracy for a single document."""
        if not gt_fields:
            return 1.0 if not pred_fields else 0.0
        
        correct_fields = 0
        total_fields = len(gt_fields)
        
        for field_name, gt_value in gt_fields.items():
            pred_value = pred_fields.get(field_name)
            
            if self.metrics_calculator._compare_field_values(pred_value, gt_value, field_name):
                correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_business_metrics(self, predictions: List[Dict[str, Any]], 
                                  ground_truth: List[Dict[str, Any]], 
                                  correct_predictions: List[bool]) -> Dict[str, Any]:
        """Calculate business-specific metrics."""
        total_docs = len(predictions)
        correct_docs = sum(correct_predictions)
        
        # Processing efficiency
        total_processing_time = sum(pred.get('processing_time', 0.0) for pred in predictions)
        avg_processing_time = total_processing_time / total_docs if total_docs > 0 else 0.0
        
        # Confidence distribution
        confidences = [pred.get('confidence', 0.0) for pred in predictions]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Field completeness
        field_completeness = self._calculate_field_completeness(predictions, ground_truth)
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions, ground_truth, correct_predictions)
        
        return {
            'overall_accuracy': correct_docs / total_docs if total_docs > 0 else 0.0,
            'processing_efficiency': {
                'total_time': total_processing_time,
                'avg_time_per_document': avg_processing_time,
                'documents_per_hour': 3600 / avg_processing_time if avg_processing_time > 0 else 0
            },
            'confidence_metrics': {
                'avg_confidence': avg_confidence,
                'confidence_std': np.std(confidences) if confidences else 0.0,
                'low_confidence_count': sum(1 for c in confidences if c < 0.7)
            },
            'field_completeness': field_completeness,
            'error_analysis': error_analysis
        }
    
    def _calculate_field_completeness(self, predictions: List[Dict[str, Any]], 
                                    ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate field completeness metrics."""
        # Get all unique field names
        all_field_names = set()
        for gt in ground_truth:
            for field in gt.get('fields', []):
                all_field_names.add(field['name'])
        
        field_completeness = {}
        
        for field_name in all_field_names:
            total_expected = 0
            total_extracted = 0
            
            for pred, gt in zip(predictions, ground_truth):
                # Count expected occurrences
                gt_fields = {f['name']: f['value'] for f in gt.get('fields', [])}
                if field_name in gt_fields:
                    total_expected += 1
                
                # Count extracted occurrences
                pred_fields = {f['name']: f['value'] for f in pred.get('fields', [])}
                if field_name in pred_fields:
                    total_extracted += 1
            
            completeness = total_extracted / total_expected if total_expected > 0 else 0.0
            field_completeness[field_name] = completeness
        
        return field_completeness
    
    def _analyze_errors(self, predictions: List[Dict[str, Any]], 
                       ground_truth: List[Dict[str, Any]], 
                       correct_predictions: List[bool]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_analysis = {
            'type_errors': {},
            'field_errors': {},
            'confidence_errors': {},
            'common_error_patterns': []
        }
        
        # Analyze type errors
        for pred, gt, is_correct in zip(predictions, ground_truth, correct_predictions):
            if not is_correct:
                pred_type = pred.get('document_type', 'unknown')
                gt_type = gt.get('document_type', 'unknown')
                
                if pred_type != gt_type:
                    error_key = f"{gt_type} -> {pred_type}"
                    error_analysis['type_errors'][error_key] = error_analysis['type_errors'].get(error_key, 0) + 1
        
        # Analyze field errors
        for pred, gt, is_correct in zip(predictions, ground_truth, correct_predictions):
            if not is_correct:
                pred_fields = {f['name']: f['value'] for f in pred.get('fields', [])}
                gt_fields = {f['name']: f['value'] for f in gt.get('fields', [])}
                
                for field_name in set(pred_fields.keys()) | set(gt_fields.keys()):
                    pred_value = pred_fields.get(field_name)
                    gt_value = gt_fields.get(field_name)
                    
                    if not self.metrics_calculator._compare_field_values(pred_value, gt_value, field_name):
                        error_analysis['field_errors'][field_name] = error_analysis['field_errors'].get(field_name, 0) + 1
        
        # Analyze confidence errors
        for pred, is_correct in zip(predictions, correct_predictions):
            confidence = pred.get('confidence', 0.0)
            
            if is_correct and confidence < 0.7:
                error_analysis['confidence_errors']['low_confidence_correct'] = error_analysis['confidence_errors'].get('low_confidence_correct', 0) + 1
            elif not is_correct and confidence > 0.8:
                error_analysis['confidence_errors']['high_confidence_incorrect'] = error_analysis['confidence_errors'].get('high_confidence_incorrect', 0) + 1
        
        return error_analysis
    
    def _create_summary(self, classification_results: Dict[str, Any], 
                       field_extraction_results: Dict[str, FieldMetrics], 
                       confidence_calibration: Dict[str, Any], 
                       processing_performance: Dict[str, Any], 
                       business_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        # Calculate overall field metrics
        field_metrics_list = list(field_extraction_results.values())
        if field_metrics_list:
            avg_field_precision = np.mean([m.precision for m in field_metrics_list])
            avg_field_recall = np.mean([m.recall for m in field_metrics_list])
            avg_field_f1 = np.mean([m.f1_score for m in field_metrics_list])
        else:
            avg_field_precision = avg_field_recall = avg_field_f1 = 0.0
        
        return {
            'overall_accuracy': business_metrics['overall_accuracy'],
            'classification_accuracy': classification_results['accuracy'],
            'field_extraction_metrics': {
                'avg_precision': avg_field_precision,
                'avg_recall': avg_field_recall,
                'avg_f1_score': avg_field_f1
            },
            'confidence_calibration': {
                'ece': confidence_calibration['ece'],
                'brier_score': confidence_calibration['brier_score']
            },
            'processing_performance': {
                'avg_processing_time': processing_performance['avg_processing_time'],
                'documents_per_second': processing_performance['documents_per_second']
            },
            'recommendations': self._generate_recommendations(
                classification_results, field_extraction_results, 
                confidence_calibration, processing_performance, business_metrics
            )
        }
    
    def _generate_recommendations(self, classification_results: Dict[str, Any], 
                                field_extraction_results: Dict[str, FieldMetrics], 
                                confidence_calibration: Dict[str, Any], 
                                processing_performance: Dict[str, Any], 
                                business_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Classification recommendations
        if classification_results['accuracy'] < 0.9:
            recommendations.append("Consider improving document type classification with more training data or feature engineering")
        
        # Field extraction recommendations
        field_metrics_list = list(field_extraction_results.values())
        if field_metrics_list:
            avg_field_f1 = np.mean([m.f1_score for m in field_metrics_list])
            if avg_field_f1 < 0.8:
                recommendations.append("Field extraction performance is below target - consider pattern refinement or ML model improvement")
        
        # Confidence calibration recommendations
        if confidence_calibration['ece'] > 0.1:
            recommendations.append("Confidence calibration needs improvement - consider calibration techniques or threshold adjustment")
        
        # Processing performance recommendations
        if processing_performance['avg_processing_time'] > 1.0:
            recommendations.append("Processing time is high - consider optimization or parallel processing")
        
        # Business metrics recommendations
        if business_metrics['overall_accuracy'] < 0.85:
            recommendations.append("Overall accuracy is below production threshold - implement human-in-loop validation")
        
        if business_metrics['confidence_metrics']['low_confidence_count'] > len(field_metrics_list) * 0.2:
            recommendations.append("High number of low-confidence predictions - consider confidence threshold adjustment")
        
        return recommendations
    
    def create_leaderboard(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a leaderboard of performance metrics."""
        leaderboard_data = []
        
        # Document classification metrics
        if 'classification' in evaluation_results:
            cls_results = evaluation_results['classification']
            leaderboard_data.append({
                'Metric': 'Document Classification Accuracy',
                'Value': cls_results['accuracy'],
                'Category': 'Classification',
                'Target': 0.95,
                'Status': 'Good' if cls_results['accuracy'] >= 0.95 else 'Needs Improvement'
            })
        
        # Field extraction metrics
        if 'field_extraction' in evaluation_results:
            field_results = evaluation_results['field_extraction']
            for field_name, metrics in field_results.items():
                leaderboard_data.append({
                    'Metric': f'Field Extraction F1 - {field_name}',
                    'Value': metrics.f1_score,
                    'Category': 'Field Extraction',
                    'Target': 0.9,
                    'Status': 'Good' if metrics.f1_score >= 0.9 else 'Needs Improvement'
                })
        
        # Processing performance metrics
        if 'processing_performance' in evaluation_results:
            perf_results = evaluation_results['processing_performance']
            leaderboard_data.append({
                'Metric': 'Processing Speed (docs/sec)',
                'Value': perf_results['documents_per_second'],
                'Category': 'Performance',
                'Target': 10.0,
                'Status': 'Good' if perf_results['documents_per_second'] >= 10.0 else 'Needs Improvement'
            })
        
        # Confidence calibration metrics
        if 'confidence_calibration' in evaluation_results:
            cal_results = evaluation_results['confidence_calibration']
            leaderboard_data.append({
                'Metric': 'Expected Calibration Error',
                'Value': cal_results['ece'],
                'Category': 'Calibration',
                'Target': 0.05,
                'Status': 'Good' if cal_results['ece'] <= 0.05 else 'Needs Improvement'
            })
        
        return pd.DataFrame(leaderboard_data)
    
    def save_evaluation_report(self, evaluation_results: Dict[str, Any], 
                             output_dir: str) -> None:
        """Save comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        self.metrics_calculator.save_evaluation_results(evaluation_results, str(results_file))
        
        # Save leaderboard
        leaderboard = self.create_leaderboard(evaluation_results)
        leaderboard_file = output_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_file, index=False)
        
        # Save text report
        report = self.metrics_calculator.create_evaluation_report(evaluation_results)
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def compare_evaluations(self, evaluation1: Dict[str, Any], 
                          evaluation2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two evaluation results."""
        comparison = {
            'evaluation1_timestamp': evaluation1.get('evaluation_timestamp'),
            'evaluation2_timestamp': evaluation2.get('evaluation_timestamp'),
            'improvements': {},
            'regressions': {},
            'summary': {}
        }
        
        # Compare classification accuracy
        if 'classification' in both evaluations:
            acc1 = evaluation1['classification']['accuracy']
            acc2 = evaluation2['classification']['accuracy']
            diff = acc2 - acc1
            
            if diff > 0.01:
                comparison['improvements']['classification_accuracy'] = diff
            elif diff < -0.01:
                comparison['regressions']['classification_accuracy'] = diff
        
        # Compare field extraction metrics
        if 'field_extraction' in both evaluations:
            fields1 = evaluation1['field_extraction']
            fields2 = evaluation2['field_extraction']
            
            for field_name in set(fields1.keys()) | set(fields2.keys()):
                if field_name in fields1 and field_name in fields2:
                    f1_1 = fields1[field_name].f1_score
                    f1_2 = fields2[field_name].f1_score
                    diff = f1_2 - f1_1
                    
                    if diff > 0.01:
                        comparison['improvements'][f'field_{field_name}_f1'] = diff
                    elif diff < -0.01:
                        comparison['regressions'][f'field_{field_name}_f1'] = diff
        
        # Compare processing performance
        if 'processing_performance' in both evaluations:
            perf1 = evaluation1['processing_performance']
            perf2 = evaluation2['processing_performance']
            
            speed1 = perf1['documents_per_second']
            speed2 = perf2['documents_per_second']
            speed_diff = speed2 - speed1
            
            if speed_diff > 1.0:
                comparison['improvements']['processing_speed'] = speed_diff
            elif speed_diff < -1.0:
                comparison['regressions']['processing_speed'] = speed_diff
        
        return comparison
