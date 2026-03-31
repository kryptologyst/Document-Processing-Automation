"""
Pytest configuration and shared fixtures for document processing automation tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.processing.document_processor import DocumentProcessor, DocumentResult, DocumentField
from src.data.data_generator import DocumentDataGenerator
from src.data.schema import DocumentType
from src.eval.evaluator import DocumentProcessingEvaluator
from src.viz.explainability import ExplainabilityEngine


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_document_texts():
    """Sample document texts for testing."""
    return [
        "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00",
        "Receipt #R789\nDate: 2024-01-16\nMerchant: TechStore\nTotal: $89.99",
        "Contract #CON-001\nDate: 2024-01-17\nParties: Company A and Company B\nValue: $50,000.00"
    ]


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return [
        {
            'invoice_number': '12345',
            'date': '2024-01-15',
            'customer': 'ABC Corp',
            'total': 1100.00
        },
        {
            'receipt_number': 'R789',
            'date': '2024-01-16',
            'merchant': 'TechStore',
            'total': 89.99
        },
        {
            'contract_id': 'CON-001',
            'date': '2024-01-17',
            'parties': 'Company A and Company B',
            'value': 50000.00
        }
    ]


@pytest.fixture
def document_processor():
    """Document processor instance for testing."""
    return DocumentProcessor(confidence_threshold=0.7)


@pytest.fixture
def data_generator():
    """Data generator instance for testing."""
    return DocumentDataGenerator(seed=42)


@pytest.fixture
def evaluator():
    """Evaluator instance for testing."""
    return DocumentProcessingEvaluator()


@pytest.fixture
def explainability_engine():
    """Explainability engine instance for testing."""
    return ExplainabilityEngine()


@pytest.fixture
def sample_document_result():
    """Sample document result for testing."""
    fields = [
        DocumentField(
            name='invoice_number',
            value='12345',
            confidence=0.9,
            source_text='Invoice #12345'
        ),
        DocumentField(
            name='date',
            value='2024-01-15',
            confidence=0.8,
            source_text='Date: 2024-01-15'
        ),
        DocumentField(
            name='customer',
            value='ABC Corp',
            confidence=0.85,
            source_text='Customer: ABC Corp'
        ),
        DocumentField(
            name='total',
            value=1100.00,
            confidence=0.9,
            source_text='Total: $1,100.00'
        )
    ]
    
    return DocumentResult(
        document_id='test_doc_001',
        document_type='invoice',
        fields=fields,
        overall_confidence=0.86,
        processing_time=0.1,
        raw_text='Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00'
    )


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return [
        {
            'document_type': 'invoice',
            'fields': [
                {'name': 'invoice_number', 'value': '12345'},
                {'name': 'date', 'value': '2024-01-15'},
                {'name': 'customer', 'value': 'ABC Corp'},
                {'name': 'total', 'value': 1100.00}
            ],
            'confidence': 0.86,
            'processing_time': 0.1
        },
        {
            'document_type': 'receipt',
            'fields': [
                {'name': 'receipt_number', 'value': 'R789'},
                {'name': 'date', 'value': '2024-01-16'},
                {'name': 'merchant', 'value': 'TechStore'},
                {'name': 'total', 'value': 89.99}
            ],
            'confidence': 0.82,
            'processing_time': 0.08
        }
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    return [
        {
            'document_type': 'invoice',
            'fields': [
                {'name': 'invoice_number', 'value': '12345'},
                {'name': 'date', 'value': '2024-01-15'},
                {'name': 'customer', 'value': 'ABC Corp'},
                {'name': 'total', 'value': 1100.00}
            ]
        },
        {
            'document_type': 'receipt',
            'fields': [
                {'name': 'receipt_number', 'value': 'R789'},
                {'name': 'date', 'value': '2024-01-16'},
                {'name': 'merchant', 'value': 'TechStore'},
                {'name': 'total', 'value': 89.99}
            ]
        }
    ]


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return {
        'evaluation_timestamp': '2024-01-01T00:00:00',
        'total_documents': 2,
        'classification': {
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0
        },
        'field_extraction': {
            'invoice_number': type('FieldMetrics', (), {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'accuracy': 1.0,
                'total_extracted': 1,
                'total_expected': 1,
                'total_correct': 1,
                'false_positives': 0,
                'false_negatives': 0
            })(),
            'date': type('FieldMetrics', (), {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'accuracy': 1.0,
                'total_extracted': 2,
                'total_expected': 2,
                'total_correct': 2,
                'false_positives': 0,
                'false_negatives': 0
            })()
        },
        'processing_performance': {
            'avg_processing_time': 0.09,
            'documents_per_second': 11.11
        },
        'summary': {
            'overall_accuracy': 1.0,
            'classification_accuracy': 1.0,
            'field_extraction_metrics': {
                'avg_precision': 1.0,
                'avg_recall': 1.0,
                'avg_f1_score': 1.0
            }
        }
    }


@pytest.fixture
def sample_confidence_scores():
    """Sample confidence scores for testing."""
    return [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@pytest.fixture
def sample_correct_predictions():
    """Sample correct predictions for testing."""
    return [True, True, False, True, False, False, False, False, False]


@pytest.fixture
def sample_processing_times():
    """Sample processing times for testing."""
    return [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28, 0.12]


@pytest.fixture
def sample_document_sizes():
    """Sample document sizes for testing."""
    return [100, 200, 150, 300, 250, 180, 220, 280, 120]


@pytest.fixture
def mock_document_types():
    """Mock document types for testing."""
    return ['invoice', 'receipt', 'contract', 'form', 'rfp']


@pytest.fixture
def mock_field_names():
    """Mock field names for testing."""
    return ['date', 'total', 'customer', 'invoice_number', 'merchant']


@pytest.fixture
def mock_field_values():
    """Mock field values for testing."""
    return ['2024-01-15', 1100.00, 'ABC Corp', '12345', 'TechStore']


@pytest.fixture
def sample_text_quality_scores():
    """Sample text quality scores for testing."""
    return [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@pytest.fixture
def sample_pattern_matching_scores():
    """Sample pattern matching scores for testing."""
    return [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]


@pytest.fixture
def sample_anomaly_scores():
    """Sample anomaly scores for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def sample_context_relevance_scores():
    """Sample context relevance scores for testing."""
    return [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


@pytest.fixture
def sample_explanation_factors():
    """Sample explanation factors for testing."""
    return {
        'pattern_match_quality': 0.8,
        'field_completeness': 0.9,
        'consistency_check': 0.7,
        'anomaly_score': 0.6,
        'context_relevance': 0.85
    }


@pytest.fixture
def sample_recommendations():
    """Sample recommendations for testing."""
    return [
        "Consider improving document type classification with more training data",
        "Field extraction performance is below target - consider pattern refinement",
        "Confidence calibration needs improvement - consider calibration techniques",
        "Processing time is high - consider optimization or parallel processing"
    ]


@pytest.fixture
def sample_error_analysis():
    """Sample error analysis for testing."""
    return {
        'error_types': {
            'Field Missing': 25,
            'Low Confidence': 15,
            'Type Mismatch': 10,
            'Processing Error': 5
        },
        'errors_by_type': {
            'invoice': 20,
            'receipt': 15,
            'contract': 10
        },
        'error_trends': [5, 4, 6, 3, 7, 2, 8, 1, 9, 0],
        'confidence_vs_errors': [
            {'confidence': 0.3, 'error_count': 8},
            {'confidence': 0.5, 'error_count': 5},
            {'confidence': 0.7, 'error_count': 2},
            {'confidence': 0.9, 'error_count': 1}
        ]
    }


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results for testing."""
    return {
        'benchmark_timestamp': '2024-01-01T00:00:00',
        'test_sizes': [10, 50, 100],
        'results': {
            10: {
                'performance_metrics': {
                    'avg_processing_time': 0.1,
                    'avg_throughput': 10.0
                },
                'accuracy_metrics': {
                    'avg_classification_accuracy': 0.95,
                    'avg_field_extraction_f1': 0.90
                }
            },
            50: {
                'performance_metrics': {
                    'avg_processing_time': 0.12,
                    'avg_throughput': 8.33
                },
                'accuracy_metrics': {
                    'avg_classification_accuracy': 0.93,
                    'avg_field_extraction_f1': 0.88
                }
            },
            100: {
                'performance_metrics': {
                    'avg_processing_time': 0.15,
                    'avg_throughput': 6.67
                },
                'accuracy_metrics': {
                    'avg_classification_accuracy': 0.91,
                    'avg_field_extraction_f1': 0.85
                }
            }
        },
        'aggregate_metrics': {
            'scalability': {
                'linear_scaling': False,
                'processing_time_scaling': 1.5,
                'throughput_scaling': 0.67
            },
            'performance_trends': {
                'processing_time_trend': 'increasing',
                'throughput_trend': 'decreasing'
            },
            'accuracy_trends': {
                'accuracy_trend': 'decreasing',
                'f1_score_trend': 'decreasing'
            }
        },
        'recommendations': [
            "Performance does not scale linearly - consider optimization for larger datasets",
            "Processing time increases with dataset size - implement caching or parallel processing",
            "Accuracy decreases with dataset size - consider model retraining or data quality improvement"
        ]
    }


@pytest.fixture
def sample_stress_test_results():
    """Sample stress test results for testing."""
    return {
        'total_documents': 1000,
        'total_processing_time': 120.5,
        'avg_processing_time': 0.1205,
        'documents_per_second': 8.3,
        'batch_times': [2.1, 2.3, 2.0, 2.4, 2.2],
        'batch_performance': {
            'avg_batch_time': 2.2,
            'std_batch_time': 0.15,
            'min_batch_time': 2.0,
            'max_batch_time': 2.4
        },
        'error_rate': 0.05,
        'avg_confidence': 0.75
    }


@pytest.fixture
def sample_concurrent_test_results():
    """Sample concurrent test results for testing."""
    return {
        'num_threads': 4,
        'documents_per_thread': 50,
        'total_documents': 200,
        'total_processing_time': 25.0,
        'avg_processing_time': 0.125,
        'documents_per_second': 8.0,
        'thread_times': [6.0, 6.5, 6.2, 6.3],
        'thread_performance': {
            'avg_thread_time': 6.25,
            'std_thread_time': 0.2,
            'min_thread_time': 6.0,
            'max_thread_time': 6.5
        },
        'concurrency_efficiency': 0.8,
        'error_rate': 0.02
    }


@pytest.fixture
def sample_feature_importance():
    """Sample feature importance for testing."""
    return {
        'invoice_keywords': 0.25,
        'receipt_keywords': 0.20,
        'contract_keywords': 0.18,
        'date_patterns': 0.15,
        'amount_patterns': 0.12,
        'customer_patterns': 0.10
    }


@pytest.fixture
def sample_timeline_data():
    """Sample timeline data for testing."""
    return [
        {'timestamp': '2024-01-01T10:00:00', 'processing_time': 0.1, 'confidence': 0.9},
        {'timestamp': '2024-01-01T10:01:00', 'processing_time': 0.12, 'confidence': 0.85},
        {'timestamp': '2024-01-01T10:02:00', 'processing_time': 0.08, 'confidence': 0.95},
        {'timestamp': '2024-01-01T10:03:00', 'processing_time': 0.15, 'confidence': 0.75},
        {'timestamp': '2024-01-01T10:04:00', 'processing_time': 0.11, 'confidence': 0.88}
    ]


@pytest.fixture
def sample_reliability_data():
    """Sample reliability data for testing."""
    return [
        {'bin_center': 0.1, 'accuracy': 0.15, 'confidence': 0.1, 'count': 50},
        {'bin_center': 0.3, 'accuracy': 0.35, 'confidence': 0.3, 'count': 100},
        {'bin_center': 0.5, 'accuracy': 0.55, 'confidence': 0.5, 'count': 150},
        {'bin_center': 0.7, 'accuracy': 0.75, 'confidence': 0.7, 'count': 200},
        {'bin_center': 0.9, 'accuracy': 0.85, 'confidence': 0.9, 'count': 100}
    ]


@pytest.fixture
def sample_calibration_data():
    """Sample calibration data for testing."""
    return {
        'ece': 0.05,
        'brier_score': 0.15,
        'reliability_data': [
            {'bin_center': 0.1, 'accuracy': 0.15, 'confidence': 0.1, 'count': 50},
            {'bin_center': 0.3, 'accuracy': 0.35, 'confidence': 0.3, 'count': 100},
            {'bin_center': 0.5, 'accuracy': 0.55, 'confidence': 0.5, 'count': 150},
            {'bin_center': 0.7, 'accuracy': 0.75, 'confidence': 0.7, 'count': 200},
            {'bin_center': 0.9, 'accuracy': 0.85, 'confidence': 0.9, 'count': 100}
        ]
    }


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing."""
    return {
        'processing_times': [0.1, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.07, 0.14, 0.10],
        'throughput': [10.0, 8.33, 12.5, 6.67, 9.09, 11.11, 7.69, 14.29, 7.14, 10.0],
        'document_sizes': [100, 200, 150, 300, 250, 180, 220, 280, 120, 160],
        'batch_times': [2.1, 2.3, 2.0, 2.4, 2.2]
    }


@pytest.fixture
def sample_explainability_report():
    """Sample explainability report for testing."""
    return {
        'total_documents': 100,
        'decision_distribution': {
            'auto_approved': 70,
            'human_review_required': 25,
            'auto_rejected': 5
        },
        'confidence_statistics': {
            'mean': 0.75,
            'std': 0.15,
            'min': 0.3,
            'max': 0.95
        },
        'factor_statistics': {
            'pattern_match_quality': {'mean': 0.8, 'std': 0.1},
            'field_completeness': {'mean': 0.85, 'std': 0.12},
            'consistency_check': {'mean': 0.75, 'std': 0.15}
        },
        'common_recommendations': [
            ('Consider manual review of extracted fields', 25),
            ('Verify document type classification', 20),
            ('Review low-confidence field extractions', 15)
        ],
        'human_review_rate': 0.25
    }
