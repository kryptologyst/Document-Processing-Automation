"""
Benchmark Suite Module

This module provides benchmarking capabilities for document processing
automation systems, including performance testing and comparison.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime
import concurrent.futures
import threading

from .evaluator import DocumentProcessingEvaluator
from ..data.data_generator import DocumentDataGenerator
from ..data.schema import DocumentType

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Comprehensive benchmark suite for document processing systems."""
    
    def __init__(self, evaluator: Optional[DocumentProcessingEvaluator] = None):
        """
        Initialize benchmark suite.
        
        Args:
            evaluator: Document processing evaluator instance
        """
        self.evaluator = evaluator or DocumentProcessingEvaluator()
        self.data_generator = DocumentDataGenerator(seed=42)
        self.benchmark_results = []
    
    def run_comprehensive_benchmark(self, processor, test_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple document types and sizes.
        
        Args:
            processor: Document processor instance
            test_sizes: List of test dataset sizes
            
        Returns:
            Comprehensive benchmark results
        """
        if test_sizes is None:
            test_sizes = [10, 50, 100, 500]
        
        logger.info("Starting comprehensive benchmark")
        
        benchmark_results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'test_sizes': test_sizes,
            'document_types': [dt.value for dt in DocumentType],
            'results': {}
        }
        
        for size in test_sizes:
            logger.info(f"Running benchmark for size {size}")
            size_results = self._benchmark_size(processor, size)
            benchmark_results['results'][size] = size_results
        
        # Calculate aggregate metrics
        benchmark_results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            benchmark_results['results']
        )
        
        # Generate recommendations
        benchmark_results['recommendations'] = self._generate_benchmark_recommendations(
            benchmark_results['aggregate_metrics']
        )
        
        self.benchmark_results.append(benchmark_results)
        
        return benchmark_results
    
    def _benchmark_size(self, processor, size: int) -> Dict[str, Any]:
        """Benchmark for a specific dataset size."""
        results = {
            'dataset_size': size,
            'document_type_results': {},
            'performance_metrics': {},
            'accuracy_metrics': {}
        }
        
        # Test each document type
        for doc_type in DocumentType:
            logger.info(f"Testing {doc_type.value} with {size} documents")
            type_results = self._benchmark_document_type(processor, doc_type, size)
            results['document_type_results'][doc_type.value] = type_results
        
        # Calculate overall performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(
            results['document_type_results']
        )
        
        # Calculate overall accuracy metrics
        results['accuracy_metrics'] = self._calculate_accuracy_metrics(
            results['document_type_results']
        )
        
        return results
    
    def _benchmark_document_type(self, processor, doc_type: DocumentType, size: int) -> Dict[str, Any]:
        """Benchmark for a specific document type."""
        # Generate test data
        test_texts = self.data_generator.generate_text_batch(doc_type, size)
        test_data = self.data_generator.generate_batch(doc_type, size)
        
        # Measure processing time
        start_time = time.time()
        predictions = []
        
        for text in test_texts:
            try:
                result = processor.process_document(text)
                predictions.append({
                    'document_type': result.document_type,
                    'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                    'confidence': result.overall_confidence,
                    'processing_time': result.processing_time
                })
            except Exception as e:
                logger.error(f"Processing error: {e}")
                predictions.append({
                    'document_type': 'error',
                    'fields': [],
                    'confidence': 0.0,
                    'processing_time': 0.0
                })
        
        total_time = time.time() - start_time
        
        # Prepare ground truth
        ground_truth = []
        for data in test_data:
            ground_truth.append({
                'document_type': doc_type.value,
                'fields': [{'name': k, 'value': v} for k, v in data.items() if v is not None]
            })
        
        # Evaluate results
        evaluation_results = self.evaluator.evaluate_system(predictions, ground_truth)
        
        return {
            'document_type': doc_type.value,
            'dataset_size': size,
            'total_processing_time': total_time,
            'avg_processing_time': total_time / size if size > 0 else 0,
            'documents_per_second': size / total_time if total_time > 0 else 0,
            'evaluation_results': evaluation_results
        }
    
    def _calculate_performance_metrics(self, document_type_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        all_times = []
        all_throughputs = []
        
        for doc_type, results in document_type_results.items():
            all_times.append(results['avg_processing_time'])
            all_throughputs.append(results['documents_per_second'])
        
        return {
            'avg_processing_time': np.mean(all_times),
            'median_processing_time': np.median(all_times),
            'std_processing_time': np.std(all_times),
            'avg_throughput': np.mean(all_throughputs),
            'median_throughput': np.median(all_throughputs),
            'std_throughput': np.std(all_throughputs)
        }
    
    def _calculate_accuracy_metrics(self, document_type_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall accuracy metrics."""
        all_accuracies = []
        all_f1_scores = []
        
        for doc_type, results in document_type_results.items():
            eval_results = results['evaluation_results']
            if 'classification' in eval_results:
                all_accuracies.append(eval_results['classification']['accuracy'])
            
            if 'field_extraction' in eval_results:
                field_metrics = eval_results['field_extraction']
                if field_metrics:
                    f1_scores = [m.f1_score for m in field_metrics.values()]
                    all_f1_scores.extend(f1_scores)
        
        return {
            'avg_classification_accuracy': np.mean(all_accuracies) if all_accuracies else 0,
            'avg_field_extraction_f1': np.mean(all_f1_scores) if all_f1_scores else 0,
            'min_classification_accuracy': np.min(all_accuracies) if all_accuracies else 0,
            'max_classification_accuracy': np.max(all_accuracies) if all_accuracies else 0
        }
    
    def _calculate_aggregate_metrics(self, results: Dict[int, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all test sizes."""
        aggregate = {
            'scalability': {},
            'performance_trends': {},
            'accuracy_trends': {}
        }
        
        sizes = sorted(results.keys())
        
        # Scalability analysis
        processing_times = [results[size]['performance_metrics']['avg_processing_time'] for size in sizes]
        throughputs = [results[size]['performance_metrics']['avg_throughput'] for size in sizes]
        
        aggregate['scalability'] = {
            'processing_time_scaling': self._calculate_scaling_factor(processing_times),
            'throughput_scaling': self._calculate_scaling_factor(throughputs),
            'linear_scaling': self._check_linear_scaling(processing_times, sizes)
        }
        
        # Performance trends
        aggregate['performance_trends'] = {
            'processing_time_trend': self._calculate_trend(processing_times),
            'throughput_trend': self._calculate_trend(throughputs)
        }
        
        # Accuracy trends
        accuracies = [results[size]['accuracy_metrics']['avg_classification_accuracy'] for size in sizes]
        f1_scores = [results[size]['accuracy_metrics']['avg_field_extraction_f1'] for size in sizes]
        
        aggregate['accuracy_trends'] = {
            'accuracy_trend': self._calculate_trend(accuracies),
            'f1_score_trend': self._calculate_trend(f1_scores)
        }
        
        return aggregate
    
    def _calculate_scaling_factor(self, values: List[float]) -> float:
        """Calculate scaling factor for performance metrics."""
        if len(values) < 2:
            return 1.0
        
        # Calculate ratio of last to first value
        return values[-1] / values[0] if values[0] != 0 else 1.0
    
    def _check_linear_scaling(self, values: List[float], sizes: List[int]) -> bool:
        """Check if performance scales linearly with dataset size."""
        if len(values) < 2:
            return True
        
        # Calculate correlation between values and sizes
        correlation = np.corrcoef(values, sizes)[0, 1]
        return abs(correlation) > 0.8
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Calculate slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_benchmark_recommendations(self, aggregate_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Scalability recommendations
        scalability = aggregate_metrics.get('scalability', {})
        if not scalability.get('linear_scaling', True):
            recommendations.append("Performance does not scale linearly - consider optimization for larger datasets")
        
        # Performance recommendations
        performance_trends = aggregate_metrics.get('performance_trends', {})
        if performance_trends.get('processing_time_trend') == 'increasing':
            recommendations.append("Processing time increases with dataset size - implement caching or parallel processing")
        
        # Accuracy recommendations
        accuracy_trends = aggregate_metrics.get('accuracy_trends', {})
        if accuracy_trends.get('accuracy_trend') == 'decreasing':
            recommendations.append("Accuracy decreases with dataset size - consider model retraining or data quality improvement")
        
        return recommendations
    
    def run_stress_test(self, processor, max_documents: int = 1000, 
                       batch_size: int = 100) -> Dict[str, Any]:
        """
        Run stress test to evaluate system under high load.
        
        Args:
            processor: Document processor instance
            max_documents: Maximum number of documents to process
            batch_size: Batch size for processing
            
        Returns:
            Stress test results
        """
        logger.info(f"Running stress test with {max_documents} documents")
        
        # Generate mixed document types
        doc_counts = {
            DocumentType.INVOICE: max_documents // 3,
            DocumentType.RECEIPT: max_documents // 3,
            DocumentType.CONTRACT: max_documents // 3
        }
        
        test_data = self.data_generator.generate_mixed_batch(doc_counts)
        
        # Process in batches
        start_time = time.time()
        results = []
        batch_times = []
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_start = time.time()
            
            batch_results = []
            for doc_type, text, data in batch:
                try:
                    result = processor.process_document(text)
                    batch_results.append({
                        'document_type': result.document_type,
                        'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                        'confidence': result.overall_confidence,
                        'processing_time': result.processing_time
                    })
                except Exception as e:
                    logger.error(f"Stress test error: {e}")
                    batch_results.append({
                        'document_type': 'error',
                        'fields': [],
                        'confidence': 0.0,
                        'processing_time': 0.0
                    })
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}, time: {batch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Analyze results
        stress_results = {
            'total_documents': len(test_data),
            'total_processing_time': total_time,
            'avg_processing_time': total_time / len(test_data),
            'documents_per_second': len(test_data) / total_time,
            'batch_times': batch_times,
            'batch_performance': {
                'avg_batch_time': np.mean(batch_times),
                'std_batch_time': np.std(batch_times),
                'min_batch_time': np.min(batch_times),
                'max_batch_time': np.max(batch_times)
            },
            'error_rate': sum(1 for r in results if r['document_type'] == 'error') / len(results),
            'avg_confidence': np.mean([r['confidence'] for r in results])
        }
        
        return stress_results
    
    def run_concurrent_test(self, processor, num_threads: int = 4, 
                          documents_per_thread: int = 50) -> Dict[str, Any]:
        """
        Run concurrent processing test.
        
        Args:
            processor: Document processor instance
            num_threads: Number of concurrent threads
            documents_per_thread: Number of documents per thread
            
        Returns:
            Concurrent test results
        """
        logger.info(f"Running concurrent test with {num_threads} threads")
        
        # Generate test data
        test_texts = self.data_generator.generate_text_batch(
            DocumentType.INVOICE, num_threads * documents_per_thread
        )
        
        # Thread-safe results collection
        results = []
        results_lock = threading.Lock()
        
        def process_batch(thread_id: int, texts: List[str]):
            """Process a batch of documents in a thread."""
            thread_results = []
            start_time = time.time()
            
            for text in texts:
                try:
                    result = processor.process_document(text)
                    thread_results.append({
                        'thread_id': thread_id,
                        'document_type': result.document_type,
                        'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                        'confidence': result.overall_confidence,
                        'processing_time': result.processing_time
                    })
                except Exception as e:
                    logger.error(f"Thread {thread_id} error: {e}")
                    thread_results.append({
                        'thread_id': thread_id,
                        'document_type': 'error',
                        'fields': [],
                        'confidence': 0.0,
                        'processing_time': 0.0
                    })
            
            thread_time = time.time() - start_time
            
            with results_lock:
                results.extend(thread_results)
            
            return thread_time, len(thread_results)
        
        # Run concurrent processing
        start_time = time.time()
        thread_times = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads):
                start_idx = i * documents_per_thread
                end_idx = start_idx + documents_per_thread
                thread_texts = test_texts[start_idx:end_idx]
                
                future = executor.submit(process_batch, i, thread_texts)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                thread_time, num_processed = future.result()
                thread_times.append(thread_time)
        
        total_time = time.time() - start_time
        
        # Analyze results
        concurrent_results = {
            'num_threads': num_threads,
            'documents_per_thread': documents_per_thread,
            'total_documents': len(test_texts),
            'total_processing_time': total_time,
            'avg_processing_time': total_time / len(test_texts),
            'documents_per_second': len(test_texts) / total_time,
            'thread_times': thread_times,
            'thread_performance': {
                'avg_thread_time': np.mean(thread_times),
                'std_thread_time': np.std(thread_times),
                'min_thread_time': np.min(thread_times),
                'max_thread_time': np.max(thread_times)
            },
            'concurrency_efficiency': self._calculate_concurrency_efficiency(thread_times, total_time),
            'error_rate': sum(1 for r in results if r['document_type'] == 'error') / len(results)
        }
        
        return concurrent_results
    
    def _calculate_concurrency_efficiency(self, thread_times: List[float], 
                                        total_time: float) -> float:
        """Calculate concurrency efficiency."""
        if not thread_times or total_time == 0:
            return 0.0
        
        # Ideal time if perfectly parallel
        ideal_time = max(thread_times)
        
        # Actual time
        actual_time = total_time
        
        # Efficiency = ideal_time / actual_time
        efficiency = ideal_time / actual_time if actual_time > 0 else 0.0
        
        return min(efficiency, 1.0)  # Cap at 100%
    
    def save_benchmark_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save benchmark results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary = self._create_benchmark_summary(results)
        summary_file = output_dir / "benchmark_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Benchmark results saved to {output_dir}")
    
    def _create_benchmark_summary(self, results: Dict[str, Any]) -> str:
        """Create a summary of benchmark results."""
        summary = []
        summary.append("=" * 80)
        summary.append("DOCUMENT PROCESSING AUTOMATION - BENCHMARK SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Overall performance
        if 'aggregate_metrics' in results:
            agg_metrics = results['aggregate_metrics']
            summary.append("OVERALL PERFORMANCE:")
            summary.append("-" * 40)
            
            if 'scalability' in agg_metrics:
                scaling = agg_metrics['scalability']
                summary.append(f"Linear Scaling: {scaling.get('linear_scaling', 'Unknown')}")
                summary.append(f"Processing Time Scaling: {scaling.get('processing_time_scaling', 1.0):.2f}")
                summary.append(f"Throughput Scaling: {scaling.get('throughput_scaling', 1.0):.2f}")
                summary.append("")
        
        # Performance by dataset size
        if 'results' in results:
            summary.append("PERFORMANCE BY DATASET SIZE:")
            summary.append("-" * 40)
            
            for size, size_results in results['results'].items():
                perf_metrics = size_results.get('performance_metrics', {})
                acc_metrics = size_results.get('accuracy_metrics', {})
                
                summary.append(f"Dataset Size: {size}")
                summary.append(f"  Avg Processing Time: {perf_metrics.get('avg_processing_time', 0):.4f}s")
                summary.append(f"  Avg Throughput: {perf_metrics.get('avg_throughput', 0):.2f} docs/sec")
                summary.append(f"  Classification Accuracy: {acc_metrics.get('avg_classification_accuracy', 0):.4f}")
                summary.append(f"  Field Extraction F1: {acc_metrics.get('avg_field_extraction_f1', 0):.4f}")
                summary.append("")
        
        # Recommendations
        if 'recommendations' in results:
            summary.append("RECOMMENDATIONS:")
            summary.append("-" * 40)
            for i, rec in enumerate(results['recommendations'], 1):
                summary.append(f"{i}. {rec}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
