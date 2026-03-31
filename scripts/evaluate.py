#!/usr/bin/env python3
"""
Evaluation Script

This script provides a command-line interface to run evaluations
on the document processing automation system.
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.processing.document_processor import DocumentProcessor
from src.data.data_generator import DocumentDataGenerator
from src.data.schema import DocumentType
from src.eval.evaluator import DocumentProcessingEvaluator
from src.eval.benchmark import BenchmarkSuite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_quick_evaluation(output_dir: str = "results/quick_eval"):
    """Run a quick evaluation."""
    logger.info("Running quick evaluation...")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    evaluator = DocumentProcessingEvaluator()
    
    # Generate test data
    test_documents = []
    ground_truth = []
    
    for doc_type in DocumentType:
        # Generate documents
        texts = data_generator.generate_text_batch(doc_type, 10)
        data = data_generator.generate_batch(doc_type, 10)
        
        # Process documents
        results = processor.process_batch(texts)
        
        # Prepare predictions and ground truth
        for result, gt_data in zip(results, data):
            predictions = {
                'document_type': result.document_type,
                'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                'confidence': result.overall_confidence,
                'processing_time': result.processing_time
            }
            test_documents.append(predictions)
            
            ground_truth_item = {
                'document_type': doc_type.value,
                'fields': [{'name': k, 'value': v} for k, v in gt_data.items() if v is not None]
            }
            ground_truth.append(ground_truth_item)
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_system(test_documents, ground_truth)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Save summary report
    summary_file = output_path / "summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write(generate_summary_report(evaluation_results))
    
    logger.info(f"Evaluation results saved to {output_path}")
    return evaluation_results


def run_comprehensive_evaluation(output_dir: str = "results/comprehensive_eval"):
    """Run a comprehensive evaluation."""
    logger.info("Running comprehensive evaluation...")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    evaluator = DocumentProcessingEvaluator()
    
    # Generate larger test dataset
    test_documents = []
    ground_truth = []
    
    for doc_type in DocumentType:
        # Generate more documents for comprehensive evaluation
        texts = data_generator.generate_text_batch(doc_type, 50)
        data = data_generator.generate_batch(doc_type, 50)
        
        # Process documents
        results = processor.process_batch(texts)
        
        # Prepare predictions and ground truth
        for result, gt_data in zip(results, data):
            predictions = {
                'document_type': result.document_type,
                'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                'confidence': result.overall_confidence,
                'processing_time': result.processing_time
            }
            test_documents.append(predictions)
            
            ground_truth_item = {
                'document_type': doc_type.value,
                'fields': [{'name': k, 'value': v} for k, v in gt_data.items() if v is not None]
            }
            ground_truth.append(ground_truth_item)
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_system(test_documents, ground_truth)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Save summary report
    summary_file = output_path / "summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write(generate_summary_report(evaluation_results))
    
    # Save leaderboard
    leaderboard = evaluator.create_leaderboard(evaluation_results)
    leaderboard_file = output_path / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_file, index=False)
    
    logger.info(f"Comprehensive evaluation results saved to {output_path}")
    return evaluation_results


def run_benchmark_evaluation(output_dir: str = "results/benchmark_eval"):
    """Run a benchmark evaluation."""
    logger.info("Running benchmark evaluation...")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    benchmark_suite = BenchmarkSuite()
    
    # Run comprehensive benchmark
    benchmark_results = benchmark_suite.run_comprehensive_benchmark(processor)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save benchmark results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    # Save summary report
    summary_file = output_path / "benchmark_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(generate_benchmark_summary(benchmark_results))
    
    logger.info(f"Benchmark results saved to {output_path}")
    return benchmark_results


def generate_summary_report(evaluation_results: dict) -> str:
    """Generate a summary report."""
    report = []
    report.append("=" * 80)
    report.append("DOCUMENT PROCESSING AUTOMATION - EVALUATION SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Basic information
    report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Documents: {evaluation_results.get('total_documents', 0)}")
    report.append("")
    
    # Classification results
    if 'classification' in evaluation_results:
        cls_results = evaluation_results['classification']
        report.append("DOCUMENT CLASSIFICATION:")
        report.append("-" * 40)
        report.append(f"Accuracy: {cls_results.get('accuracy', 0):.3f}")
        report.append(f"Precision: {cls_results.get('precision', 0):.3f}")
        report.append(f"Recall: {cls_results.get('recall', 0):.3f}")
        report.append(f"F1-Score: {cls_results.get('f1_score', 0):.3f}")
        report.append("")
    
    # Field extraction results
    if 'field_extraction' in evaluation_results:
        field_results = evaluation_results['field_extraction']
        report.append("FIELD EXTRACTION:")
        report.append("-" * 40)
        
        for field_name, metrics in field_results.items():
            report.append(f"{field_name.upper()}:")
            report.append(f"  Precision: {metrics.precision:.3f}")
            report.append(f"  Recall: {metrics.recall:.3f}")
            report.append(f"  F1-Score: {metrics.f1_score:.3f}")
            report.append(f"  Accuracy: {metrics.accuracy:.3f}")
            report.append("")
    
    # Processing performance
    if 'processing_performance' in evaluation_results:
        perf_results = evaluation_results['processing_performance']
        report.append("PROCESSING PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Average Processing Time: {perf_results.get('avg_processing_time', 0):.3f}s")
        report.append(f"Documents per Second: {perf_results.get('documents_per_second', 0):.2f}")
        report.append("")
    
    # Summary
    if 'summary' in evaluation_results:
        summary = evaluation_results['summary']
        report.append("OVERALL SUMMARY:")
        report.append("-" * 40)
        report.append(f"Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
        report.append(f"Classification Accuracy: {summary.get('classification_accuracy', 0):.3f}")
        
        field_metrics = summary.get('field_extraction_metrics', {})
        if field_metrics:
            report.append(f"Average Field Precision: {field_metrics.get('avg_precision', 0):.3f}")
            report.append(f"Average Field Recall: {field_metrics.get('avg_recall', 0):.3f}")
            report.append(f"Average Field F1-Score: {field_metrics.get('avg_f1_score', 0):.3f}")
        
        report.append("")
        
        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_benchmark_summary(benchmark_results: dict) -> str:
    """Generate a benchmark summary report."""
    report = []
    report.append("=" * 80)
    report.append("DOCUMENT PROCESSING AUTOMATION - BENCHMARK SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Basic information
    report.append(f"Benchmark Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Test Sizes: {benchmark_results.get('test_sizes', [])}")
    report.append("")
    
    # Aggregate metrics
    if 'aggregate_metrics' in benchmark_results:
        agg_metrics = benchmark_results['aggregate_metrics']
        report.append("AGGREGATE METRICS:")
        report.append("-" * 40)
        
        if 'scalability' in agg_metrics:
            scaling = agg_metrics['scalability']
            report.append(f"Linear Scaling: {scaling.get('linear_scaling', 'Unknown')}")
            report.append(f"Processing Time Scaling: {scaling.get('processing_time_scaling', 1.0):.2f}")
            report.append(f"Throughput Scaling: {scaling.get('throughput_scaling', 1.0):.2f}")
            report.append("")
        
        if 'performance_trends' in agg_metrics:
            trends = agg_metrics['performance_trends']
            report.append("PERFORMANCE TRENDS:")
            report.append(f"Processing Time Trend: {trends.get('processing_time_trend', 'Unknown')}")
            report.append(f"Throughput Trend: {trends.get('throughput_trend', 'Unknown')}")
            report.append("")
        
        if 'accuracy_trends' in agg_metrics:
            acc_trends = agg_metrics['accuracy_trends']
            report.append("ACCURACY TRENDS:")
            report.append(f"Accuracy Trend: {acc_trends.get('accuracy_trend', 'Unknown')}")
            report.append(f"F1-Score Trend: {acc_trends.get('f1_score_trend', 'Unknown')}")
            report.append("")
    
    # Performance by dataset size
    if 'results' in benchmark_results:
        report.append("PERFORMANCE BY DATASET SIZE:")
        report.append("-" * 40)
        
        for size, size_results in benchmark_results['results'].items():
            perf_metrics = size_results.get('performance_metrics', {})
            acc_metrics = size_results.get('accuracy_metrics', {})
            
            report.append(f"Dataset Size: {size}")
            report.append(f"  Avg Processing Time: {perf_metrics.get('avg_processing_time', 0):.4f}s")
            report.append(f"  Avg Throughput: {perf_metrics.get('avg_throughput', 0):.2f} docs/sec")
            report.append(f"  Classification Accuracy: {acc_metrics.get('avg_classification_accuracy', 0):.4f}")
            report.append(f"  Field Extraction F1: {acc_metrics.get('avg_field_extraction_f1', 0):.4f}")
            report.append("")
    
    # Recommendations
    if 'recommendations' in benchmark_results:
        recommendations = benchmark_results['recommendations']
        if recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Document Processing Automation Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --quick                        # Run quick evaluation
  python evaluate.py --comprehensive                # Run comprehensive evaluation
  python evaluate.py --benchmark                    # Run benchmark evaluation
  python evaluate.py --all                          # Run all evaluations
  python evaluate.py --quick --output results/      # Custom output directory
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation"
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive evaluation"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark evaluation"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluations"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if any evaluation is selected
    if not any([args.quick, args.comprehensive, args.benchmark, args.all]):
        parser.print_help()
        return 1
    
    # Run selected evaluations
    success = True
    
    if args.all or args.quick:
        logger.info("Running quick evaluation...")
        try:
            run_quick_evaluation(f"{args.output}/quick_eval")
        except Exception as e:
            logger.error(f"Quick evaluation failed: {e}")
            success = False
    
    if args.all or args.comprehensive:
        logger.info("Running comprehensive evaluation...")
        try:
            run_comprehensive_evaluation(f"{args.output}/comprehensive_eval")
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            success = False
    
    if args.all or args.benchmark:
        logger.info("Running benchmark evaluation...")
        try:
            run_benchmark_evaluation(f"{args.output}/benchmark_eval")
        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            success = False
    
    if success:
        logger.info("All evaluations completed successfully!")
        return 0
    else:
        logger.error("Some evaluations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
