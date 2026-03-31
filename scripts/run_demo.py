#!/usr/bin/env python3
"""
Demo Runner Script

This script provides a command-line interface to run the document processing
automation demo with various options and configurations.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.processing.document_processor import DocumentProcessor
from src.data.data_generator import DocumentDataGenerator
from src.data.schema import DocumentType
from src.eval.evaluator import DocumentProcessingEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_streamlit_demo(port: int = 8501, host: str = "localhost"):
    """Run the Streamlit demo application."""
    logger.info(f"Starting Streamlit demo on {host}:{port}")
    
    demo_path = Path(__file__).parent.parent / "demo" / "app.py"
    
    if not demo_path.exists():
        logger.error(f"Demo application not found at {demo_path}")
        return False
    
    try:
        # Run streamlit
        cmd = [
            "streamlit", "run", str(demo_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ]
        
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run Streamlit demo: {e}")
        return False
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install it with: pip install streamlit")
        return False


def run_quick_demo():
    """Run a quick command-line demo."""
    logger.info("Running quick command-line demo")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    
    # Generate sample documents
    logger.info("Generating sample documents...")
    sample_documents = []
    
    for doc_type in [DocumentType.INVOICE, DocumentType.RECEIPT, DocumentType.CONTRACT]:
        texts = data_generator.generate_text_batch(doc_type, 3)
        sample_documents.extend(texts)
    
    # Process documents
    logger.info("Processing documents...")
    results = processor.process_batch(sample_documents)
    
    # Display results
    logger.info("Processing Results:")
    logger.info("=" * 50)
    
    for result in results:
        logger.info(f"Document ID: {result.document_id}")
        logger.info(f"Type: {result.document_type}")
        logger.info(f"Confidence: {result.overall_confidence:.3f}")
        logger.info(f"Processing Time: {result.processing_time:.3f}s")
        logger.info("Extracted Fields:")
        
        for field in result.fields:
            logger.info(f"  {field.name}: {field.value} (confidence: {field.confidence:.3f})")
        
        logger.info("-" * 30)
    
    # Summary statistics
    total_docs = len(results)
    avg_confidence = sum(r.overall_confidence for r in results) / total_docs
    avg_time = sum(r.processing_time for r in results) / total_docs
    
    logger.info("Summary Statistics:")
    logger.info(f"Total Documents: {total_docs}")
    logger.info(f"Average Confidence: {avg_confidence:.3f}")
    logger.info(f"Average Processing Time: {avg_time:.3f}s")
    
    return True


def run_evaluation_demo():
    """Run an evaluation demo."""
    logger.info("Running evaluation demo")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    evaluator = DocumentProcessingEvaluator()
    
    # Generate test data
    logger.info("Generating test data...")
    test_documents = []
    ground_truth = []
    
    for doc_type in DocumentType:
        # Generate documents
        texts = data_generator.generate_text_batch(doc_type, 5)
        data = data_generator.generate_batch(doc_type, 5)
        
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
    logger.info("Running evaluation...")
    evaluation_results = evaluator.evaluate_system(test_documents, ground_truth)
    
    # Display results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    
    # Classification results
    if 'classification' in evaluation_results:
        cls_results = evaluation_results['classification']
        logger.info("Document Classification:")
        logger.info(f"  Accuracy: {cls_results.get('accuracy', 0):.3f}")
        logger.info(f"  Precision: {cls_results.get('precision', 0):.3f}")
        logger.info(f"  Recall: {cls_results.get('recall', 0):.3f}")
        logger.info(f"  F1-Score: {cls_results.get('f1_score', 0):.3f}")
    
    # Field extraction results
    if 'field_extraction' in evaluation_results:
        field_results = evaluation_results['field_extraction']
        logger.info("Field Extraction:")
        for field_name, metrics in field_results.items():
            logger.info(f"  {field_name}:")
            logger.info(f"    Precision: {metrics.precision:.3f}")
            logger.info(f"    Recall: {metrics.recall:.3f}")
            logger.info(f"    F1-Score: {metrics.f1_score:.3f}")
    
    # Processing performance
    if 'processing_performance' in evaluation_results:
        perf_results = evaluation_results['processing_performance']
        logger.info("Processing Performance:")
        logger.info(f"  Average Processing Time: {perf_results.get('avg_processing_time', 0):.3f}s")
        logger.info(f"  Documents per Second: {perf_results.get('documents_per_second', 0):.2f}")
    
    return True


def run_benchmark_demo():
    """Run a benchmark demo."""
    logger.info("Running benchmark demo")
    
    # Initialize components
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    
    # Test different batch sizes
    batch_sizes = [5, 10, 20]
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Generate documents
        texts = data_generator.generate_text_batch(DocumentType.INVOICE, batch_size)
        
        # Process documents
        import time
        start_time = time.time()
        results = processor.process_batch(texts)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / batch_size
        throughput = batch_size / total_time
        
        logger.info(f"  Total Time: {total_time:.3f}s")
        logger.info(f"  Average Time per Document: {avg_time:.3f}s")
        logger.info(f"  Throughput: {throughput:.2f} documents/second")
        logger.info(f"  Average Confidence: {sum(r.overall_confidence for r in results) / len(results):.3f}")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Document Processing Automation Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --streamlit                    # Run Streamlit demo
  python run_demo.py --streamlit --port 8080       # Run on custom port
  python run_demo.py --quick                       # Run quick CLI demo
  python run_demo.py --evaluation                  # Run evaluation demo
  python run_demo.py --benchmark                   # Run benchmark demo
  python run_demo.py --all                         # Run all demos
        """
    )
    
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit demo application"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick command-line demo"
    )
    
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Run evaluation demo"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark demo"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit demo (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for Streamlit demo (default: localhost)"
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
    
    # Check if any demo is selected
    if not any([args.streamlit, args.quick, args.evaluation, args.benchmark, args.all]):
        parser.print_help()
        return 1
    
    # Run selected demos
    success = True
    
    if args.all or args.streamlit:
        logger.info("Running Streamlit demo...")
        if not run_streamlit_demo(args.port, args.host):
            success = False
    
    if args.all or args.quick:
        logger.info("Running quick demo...")
        if not run_quick_demo():
            success = False
    
    if args.all or args.evaluation:
        logger.info("Running evaluation demo...")
        if not run_evaluation_demo():
            success = False
    
    if args.all or args.benchmark:
        logger.info("Running benchmark demo...")
        if not run_benchmark_demo():
            success = False
    
    if success:
        logger.info("All demos completed successfully!")
        return 0
    else:
        logger.error("Some demos failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
