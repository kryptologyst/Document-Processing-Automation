#!/usr/bin/env python3
"""
Quickstart Script

This script provides a quick way to get started with the document processing
automation system, demonstrating key features and capabilities.
"""

import sys
from pathlib import Path
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


def main():
    """Main quickstart function."""
    print("=" * 80)
    print("DOCUMENT PROCESSING AUTOMATION - QUICKSTART")
    print("=" * 80)
    print()
    
    # Initialize components
    logger.info("Initializing document processing system...")
    processor = DocumentProcessor(confidence_threshold=0.7)
    data_generator = DocumentDataGenerator(seed=42)
    
    # Generate sample documents
    logger.info("Generating sample documents...")
    sample_documents = []
    
    for doc_type in [DocumentType.INVOICE, DocumentType.RECEIPT, DocumentType.CONTRACT]:
        texts = data_generator.generate_text_batch(doc_type, 2)
        sample_documents.extend(texts)
    
    # Process documents
    logger.info("Processing documents...")
    results = processor.process_batch(sample_documents)
    
    # Display results
    print("PROCESSING RESULTS:")
    print("-" * 50)
    
    for result in results:
        print(f"\nDocument ID: {result.document_id}")
        print(f"Type: {result.document_type}")
        print(f"Confidence: {result.overall_confidence:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print("Extracted Fields:")
        
        for field in result.fields:
            print(f"  {field.name}: {field.value} (confidence: {field.confidence:.3f})")
    
    # Summary statistics
    total_docs = len(results)
    avg_confidence = sum(r.overall_confidence for r in results) / total_docs
    avg_time = sum(r.processing_time for r in results) / total_docs
    
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS:")
    print(f"Total Documents: {total_docs}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Processing Time: {avg_time:.3f}s")
    print(f"Documents per Second: {total_docs / sum(r.processing_time for r in results):.2f}")
    
    # Run evaluation
    logger.info("Running evaluation...")
    evaluator = DocumentProcessingEvaluator()
    
    # Prepare test data
    test_documents = []
    ground_truth = []
    
    for doc_type in DocumentType:
        texts = data_generator.generate_text_batch(doc_type, 3)
        data = data_generator.generate_batch(doc_type, 3)
        
        results = processor.process_batch(texts)
        
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
    
    # Display evaluation results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    
    if 'classification' in evaluation_results:
        cls_results = evaluation_results['classification']
        print(f"Classification Accuracy: {cls_results.get('accuracy', 0):.3f}")
        print(f"Classification Precision: {cls_results.get('precision', 0):.3f}")
        print(f"Classification Recall: {cls_results.get('recall', 0):.3f}")
        print(f"Classification F1-Score: {cls_results.get('f1_score', 0):.3f}")
    
    if 'processing_performance' in evaluation_results:
        perf_results = evaluation_results['processing_performance']
        print(f"Average Processing Time: {perf_results.get('avg_processing_time', 0):.3f}s")
        print(f"Documents per Second: {perf_results.get('documents_per_second', 0):.2f}")
    
    if 'summary' in evaluation_results:
        summary = evaluation_results['summary']
        print(f"Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
    
    # Next steps
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("1. Run the Streamlit demo: streamlit run demo/app.py")
    print("2. Explore the documentation: README.md")
    print("3. Run evaluations: python scripts/evaluate.py --quick")
    print("4. Check out the examples in the notebooks/ directory")
    print("5. Customize the configuration in configs/config.yaml")
    
    print("\n" + "=" * 50)
    print("QUICKSTART COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
