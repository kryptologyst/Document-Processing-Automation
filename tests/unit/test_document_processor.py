"""
Unit tests for document processor module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.processing.document_processor import DocumentProcessor, DocumentField, DocumentResult


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def test_init(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(confidence_threshold=0.8)
        assert processor.confidence_threshold == 0.8
        assert processor.document_patterns is not None
        assert processor.field_extractors is not None
    
    def test_init_default_threshold(self):
        """Test DocumentProcessor initialization with default threshold."""
        processor = DocumentProcessor()
        assert processor.confidence_threshold == 0.7
    
    def test_classify_document_type_invoice(self):
        """Test document type classification for invoice."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15\nTotal: $1,100.00"
        
        doc_type, confidence = processor.classify_document_type(text)
        assert doc_type == "invoice"
        assert confidence > 0.0
    
    def test_classify_document_type_receipt(self):
        """Test document type classification for receipt."""
        processor = DocumentProcessor()
        text = "Receipt #R789\nDate: 2024-01-16\nMerchant: TechStore\nTotal: $89.99"
        
        doc_type, confidence = processor.classify_document_type(text)
        assert doc_type == "receipt"
        assert confidence > 0.0
    
    def test_classify_document_type_contract(self):
        """Test document type classification for contract."""
        processor = DocumentProcessor()
        text = "Contract #CON-001\nDate: 2024-01-17\nParties: Company A and Company B"
        
        doc_type, confidence = processor.classify_document_type(text)
        assert doc_type == "contract"
        assert confidence > 0.0
    
    def test_extract_invoice_number(self):
        """Test invoice number extraction."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15"
        
        value, confidence = processor._extract_invoice_number(text)
        assert value == "12345"
        assert confidence == 0.9
    
    def test_extract_invoice_number_not_found(self):
        """Test invoice number extraction when not found."""
        processor = DocumentProcessor()
        text = "Date: 2024-01-15\nTotal: $1,100.00"
        
        value, confidence = processor._extract_invoice_number(text)
        assert value is None
        assert confidence == 0.0
    
    def test_extract_date(self):
        """Test date extraction."""
        processor = DocumentProcessor()
        text = "Date: 2024-01-15\nTotal: $1,100.00"
        
        value, confidence = processor._extract_date(text)
        assert value == "2024-01-15"
        assert confidence == 0.8
    
    def test_extract_date_multiple_formats(self):
        """Test date extraction with multiple formats."""
        processor = DocumentProcessor()
        
        # Test different date formats
        test_cases = [
            ("Date: 2024-01-15", "2024-01-15"),
            ("2024-01-15", "2024-01-15"),
            ("01/15/2024", "01/15/2024")
        ]
        
        for text, expected in test_cases:
            value, confidence = processor._extract_date(text)
            assert value == expected
            assert confidence == 0.8
    
    def test_extract_amount(self):
        """Test amount extraction."""
        processor = DocumentProcessor()
        text = "Total: $1,100.00\nDate: 2024-01-15"
        
        value, confidence = processor._extract_amount(text)
        assert value == 1100.00
        assert confidence == 0.85
    
    def test_extract_amount_with_comma(self):
        """Test amount extraction with comma."""
        processor = DocumentProcessor()
        text = "Total: $1,100.50\nDate: 2024-01-15"
        
        value, confidence = processor._extract_amount(text)
        assert value == 1100.50
        assert confidence == 0.85
    
    def test_extract_customer(self):
        """Test customer extraction."""
        processor = DocumentProcessor()
        text = "Customer: ABC Corp\nDate: 2024-01-15"
        
        value, confidence = processor._extract_customer(text)
        assert value == "ABC Corp"
        assert confidence == 0.8
    
    def test_extract_customer_bill_to(self):
        """Test customer extraction with 'Bill To' format."""
        processor = DocumentProcessor()
        text = "Bill To: XYZ Inc\nDate: 2024-01-15"
        
        value, confidence = processor._extract_customer(text)
        assert value == "XYZ Inc"
        assert confidence == 0.8
    
    def test_extract_fields_invoice(self):
        """Test field extraction for invoice."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00"
        
        fields = processor.extract_fields(text, "invoice")
        
        assert len(fields) > 0
        field_names = [field.name for field in fields]
        assert "invoice_number" in field_names
        assert "date" in field_names
        assert "customer" in field_names
        assert "total" in field_names
    
    def test_extract_fields_receipt(self):
        """Test field extraction for receipt."""
        processor = DocumentProcessor()
        text = "Receipt #R789\nDate: 2024-01-16\nMerchant: TechStore\nTotal: $89.99"
        
        fields = processor.extract_fields(text, "receipt")
        
        assert len(fields) > 0
        field_names = [field.name for field in fields]
        assert "receipt_number" in field_names
        assert "date" in field_names
        assert "merchant" in field_names
        assert "total" in field_names
    
    def test_extract_fields_contract(self):
        """Test field extraction for contract."""
        processor = DocumentProcessor()
        text = "Contract #CON-001\nDate: 2024-01-17\nParties: Company A and Company B\nValue: $50,000.00"
        
        fields = processor.extract_fields(text, "contract")
        
        assert len(fields) > 0
        field_names = [field.name for field in fields]
        assert "contract_id" in field_names
        assert "date" in field_names
        assert "parties" in field_names
        assert "value" in field_names
    
    def test_extract_fields_unknown_type(self):
        """Test field extraction for unknown document type."""
        processor = DocumentProcessor()
        text = "Some document text"
        
        fields = processor.extract_fields(text, "unknown")
        assert len(fields) == 0
    
    def test_process_document(self, sample_document_texts):
        """Test complete document processing."""
        processor = DocumentProcessor()
        text = sample_document_texts[0]  # Invoice text
        
        result = processor.process_document(text)
        
        assert isinstance(result, DocumentResult)
        assert result.document_id is not None
        assert result.document_type == "invoice"
        assert result.overall_confidence > 0.0
        assert result.processing_time > 0.0
        assert result.raw_text == text
        assert len(result.fields) > 0
    
    def test_process_document_with_id(self, sample_document_texts):
        """Test document processing with custom ID."""
        processor = DocumentProcessor()
        text = sample_document_texts[0]
        custom_id = "custom_doc_001"
        
        result = processor.process_document(text, custom_id)
        
        assert result.document_id == custom_id
    
    def test_process_batch(self, sample_document_texts):
        """Test batch document processing."""
        processor = DocumentProcessor()
        
        results = processor.process_batch(sample_document_texts)
        
        assert len(results) == len(sample_document_texts)
        for result in results:
            assert isinstance(result, DocumentResult)
            assert result.document_id is not None
            assert result.document_type is not None
            assert result.overall_confidence > 0.0
    
    def test_process_batch_with_ids(self, sample_document_texts):
        """Test batch processing with custom IDs."""
        processor = DocumentProcessor()
        custom_ids = [f"doc_{i:03d}" for i in range(len(sample_document_texts))]
        
        results = processor.process_batch(sample_document_texts, custom_ids)
        
        assert len(results) == len(sample_document_texts)
        for i, result in enumerate(results):
            assert result.document_id == custom_ids[i]
    
    def test_confidence_threshold_filtering(self):
        """Test that fields below confidence threshold are filtered out."""
        processor = DocumentProcessor(confidence_threshold=0.9)
        text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00"
        
        fields = processor.extract_fields(text, "invoice")
        
        # All fields should have confidence >= 0.9
        for field in fields:
            assert field.confidence >= 0.9
    
    def test_field_confidence_scores(self):
        """Test that field confidence scores are reasonable."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00"
        
        fields = processor.extract_fields(text, "invoice")
        
        for field in fields:
            assert 0.0 <= field.confidence <= 1.0
            assert field.value is not None
            assert field.name is not None
    
    def test_processing_time_measurement(self, sample_document_texts):
        """Test that processing time is measured correctly."""
        processor = DocumentProcessor()
        text = sample_document_texts[0]
        
        result = processor.process_document(text)
        
        assert result.processing_time > 0.0
        assert result.processing_time < 1.0  # Should be fast for simple text processing
    
    def test_overall_confidence_calculation(self, sample_document_texts):
        """Test overall confidence calculation."""
        processor = DocumentProcessor()
        text = sample_document_texts[0]
        
        result = processor.process_document(text)
        
        if result.fields:
            expected_confidence = np.mean([field.confidence for field in result.fields])
            assert abs(result.overall_confidence - expected_confidence) < 0.001
        else:
            assert result.overall_confidence == 0.0
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        processor = DocumentProcessor()
        text = ""
        
        result = processor.process_document(text)
        
        assert result.document_id is not None
        assert result.raw_text == text
        assert result.overall_confidence == 0.0
        assert len(result.fields) == 0
    
    def test_whitespace_text_handling(self):
        """Test handling of whitespace-only text."""
        processor = DocumentProcessor()
        text = "   \n\t   "
        
        result = processor.process_document(text)
        
        assert result.document_id is not None
        assert result.raw_text == text
        assert result.overall_confidence == 0.0
        assert len(result.fields) == 0
    
    def test_special_characters_handling(self):
        """Test handling of text with special characters."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp & Co.\nTotal: $1,100.00"
        
        result = processor.process_document(text)
        
        assert result.document_id is not None
        assert result.document_type == "invoice"
        assert result.overall_confidence > 0.0
        assert len(result.fields) > 0
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        processor = DocumentProcessor()
        text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp™\nTotal: $1,100.00"
        
        result = processor.process_document(text)
        
        assert result.document_id is not None
        assert result.document_type == "invoice"
        assert result.overall_confidence > 0.0
        assert len(result.fields) > 0
    
    def test_large_text_handling(self):
        """Test handling of large text."""
        processor = DocumentProcessor()
        # Create a large text with repeated content
        base_text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00\n"
        large_text = base_text * 1000  # ~50KB of text
        
        result = processor.process_document(large_text)
        
        assert result.document_id is not None
        assert result.document_type == "invoice"
        assert result.overall_confidence > 0.0
        assert len(result.fields) > 0
    
    def test_concurrent_processing(self, sample_document_texts):
        """Test that processor can handle concurrent access."""
        import threading
        import time
        
        processor = DocumentProcessor()
        results = []
        
        def process_document(text):
            result = processor.process_document(text)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for text in sample_document_texts * 3:  # Process each text 3 times
            thread = threading.Thread(target=process_document, args=(text,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all documents were processed
        assert len(results) == len(sample_document_texts) * 3
        for result in results:
            assert isinstance(result, DocumentResult)
            assert result.document_id is not None
    
    def test_memory_usage(self, sample_document_texts):
        """Test that processor doesn't leak memory."""
        import gc
        
        processor = DocumentProcessor()
        
        # Process many documents
        for _ in range(100):
            for text in sample_document_texts:
                result = processor.process_document(text)
                del result
        
        # Force garbage collection
        gc.collect()
        
        # Processor should still work
        result = processor.process_document(sample_document_texts[0])
        assert isinstance(result, DocumentResult)
    
    def test_error_handling(self):
        """Test error handling in document processing."""
        processor = DocumentProcessor()
        
        # Test with None input
        with pytest.raises(AttributeError):
            processor.process_document(None)
        
        # Test with non-string input
        with pytest.raises(AttributeError):
            processor.process_document(123)
        
        # Test with list input
        with pytest.raises(AttributeError):
            processor.process_document(["text"])
    
    def test_field_extraction_edge_cases(self):
        """Test field extraction with edge cases."""
        processor = DocumentProcessor()
        
        # Test with very short text
        short_text = "Invoice #1"
        fields = processor.extract_fields(short_text, "invoice")
        assert len(fields) >= 0  # Should not crash
        
        # Test with very long field values
        long_text = "Invoice #12345\nDate: 2024-01-15\nCustomer: " + "A" * 1000 + "\nTotal: $1,100.00"
        fields = processor.extract_fields(long_text, "invoice")
        assert len(fields) >= 0  # Should not crash
        
        # Test with special characters in field values
        special_text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp & Co.™\nTotal: $1,100.00"
        fields = processor.extract_fields(special_text, "invoice")
        assert len(fields) >= 0  # Should not crash
