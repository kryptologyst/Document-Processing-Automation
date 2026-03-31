"""
Document Processing Core Module

This module provides the main document processing functionality for extracting
structured data from unstructured documents like invoices, contracts, receipts, and forms.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentField:
    """Represents an extracted field from a document."""
    name: str
    value: Any
    confidence: float
    position: Optional[Tuple[int, int]] = None
    source_text: Optional[str] = None


@dataclass
class DocumentResult:
    """Represents the complete extraction result for a document."""
    document_id: str
    document_type: str
    fields: List[DocumentField]
    overall_confidence: float
    processing_time: float
    raw_text: str


class DocumentProcessor:
    """Main document processing class with modern NLP capabilities."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the document processor.
        
        Args:
            confidence_threshold: Minimum confidence for field extraction
        """
        self.confidence_threshold = confidence_threshold
        self.document_patterns = self._load_document_patterns()
        self.field_extractors = self._initialize_extractors()
        
    def _load_document_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load regex patterns for different document types."""
        return {
            'invoice': {
                'invoice_number': r'Invoice\s*#?\s*(\d+)',
                'date': r'Date:\s*([\d\-/]+)',
                'total': r'Total:\s*\$?([\d,]+\.?\d{0,2})',
                'customer': r'Customer:\s*(.+)',
                'due_date': r'Due\s*Date:\s*([\d\-/]+)',
                'tax': r'Tax:\s*\$?([\d,]+\.?\d{0,2})',
                'subtotal': r'Subtotal:\s*\$?([\d,]+\.?\d{0,2})'
            },
            'receipt': {
                'receipt_number': r'Receipt\s*#?\s*(\d+)',
                'date': r'Date:\s*([\d\-/]+)',
                'total': r'Total:\s*\$?([\d,]+\.?\d{0,2})',
                'merchant': r'Merchant:\s*(.+)',
                'items': r'Items?:\s*(.+)',
                'payment_method': r'Payment:\s*(.+)'
            },
            'contract': {
                'contract_id': r'Contract\s*#?\s*(\w+)',
                'date': r'Date:\s*([\d\-/]+)',
                'parties': r'Parties:\s*(.+)',
                'value': r'Value:\s*\$?([\d,]+\.?\d{0,2})',
                'term': r'Term:\s*(.+)',
                'status': r'Status:\s*(.+)'
            }
        }
    
    def _initialize_extractors(self) -> Dict[str, callable]:
        """Initialize field extraction functions."""
        return {
            'invoice_number': self._extract_invoice_number,
            'date': self._extract_date,
            'total': self._extract_amount,
            'customer': self._extract_customer,
            'receipt_number': self._extract_receipt_number,
            'merchant': self._extract_merchant,
            'contract_id': self._extract_contract_id,
            'parties': self._extract_parties
        }
    
    def _extract_invoice_number(self, text: str) -> Tuple[Optional[str], float]:
        """Extract invoice number with confidence scoring."""
        pattern = r'Invoice\s*#?\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), 0.9
        return None, 0.0
    
    def _extract_date(self, text: str) -> Tuple[Optional[str], float]:
        """Extract date with confidence scoring."""
        patterns = [
            r'Date:\s*([\d\-/]+)',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1), 0.8
        return None, 0.0
    
    def _extract_amount(self, text: str) -> Tuple[Optional[float], float]:
        """Extract monetary amount with confidence scoring."""
        patterns = [
            r'Total:\s*\$?([\d,]+\.?\d{0,2})',
            r'\$([\d,]+\.?\d{0,2})',
            r'Amount:\s*\$?([\d,]+\.?\d{0,2})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    amount = float(match.group(1).replace(',', ''))
                    return amount, 0.85
                except ValueError:
                    continue
        return None, 0.0
    
    def _extract_customer(self, text: str) -> Tuple[Optional[str], float]:
        """Extract customer name with confidence scoring."""
        patterns = [
            r'Customer:\s*(.+)',
            r'Bill\s*To:\s*(.+)',
            r'Client:\s*(.+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                customer = match.group(1).strip()
                if len(customer) > 2:  # Basic validation
                    return customer, 0.8
        return None, 0.0
    
    def _extract_receipt_number(self, text: str) -> Tuple[Optional[str], float]:
        """Extract receipt number with confidence scoring."""
        pattern = r'Receipt\s*#?\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), 0.9
        return None, 0.0
    
    def _extract_merchant(self, text: str) -> Tuple[Optional[str], float]:
        """Extract merchant name with confidence scoring."""
        patterns = [
            r'Merchant:\s*(.+)',
            r'Store:\s*(.+)',
            r'Business:\s*(.+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                merchant = match.group(1).strip()
                if len(merchant) > 2:
                    return merchant, 0.8
        return None, 0.0
    
    def _extract_contract_id(self, text: str) -> Tuple[Optional[str], float]:
        """Extract contract ID with confidence scoring."""
        pattern = r'Contract\s*#?\s*(\w+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1), 0.9
        return None, 0.0
    
    def _extract_parties(self, text: str) -> Tuple[Optional[str], float]:
        """Extract contract parties with confidence scoring."""
        patterns = [
            r'Parties:\s*(.+)',
            r'Between:\s*(.+)',
            r'Contracting\s*Parties:\s*(.+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parties = match.group(1).strip()
                if len(parties) > 5:
                    return parties, 0.8
        return None, 0.0
    
    def classify_document_type(self, text: str) -> Tuple[str, float]:
        """
        Classify document type based on content.
        
        Args:
            text: Raw document text
            
        Returns:
            Tuple of (document_type, confidence)
        """
        text_lower = text.lower()
        
        # Simple keyword-based classification
        invoice_keywords = ['invoice', 'bill', 'amount due', 'payment terms']
        receipt_keywords = ['receipt', 'purchase', 'sale', 'merchant']
        contract_keywords = ['contract', 'agreement', 'terms', 'parties']
        
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        receipt_score = sum(1 for keyword in receipt_keywords if keyword in text_lower)
        contract_score = sum(1 for keyword in contract_keywords if keyword in text_lower)
        
        scores = {
            'invoice': invoice_score,
            'receipt': receipt_score,
            'contract': contract_score
        }
        
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # Normalize confidence
        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.0
        
        return max_type, confidence
    
    def extract_fields(self, text: str, document_type: str) -> List[DocumentField]:
        """
        Extract fields from document text.
        
        Args:
            text: Raw document text
            document_type: Type of document (invoice, receipt, contract)
            
        Returns:
            List of extracted DocumentField objects
        """
        fields = []
        
        if document_type not in self.document_patterns:
            logger.warning(f"Unknown document type: {document_type}")
            return fields
        
        patterns = self.document_patterns[document_type]
        
        for field_name, pattern in patterns.items():
            if field_name in self.field_extractors:
                extractor = self.field_extractors[field_name]
                value, confidence = extractor(text)
                
                if confidence >= self.confidence_threshold:
                    field = DocumentField(
                        name=field_name,
                        value=value,
                        confidence=confidence,
                        source_text=text
                    )
                    fields.append(field)
        
        return fields
    
    def process_document(self, text: str, document_id: str = None) -> DocumentResult:
        """
        Process a complete document and extract all relevant information.
        
        Args:
            text: Raw document text
            document_id: Optional document identifier
            
        Returns:
            DocumentResult object with all extracted information
        """
        start_time = datetime.now()
        
        if document_id is None:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Classify document type
        doc_type, type_confidence = self.classify_document_type(text)
        
        # Extract fields
        fields = self.extract_fields(text, doc_type)
        
        # Calculate overall confidence
        if fields:
            overall_confidence = np.mean([field.confidence for field in fields])
        else:
            overall_confidence = 0.0
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentResult(
            document_id=document_id,
            document_type=doc_type,
            fields=fields,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            raw_text=text
        )
    
    def process_batch(self, documents: List[str], document_ids: List[str] = None) -> List[DocumentResult]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
            
        Returns:
            List of DocumentResult objects
        """
        if document_ids is None:
            document_ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                          for i in range(len(documents))]
        
        results = []
        for doc_text, doc_id in zip(documents, document_ids):
            result = self.process_document(doc_text, doc_id)
            results.append(result)
        
        return results
