"""
Synthetic Data Generator

This module generates synthetic document data for testing and development
purposes, ensuring privacy and reproducibility.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import json
from pathlib import Path

from .schema import DocumentType, DocumentSchema, get_schema

logger = logging.getLogger(__name__)


class DocumentDataGenerator:
    """Generates synthetic document data for testing and development."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize Faker for realistic data generation
        self.fake = Faker()
        Faker.seed(seed)
        
        # Template data for different document types
        self.templates = self._load_templates()
        
        # Field generators
        self.field_generators = self._initialize_field_generators()
    
    def _load_templates(self) -> Dict[DocumentType, List[str]]:
        """Load document templates for different types."""
        return {
            DocumentType.INVOICE: [
                "Invoice #{invoice_number}\nDate: {date}\nCustomer: {customer}\nTotal: ${total:,.2f}",
                "Invoice #{invoice_number}\nDate: {date}\nCustomer: {customer}\nSubtotal: ${subtotal:,.2f}\nTax: ${tax:,.2f}\nTotal: ${total:,.2f}",
                "Invoice #{invoice_number}\nDate: {date}\nDue Date: {due_date}\nCustomer: {customer}\nTotal: ${total:,.2f}\nCurrency: {currency}"
            ],
            DocumentType.RECEIPT: [
                "Receipt #{receipt_number}\nDate: {date}\nMerchant: {merchant}\nTotal: ${total:,.2f}",
                "Receipt #{receipt_number}\nDate: {date}\nMerchant: {merchant}\nItems: {items}\nTotal: ${total:,.2f}\nPayment: {payment_method}",
                "Receipt #{receipt_number}\nDate: {date}\nMerchant: {merchant}\nTotal: ${total:,.2f}\nPayment Method: {payment_method}"
            ],
            DocumentType.CONTRACT: [
                "Contract #{contract_id}\nDate: {date}\nParties: {parties}\nValue: ${value:,.2f}\nTerm: {term}\nStatus: {status}",
                "Contract #{contract_id}\nDate: {date}\nParties: {parties}\nValue: ${value:,.2f}\nStatus: {status}",
                "Contract #{contract_id}\nDate: {date}\nParties: {parties}\nTerm: {term}\nStatus: {status}"
            ]
        }
    
    def _initialize_field_generators(self) -> Dict[str, callable]:
        """Initialize field value generators."""
        return {
            'invoice_number': self._generate_invoice_number,
            'receipt_number': self._generate_receipt_number,
            'contract_id': self._generate_contract_id,
            'date': self._generate_date,
            'due_date': self._generate_due_date,
            'customer': self._generate_customer_name,
            'merchant': self._generate_merchant_name,
            'parties': self._generate_contract_parties,
            'subtotal': self._generate_amount,
            'tax': self._generate_tax_amount,
            'total': self._generate_total_amount,
            'value': self._generate_contract_value,
            'items': self._generate_items,
            'payment_method': self._generate_payment_method,
            'term': self._generate_contract_term,
            'status': self._generate_contract_status,
            'currency': self._generate_currency
        }
    
    def _generate_invoice_number(self) -> str:
        """Generate invoice number."""
        formats = [
            f"INV-{self.fake.year()}-{self.fake.random_int(min=1, max=9999):04d}",
            f"{self.fake.random_int(min=10000, max=99999)}",
            f"INV-{self.fake.random_int(min=1, max=999):03d}",
            f"{self.fake.random_letter().upper()}{self.fake.random_int(min=1000, max=9999)}"
        ]
        return random.choice(formats)
    
    def _generate_receipt_number(self) -> str:
        """Generate receipt number."""
        formats = [
            f"R-{self.fake.year()}-{self.fake.random_int(min=1, max=9999):04d}",
            f"{self.fake.random_int(min=1000, max=99999)}",
            f"REC-{self.fake.random_int(min=1, max=999):03d}",
            f"{self.fake.random_letter().upper()}{self.fake.random_int(min=100, max=999)}"
        ]
        return random.choice(formats)
    
    def _generate_contract_id(self) -> str:
        """Generate contract ID."""
        formats = [
            f"CON-{self.fake.year()}-{self.fake.random_int(min=1, max=9999):04d}",
            f"CT-{self.fake.random_int(min=1, max=999):03d}",
            f"{self.fake.year()}-{self.fake.random_int(min=1, max=999):03d}",
            f"{self.fake.random_letter().upper()}{self.fake.random_int(min=1000, max=9999)}"
        ]
        return random.choice(formats)
    
    def _generate_date(self) -> str:
        """Generate date string."""
        date = self.fake.date_between(start_date='-1y', end_date='today')
        return date.strftime('%Y-%m-%d')
    
    def _generate_due_date(self) -> str:
        """Generate due date (typically 30 days after invoice date)."""
        date = self.fake.date_between(start_date='today', end_date='+60d')
        return date.strftime('%Y-%m-%d')
    
    def _generate_customer_name(self) -> str:
        """Generate customer name."""
        formats = [
            f"{self.fake.company()}",
            f"{self.fake.name()}",
            f"{self.fake.company()} {self.fake.company_suffix()}",
            f"{self.fake.first_name()} {self.fake.last_name()}"
        ]
        return random.choice(formats)
    
    def _generate_merchant_name(self) -> str:
        """Generate merchant name."""
        formats = [
            f"{self.fake.company()}",
            f"{self.fake.company()} {self.fake.company_suffix()}",
            f"{self.fake.first_name()}'s {self.fake.word().title()}",
            f"{self.fake.word().title()} {self.fake.word().title()}"
        ]
        return random.choice(formats)
    
    def _generate_contract_parties(self) -> str:
        """Generate contract parties."""
        party1 = self._generate_customer_name()
        party2 = self._generate_customer_name()
        return f"{party1} and {party2}"
    
    def _generate_amount(self) -> float:
        """Generate monetary amount."""
        return round(random.uniform(10.0, 10000.0), 2)
    
    def _generate_tax_amount(self) -> float:
        """Generate tax amount."""
        return round(random.uniform(1.0, 1000.0), 2)
    
    def _generate_total_amount(self) -> float:
        """Generate total amount."""
        return round(random.uniform(50.0, 15000.0), 2)
    
    def _generate_contract_value(self) -> float:
        """Generate contract value."""
        return round(random.uniform(1000.0, 1000000.0), 2)
    
    def _generate_items(self) -> str:
        """Generate items list."""
        items = [
            "Laptop, Mouse, Keyboard",
            "Coffee, Pastry, Sandwich",
            "Milk, Bread, Eggs, Butter",
            "Shirt, Pants, Shoes",
            "Book, Pen, Notebook",
            "Phone, Case, Charger"
        ]
        return random.choice(items)
    
    def _generate_payment_method(self) -> str:
        """Generate payment method."""
        methods = ["Credit Card", "Debit Card", "Cash", "Check", "Bank Transfer"]
        return random.choice(methods)
    
    def _generate_contract_term(self) -> str:
        """Generate contract term."""
        terms = [
            "12 months",
            "24 months",
            "36 months",
            "1 year",
            "2 years",
            "3 years",
            "Indefinite",
            "6 months"
        ]
        return random.choice(terms)
    
    def _generate_contract_status(self) -> str:
        """Generate contract status."""
        statuses = ["Active", "Pending", "Expired", "Terminated"]
        return random.choice(statuses)
    
    def _generate_currency(self) -> str:
        """Generate currency code."""
        currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
        return random.choice(currencies)
    
    def generate_document_data(self, document_type: DocumentType, 
                             include_optional: bool = True) -> Dict[str, Any]:
        """
        Generate synthetic document data.
        
        Args:
            document_type: Type of document to generate
            include_optional: Whether to include optional fields
            
        Returns:
            Dictionary containing document data
        """
        schema = get_schema(document_type)
        if schema is None:
            raise ValueError(f"No schema found for document type: {document_type}")
        
        data = {}
        
        # Generate required fields
        for field in schema.get_required_fields():
            if field.name in self.field_generators:
                data[field.name] = self.field_generators[field.name]()
        
        # Generate optional fields if requested
        if include_optional:
            optional_fields = [field for field in schema.fields if not field.required]
            for field in optional_fields:
                if field.name in self.field_generators:
                    # Randomly include optional fields (70% chance)
                    if random.random() < 0.7:
                        data[field.name] = self.field_generators[field.name]()
        
        return data
    
    def generate_document_text(self, document_type: DocumentType, 
                             data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate document text from data.
        
        Args:
            document_type: Type of document
            data: Document data (generated if None)
            
        Returns:
            Generated document text
        """
        if data is None:
            data = self.generate_document_data(document_type)
        
        templates = self.templates.get(document_type, [])
        if not templates:
            raise ValueError(f"No templates found for document type: {document_type}")
        
        template = random.choice(templates)
        
        try:
            return template.format(**data)
        except KeyError as e:
            logger.warning(f"Missing field in template: {e}")
            # Fallback to simple format
            return self._format_simple_document(document_type, data)
    
    def _format_simple_document(self, document_type: DocumentType, data: Dict[str, Any]) -> str:
        """Format document in simple format as fallback."""
        lines = []
        
        if document_type == DocumentType.INVOICE:
            lines.append(f"Invoice #{data.get('invoice_number', 'N/A')}")
        elif document_type == DocumentType.RECEIPT:
            lines.append(f"Receipt #{data.get('receipt_number', 'N/A')}")
        elif document_type == DocumentType.CONTRACT:
            lines.append(f"Contract #{data.get('contract_id', 'N/A')}")
        
        for key, value in data.items():
            if key not in ['invoice_number', 'receipt_number', 'contract_id']:
                if isinstance(value, float):
                    lines.append(f"{key.title()}: ${value:,.2f}")
                else:
                    lines.append(f"{key.title()}: {value}")
        
        return "\n".join(lines)
    
    def generate_batch(self, document_type: DocumentType, count: int, 
                      include_optional: bool = True) -> List[Dict[str, Any]]:
        """
        Generate a batch of document data.
        
        Args:
            document_type: Type of document
            count: Number of documents to generate
            include_optional: Whether to include optional fields
            
        Returns:
            List of document data dictionaries
        """
        documents = []
        
        for _ in range(count):
            data = self.generate_document_data(document_type, include_optional)
            documents.append(data)
        
        return documents
    
    def generate_text_batch(self, document_type: DocumentType, count: int,
                           include_optional: bool = True) -> List[str]:
        """
        Generate a batch of document texts.
        
        Args:
            document_type: Type of document
            count: Number of documents to generate
            include_optional: Whether to include optional fields
            
        Returns:
            List of document text strings
        """
        texts = []
        
        for _ in range(count):
            data = self.generate_document_data(document_type, include_optional)
            text = self.generate_document_text(document_type, data)
            texts.append(text)
        
        return texts
    
    def generate_mixed_batch(self, counts: Dict[DocumentType, int],
                           include_optional: bool = True) -> List[Tuple[DocumentType, str, Dict[str, Any]]]:
        """
        Generate a mixed batch of different document types.
        
        Args:
            counts: Dictionary mapping document types to counts
            include_optional: Whether to include optional fields
            
        Returns:
            List of (document_type, text, data) tuples
        """
        results = []
        
        for doc_type, count in counts.items():
            for _ in range(count):
                data = self.generate_document_data(doc_type, include_optional)
                text = self.generate_document_text(doc_type, data)
                results.append((doc_type, text, data))
        
        # Shuffle results
        random.shuffle(results)
        
        return results
    
    def save_generated_data(self, data: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save generated data to file.
        
        Args:
            data: List of document data dictionaries
            filepath: Path to save the data
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.suffix == '.csv':
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Generated data saved to {filepath}")
    
    def load_generated_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load generated data from file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            List of document data dictionaries
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Generated data loaded from {filepath}")
        return data
    
    def create_ground_truth_dataset(self, document_type: DocumentType, count: int,
                                   include_optional: bool = True) -> pd.DataFrame:
        """
        Create a ground truth dataset for evaluation.
        
        Args:
            document_type: Type of document
            count: Number of documents to generate
            include_optional: Whether to include optional fields
            
        Returns:
            DataFrame with document text and ground truth fields
        """
        documents = self.generate_batch(document_type, count, include_optional)
        texts = [self.generate_document_text(document_type, data) for data in documents]
        
        # Create DataFrame
        df_data = []
        for i, (text, data) in enumerate(zip(texts, documents)):
            row = {'document_id': f'doc_{i:04d}', 'text': text, 'document_type': document_type.value}
            row.update(data)
            df_data.append(row)
        
        return pd.DataFrame(df_data)
