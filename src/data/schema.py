"""
Data Schema Definitions

This module defines the canonical data schemas for document processing
automation, including document types, field definitions, and validation rules.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re


class DocumentType(Enum):
    """Enumeration of supported document types."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FORM = "form"
    RFP = "rfp"
    PROPOSAL = "proposal"
    STATEMENT = "statement"
    CERTIFICATE = "certificate"


class FieldType(Enum):
    """Enumeration of supported field types."""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    IDENTIFIER = "identifier"
    PERCENTAGE = "percentage"
    BOOLEAN = "boolean"


@dataclass
class FieldSchema:
    """Schema definition for a document field."""
    name: str
    field_type: FieldType
    required: bool = False
    pattern: Optional[str] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    def validate(self, value: Any) -> bool:
        """
        Validate a field value against the schema.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return not self.required
        
        # Type validation
        if not self._validate_type(value):
            return False
        
        # Pattern validation
        if self.pattern and not re.match(self.pattern, str(value)):
            return False
        
        # Custom validation rules
        for rule_name, rule_value in self.validation_rules.items():
            if not self._apply_validation_rule(rule_name, rule_value, value):
                return False
        
        return True
    
    def _validate_type(self, value: Any) -> bool:
        """Validate value type."""
        if self.field_type == FieldType.TEXT:
            return isinstance(value, str)
        elif self.field_type == FieldType.NUMBER:
            return isinstance(value, (int, float))
        elif self.field_type == FieldType.CURRENCY:
            return isinstance(value, (int, float)) and value >= 0
        elif self.field_type == FieldType.DATE:
            return isinstance(value, str)  # Date strings
        elif self.field_type == FieldType.EMAIL:
            return isinstance(value, str) and '@' in value
        elif self.field_type == FieldType.PHONE:
            return isinstance(value, str) and re.match(r'[\d\-\+\(\)\s]+', value)
        elif self.field_type == FieldType.ADDRESS:
            return isinstance(value, str)
        elif self.field_type == FieldType.IDENTIFIER:
            return isinstance(value, str)
        elif self.field_type == FieldType.PERCENTAGE:
            return isinstance(value, (int, float)) and 0 <= value <= 100
        elif self.field_type == FieldType.BOOLEAN:
            return isinstance(value, bool)
        
        return True
    
    def _apply_validation_rule(self, rule_name: str, rule_value: Any, value: Any) -> bool:
        """Apply custom validation rule."""
        if rule_name == "min_length":
            return len(str(value)) >= rule_value
        elif rule_name == "max_length":
            return len(str(value)) <= rule_value
        elif rule_name == "min_value":
            return float(value) >= rule_value
        elif rule_name == "max_value":
            return float(value) <= rule_value
        elif rule_name == "allowed_values":
            return str(value) in rule_value
        
        return True


@dataclass
class DocumentSchema:
    """Schema definition for a document type."""
    document_type: DocumentType
    fields: List[FieldSchema]
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    def get_field_schema(self, field_name: str) -> Optional[FieldSchema]:
        """Get field schema by name."""
        for field_schema in self.fields:
            if field_schema.name == field_name:
                return field_schema
        return None
    
    def get_required_fields(self) -> List[FieldSchema]:
        """Get list of required fields."""
        return [field for field in self.fields if field.required]
    
    def validate_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate document data against schema.
        
        Args:
            document_data: Document data to validate
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_fields": [],
            "extra_fields": []
        }
        
        # Check required fields
        required_fields = self.get_required_fields()
        for field_schema in required_fields:
            if field_schema.name not in document_data:
                result["missing_fields"].append(field_schema.name)
                result["valid"] = False
        
        # Validate existing fields
        for field_name, value in document_data.items():
            field_schema = self.get_field_schema(field_name)
            
            if field_schema is None:
                result["extra_fields"].append(field_name)
                result["warnings"].append(f"Unknown field: {field_name}")
            else:
                if not field_schema.validate(value):
                    result["errors"].append(f"Invalid value for field '{field_name}': {value}")
                    result["valid"] = False
        
        return result


# Predefined document schemas
INVOICE_SCHEMA = DocumentSchema(
    document_type=DocumentType.INVOICE,
    description="Invoice document schema",
    fields=[
        FieldSchema(
            name="invoice_number",
            field_type=FieldType.IDENTIFIER,
            required=True,
            pattern=r"^[A-Z0-9\-]+$",
            description="Unique invoice identifier",
            examples=["INV-2024-001", "12345", "INV-001"]
        ),
        FieldSchema(
            name="date",
            field_type=FieldType.DATE,
            required=True,
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Invoice date",
            examples=["2024-01-15", "2024-12-31"]
        ),
        FieldSchema(
            name="due_date",
            field_type=FieldType.DATE,
            required=False,
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Payment due date",
            examples=["2024-02-15", "2024-01-31"]
        ),
        FieldSchema(
            name="customer",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"min_length": 2, "max_length": 100},
            description="Customer name or company",
            examples=["ABC Corp", "John Doe", "XYZ Inc"]
        ),
        FieldSchema(
            name="subtotal",
            field_type=FieldType.CURRENCY,
            required=False,
            validation_rules={"min_value": 0},
            description="Subtotal amount before tax",
            examples=[1000.00, 250.50]
        ),
        FieldSchema(
            name="tax",
            field_type=FieldType.CURRENCY,
            required=False,
            validation_rules={"min_value": 0},
            description="Tax amount",
            examples=[100.00, 25.05]
        ),
        FieldSchema(
            name="total",
            field_type=FieldType.CURRENCY,
            required=True,
            validation_rules={"min_value": 0},
            description="Total amount due",
            examples=[1100.00, 275.55]
        ),
        FieldSchema(
            name="currency",
            field_type=FieldType.TEXT,
            required=False,
            validation_rules={"allowed_values": ["USD", "EUR", "GBP", "CAD"]},
            description="Currency code",
            examples=["USD", "EUR", "GBP"]
        )
    ],
    examples=[
        "Invoice #INV-2024-001\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00",
        "Invoice #12345\nDate: 2024-12-31\nCustomer: XYZ Inc\nSubtotal: $1,000.00\nTax: $100.00\nTotal: $1,100.00"
    ]
)

RECEIPT_SCHEMA = DocumentSchema(
    document_type=DocumentType.RECEIPT,
    description="Receipt document schema",
    fields=[
        FieldSchema(
            name="receipt_number",
            field_type=FieldType.IDENTIFIER,
            required=True,
            pattern=r"^[A-Z0-9\-]+$",
            description="Unique receipt identifier",
            examples=["R-2024-001", "789", "REC-001"]
        ),
        FieldSchema(
            name="date",
            field_type=FieldType.DATE,
            required=True,
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Purchase date",
            examples=["2024-01-15", "2024-12-31"]
        ),
        FieldSchema(
            name="merchant",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"min_length": 2, "max_length": 100},
            description="Merchant or store name",
            examples=["TechStore", "Coffee Shop", "Grocery Store"]
        ),
        FieldSchema(
            name="items",
            field_type=FieldType.TEXT,
            required=False,
            description="List of purchased items",
            examples=["Laptop, Mouse", "Coffee, Pastry", "Milk, Bread"]
        ),
        FieldSchema(
            name="total",
            field_type=FieldType.CURRENCY,
            required=True,
            validation_rules={"min_value": 0},
            description="Total amount paid",
            examples=[89.99, 15.50, 45.75]
        ),
        FieldSchema(
            name="payment_method",
            field_type=FieldType.TEXT,
            required=False,
            validation_rules={"allowed_values": ["Cash", "Credit Card", "Debit Card", "Check"]},
            description="Payment method used",
            examples=["Credit Card", "Cash", "Debit Card"]
        )
    ],
    examples=[
        "Receipt #R-2024-001\nDate: 2024-01-15\nMerchant: TechStore\nTotal: $89.99\nPayment: Credit Card",
        "Receipt #789\nDate: 2024-12-31\nMerchant: Coffee Shop\nItems: Coffee, Pastry\nTotal: $15.50"
    ]
)

CONTRACT_SCHEMA = DocumentSchema(
    document_type=DocumentType.CONTRACT,
    description="Contract document schema",
    fields=[
        FieldSchema(
            name="contract_id",
            field_type=FieldType.IDENTIFIER,
            required=True,
            pattern=r"^[A-Z0-9\-]+$",
            description="Unique contract identifier",
            examples=["CON-2024-001", "CT-001", "2024-001"]
        ),
        FieldSchema(
            name="date",
            field_type=FieldType.DATE,
            required=True,
            pattern=r"^\d{4}-\d{2}-\d{2}$",
            description="Contract date",
            examples=["2024-01-15", "2024-12-31"]
        ),
        FieldSchema(
            name="parties",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"min_length": 10, "max_length": 500},
            description="Contracting parties",
            examples=["Company A and Company B", "John Doe and Jane Smith"]
        ),
        FieldSchema(
            name="value",
            field_type=FieldType.CURRENCY,
            required=False,
            validation_rules={"min_value": 0},
            description="Contract value",
            examples=[50000.00, 100000.00]
        ),
        FieldSchema(
            name="term",
            field_type=FieldType.TEXT,
            required=False,
            description="Contract term or duration",
            examples=["12 months", "2 years", "Indefinite"]
        ),
        FieldSchema(
            name="status",
            field_type=FieldType.TEXT,
            required=False,
            validation_rules={"allowed_values": ["Active", "Pending", "Expired", "Terminated"]},
            description="Contract status",
            examples=["Active", "Pending", "Expired"]
        )
    ],
    examples=[
        "Contract #CON-2024-001\nDate: 2024-01-15\nParties: Company A and Company B\nValue: $50,000.00\nTerm: 12 months\nStatus: Active",
        "Contract #CT-001\nDate: 2024-12-31\nParties: John Doe and Jane Smith\nValue: $100,000.00\nStatus: Pending"
    ]
)

# Schema registry
DOCUMENT_SCHEMAS = {
    DocumentType.INVOICE: INVOICE_SCHEMA,
    DocumentType.RECEIPT: RECEIPT_SCHEMA,
    DocumentType.CONTRACT: CONTRACT_SCHEMA
}


def get_schema(document_type: DocumentType) -> Optional[DocumentSchema]:
    """
    Get document schema by type.
    
    Args:
        document_type: Type of document
        
    Returns:
        Document schema or None if not found
    """
    return DOCUMENT_SCHEMAS.get(document_type)


def get_all_schemas() -> Dict[DocumentType, DocumentSchema]:
    """
    Get all available document schemas.
    
    Returns:
        Dictionary mapping document types to schemas
    """
    return DOCUMENT_SCHEMAS.copy()


def validate_document_data(document_type: DocumentType, document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate document data against its schema.
    
    Args:
        document_type: Type of document
        document_data: Document data to validate
        
    Returns:
        Validation result dictionary
    """
    schema = get_schema(document_type)
    if schema is None:
        return {
            "valid": False,
            "errors": [f"No schema found for document type: {document_type}"],
            "warnings": [],
            "missing_fields": [],
            "extra_fields": []
        }
    
    return schema.validate_document(document_data)
