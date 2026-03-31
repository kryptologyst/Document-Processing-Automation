# Document Processing Automation

A comprehensive document processing automation system for extracting structured data from unstructured documents like invoices, contracts, receipts, and forms. This system is designed for research and educational purposes with a focus on explainability, evaluation, and human-in-loop validation.

## ⚠️ Important Disclaimer

**This is a research and educational demonstration system.** It is not intended for automated decision-making without human review. All results should be validated by qualified professionals before use in production environments.

## Features

### Core Functionality
- **Document Processing**: Extract structured data from unstructured documents
- **Multiple Document Types**: Support for invoices, receipts, contracts, forms, and more
- **OCR Integration**: Text extraction from images and scanned documents
- **Layout Parsing**: Document structure analysis and text organization
- **Named Entity Recognition**: Advanced field extraction using NLP techniques

### Advanced Capabilities
- **Document Classification**: Automatic document type identification
- **Field Extraction**: Intelligent extraction of key fields with confidence scoring
- **Batch Processing**: Efficient processing of multiple documents
- **Confidence Estimation**: Comprehensive confidence scoring for all extractions
- **Human-in-Loop**: Integration points for human validation and review

### Evaluation & Analysis
- **Comprehensive Evaluation**: Multiple metrics for accuracy, performance, and reliability
- **Benchmarking**: Performance testing and scalability analysis
- **Explainability**: Detailed explanations for processing decisions
- **Visualization**: Interactive dashboards and performance charts
- **Reporting**: Automated report generation and analysis

## Supported Document Types

| Document Type | Key Fields | Description |
|---------------|------------|-------------|
| **Invoice** | Invoice number, date, customer, total amount, tax | Business invoices and billing documents |
| **Receipt** | Receipt number, date, merchant, total, payment method | Purchase receipts and transaction records |
| **Contract** | Contract ID, date, parties, value, term, status | Legal contracts and agreements |
| **Form** | Form fields, dates, signatures, checkboxes | Various forms and applications |
| **RFP** | RFP number, deadline, requirements, budget | Request for Proposal documents |
| **Proposal** | Proposal ID, client, scope, timeline, cost | Business proposals and bids |
| **Statement** | Statement period, account, balance, transactions | Financial statements and reports |
| **Certificate** | Certificate number, issuer, date, validity | Certificates and credentials |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Document-Processing-Automation.git
   cd Document-Processing-Automation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optional dependencies (for advanced features):**
   ```bash
   # For OCR functionality
   pip install pytesseract
   
   # For NLP features
   pip install spacy
   python -m spacy download en_core_web_sm
   
   # For ML tracking
   pip install -e ".[ml-tracking]"
   
   # For advanced NLP
   pip install -e ".[advanced-nlp]"
   ```

### Basic Usage

1. **Run the demo application:**
   ```bash
   streamlit run demo/app.py
   ```

2. **Process a single document:**
   ```python
   from src.processing.document_processor import DocumentProcessor
   
   # Initialize processor
   processor = DocumentProcessor(confidence_threshold=0.7)
   
   # Process document
   text = "Invoice #12345\nDate: 2024-01-15\nCustomer: ABC Corp\nTotal: $1,100.00"
   result = processor.process_document(text)
   
   # Display results
   print(f"Document Type: {result.document_type}")
   print(f"Confidence: {result.overall_confidence:.3f}")
   for field in result.fields:
       print(f"{field.name}: {field.value} (confidence: {field.confidence:.3f})")
   ```

3. **Batch processing:**
   ```python
   from src.data.data_generator import DocumentDataGenerator
   
   # Generate sample documents
   generator = DocumentDataGenerator(seed=42)
   documents = generator.generate_text_batch(DocumentType.INVOICE, 10)
   
   # Process batch
   results = processor.process_batch(documents)
   
   # Analyze results
   for result in results:
       print(f"Processed {result.document_id} with confidence {result.overall_confidence:.3f}")
   ```

## Project Structure

```
document-processing-automation/
├── src/                          # Source code
│   ├── processing/               # Document processing modules
│   │   ├── document_processor.py # Main document processor
│   │   ├── ocr_processor.py      # OCR functionality
│   │   └── layout_parser.py      # Layout analysis
│   ├── models/                   # ML models and classifiers
│   │   ├── document_classifier.py # Document type classification
│   │   ├── field_extractor.py    # Field extraction models
│   │   └── confidence_estimator.py # Confidence estimation
│   ├── data/                     # Data handling and generation
│   │   ├── data_generator.py     # Synthetic data generation
│   │   ├── data_loader.py        # Data loading utilities
│   │   └── schema.py             # Data schemas and validation
│   ├── eval/                     # Evaluation and metrics
│   │   ├── evaluator.py          # Main evaluator
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── benchmark.py          # Benchmarking suite
│   ├── viz/                      # Visualization and explainability
│   │   ├── visualizer.py         # Visualization utilities
│   │   ├── explainability.py     # Explainability engine
│   │   └── dashboard.py          # Dashboard generation
│   └── utils/                    # Utility functions
├── demo/                         # Demo application
│   ├── app.py                    # Streamlit demo app
│   └── requirements.txt          # Demo dependencies
├── data/                         # Data directory
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test files
├── assets/                       # Generated assets and reports
├── requirements.txt              # Main dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Document processing settings
CONFIDENCE_THRESHOLD=0.7
HUMAN_REVIEW_THRESHOLD=0.5
MAX_PROCESSING_TIME=30.0

# OCR settings
TESSERACT_PATH=/usr/bin/tesseract  # Linux/Mac
# TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe  # Windows

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/document_processing.log

# Data paths
DATA_DIR=data/
OUTPUT_DIR=output/
MODELS_DIR=models/
```

### Configuration Files

Configuration files are stored in the `configs/` directory:

- `config.yaml`: Main configuration file
- `document_types.yaml`: Document type definitions
- `field_patterns.yaml`: Field extraction patterns
- `evaluation.yaml`: Evaluation parameters

## Evaluation

### Running Evaluations

1. **Quick evaluation:**
   ```bash
   python scripts/evaluate.py --quick --output results/quick_eval/
   ```

2. **Comprehensive evaluation:**
   ```bash
   python scripts/evaluate.py --comprehensive --output results/full_eval/
   ```

3. **Benchmarking:**
   ```bash
   python scripts/benchmark.py --sizes 10,50,100,500 --output results/benchmark/
   ```

### Evaluation Metrics

The system provides comprehensive evaluation metrics:

#### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each document type
- **Recall**: Recall for each document type
- **F1-Score**: Harmonic mean of precision and recall

#### Field Extraction Metrics
- **Field Precision**: Accuracy of individual field extractions
- **Field Recall**: Completeness of field extraction
- **Field F1-Score**: Combined field extraction performance
- **Field Completeness**: Percentage of expected fields extracted

#### Performance Metrics
- **Processing Time**: Time to process each document
- **Throughput**: Documents processed per second
- **Memory Usage**: System resource utilization
- **Scalability**: Performance with increasing document volume

#### Confidence Metrics
- **Expected Calibration Error (ECE)**: Confidence calibration quality
- **Brier Score**: Overall confidence accuracy
- **Reliability Diagram**: Visual confidence calibration
- **Threshold Analysis**: Optimal confidence thresholds

## Explainability

### Understanding Decisions

The system provides detailed explanations for all processing decisions:

1. **Confidence Factors**: Breakdown of factors contributing to confidence scores
2. **Field Extraction Rationale**: Explanation of why fields were extracted or missed
3. **Document Classification Logic**: Reasoning behind document type classification
4. **Human Review Recommendations**: When and why human review is recommended

### Human-in-Loop Integration

- **Confidence Thresholds**: Automatic routing based on confidence scores
- **Review Queues**: Prioritized queues for human review
- **Feedback Integration**: Learning from human corrections
- **Audit Trails**: Complete history of processing decisions

## API Reference

### DocumentProcessor

Main class for document processing.

```python
class DocumentProcessor:
    def __init__(self, confidence_threshold: float = 0.7)
    def process_document(self, text: str, document_id: str = None) -> DocumentResult
    def process_batch(self, documents: List[str], document_ids: List[str] = None) -> List[DocumentResult]
    def classify_document_type(self, text: str) -> Tuple[str, float]
    def extract_fields(self, text: str, document_type: str) -> List[DocumentField]
```

### DocumentResult

Result of document processing.

```python
@dataclass
class DocumentResult:
    document_id: str
    document_type: str
    fields: List[DocumentField]
    overall_confidence: float
    processing_time: float
    raw_text: str
```

### DocumentField

Extracted field information.

```python
@dataclass
class DocumentField:
    name: str
    value: Any
    confidence: float
    position: Optional[Tuple[int, int]] = None
    source_text: Optional[str] = None
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only
```

### Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for system workflows
- `tests/fixtures/`: Test data and fixtures
- `tests/conftest.py`: Pytest configuration and shared fixtures

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```
4. **Run pre-commit hooks:**
   ```bash
   pre-commit install
   ```
5. **Make your changes and test:**
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   ```

### Code Style

- **Formatting**: Black with line length 88
- **Linting**: Ruff for code quality
- **Type Hints**: Required for all functions
- **Documentation**: Google-style docstrings
- **Testing**: Comprehensive test coverage

### Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass** and coverage is maintained
4. **Update CHANGELOG.md** with your changes
5. **Submit pull request** with clear description

## Performance

### Benchmarks

Performance benchmarks are available in the `assets/benchmarks/` directory:

- **Processing Speed**: ~50-100 documents/second (depending on complexity)
- **Memory Usage**: ~100-500MB (depending on batch size)
- **Accuracy**: 85-95% (depending on document type and quality)
- **Confidence Calibration**: ECE < 0.1 for well-calibrated models

### Optimization Tips

1. **Batch Processing**: Process documents in batches for better throughput
2. **Caching**: Cache model predictions for repeated documents
3. **Parallel Processing**: Use multiple workers for large batches
4. **Model Optimization**: Use quantized models for faster inference
5. **Resource Management**: Monitor memory usage for large document sets

## Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Ensure Tesseract is installed and in PATH
   - Check image quality and format
   - Verify language packs are installed

2. **Low Accuracy**
   - Check document quality and preprocessing
   - Verify field patterns match document format
   - Consider retraining with domain-specific data

3. **Slow Processing**
   - Reduce batch size
   - Enable parallel processing
   - Check system resources

4. **Memory Issues**
   - Reduce batch size
   - Clear cache between batches
   - Monitor memory usage

### Getting Help

- **Documentation**: Check this README and inline documentation
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to the development team

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SpaCy**: For NLP capabilities and named entity recognition
- **Tesseract**: For OCR functionality
- **Streamlit**: For the demo application interface
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning utilities
- **Pandas**: For data manipulation and analysis

## Changelog

### Version 1.0.0 (2024-01-01)
- Initial release
- Core document processing functionality
- Support for multiple document types
- Evaluation and benchmarking suite
- Explainability and human-in-loop features
- Interactive demo application
- Comprehensive documentation

## Roadmap

### Version 1.1.0 (Planned)
- [ ] Advanced OCR preprocessing
- [ ] Multi-language support
- [ ] Cloud deployment options
- [ ] Real-time processing API
- [ ] Enhanced visualization tools

### Version 1.2.0 (Planned)
- [ ] Custom model training
- [ ] Advanced field validation
- [ ] Integration with document management systems
- [ ] Performance optimization
- [ ] Extended document type support


# Document-Processing-Automation
