"""
Data Loading Module

This module provides utilities for loading and preprocessing document data
from various sources and formats.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from datetime import datetime
import re

from .schema import DocumentType, DocumentSchema, get_schema, validate_document_data

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for document processing automation."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_csv(self, filepath: str, text_column: str = "text",
                     document_type_column: Optional[str] = None) -> pd.DataFrame:
        """
        Load document data from CSV file.
        
        Args:
            filepath: Path to CSV file
            text_column: Name of column containing document text
            document_type_column: Name of column containing document type
            
        Returns:
            DataFrame with document data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            # Clean and validate data
            df = self._clean_dataframe(df, text_column, document_type_column)
            
            logger.info(f"Loaded {len(df)} documents from CSV: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV file {filepath}: {e}")
            raise
    
    def load_from_json(self, filepath: str, text_key: str = "text",
                      document_type_key: Optional[str] = None) -> pd.DataFrame:
        """
        Load document data from JSON file.
        
        Args:
            filepath: Path to JSON file
            text_key: Key containing document text
            document_type_key: Key containing document type
            
        Returns:
            DataFrame with document data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                # Single document
                df = pd.DataFrame([data])
            
            # Clean and validate data
            df = self._clean_dataframe(df, text_key, document_type_key)
            
            logger.info(f"Loaded {len(df)} documents from JSON: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load JSON file {filepath}: {e}")
            raise
    
    def load_from_text_files(self, directory: str, file_pattern: str = "*.txt",
                           document_type: Optional[DocumentType] = None) -> pd.DataFrame:
        """
        Load document data from text files.
        
        Args:
            directory: Directory containing text files
            file_pattern: Pattern to match text files
            document_type: Document type (if known)
            
        Returns:
            DataFrame with document data
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        try:
            text_files = list(directory.glob(file_pattern))
            if not text_files:
                raise ValueError(f"No text files found matching pattern: {file_pattern}")
            
            documents = []
            for file_path in text_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:  # Only include non-empty files
                    documents.append({
                        'document_id': file_path.stem,
                        'text': text,
                        'document_type': document_type.value if document_type else 'unknown',
                        'file_path': str(file_path)
                    })
            
            df = pd.DataFrame(documents)
            
            logger.info(f"Loaded {len(df)} documents from text files in: {directory}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load text files from {directory}: {e}")
            raise
    
    def load_ground_truth_data(self, filepath: str) -> pd.DataFrame:
        """
        Load ground truth data for evaluation.
        
        Args:
            filepath: Path to ground truth data file
            
        Returns:
            DataFrame with ground truth data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = self.load_from_csv(filepath)
        elif filepath.suffix == '.json':
            df = self.load_from_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Validate ground truth data structure
        required_columns = ['text', 'document_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in ground truth data: {missing_columns}")
        
        logger.info(f"Loaded ground truth data: {len(df)} documents")
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame, text_column: str,
                        document_type_column: Optional[str]) -> pd.DataFrame:
        """
        Clean and validate DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            document_type_column: Name of document type column
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with empty text
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        
        # Clean text column
        df[text_column] = df[text_column].astype(str).str.strip()
        
        # Add document IDs if not present
        if 'document_id' not in df.columns:
            df['document_id'] = [f'doc_{i:04d}' for i in range(len(df))]
        
        # Validate document types if column exists
        if document_type_column and document_type_column in df.columns:
            df[document_type_column] = df[document_type_column].astype(str)
            
            # Map common variations to standard types
            type_mapping = {
                'invoice': 'invoice',
                'receipt': 'receipt',
                'contract': 'contract',
                'form': 'form',
                'rfp': 'rfp',
                'proposal': 'proposal',
                'statement': 'statement',
                'certificate': 'certificate'
            }
            
            df[document_type_column] = df[document_type_column].str.lower().map(type_mapping).fillna('unknown')
        
        return df.reset_index(drop=True)
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                  validation_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df.get('document_type')
        )
        
        # Second split: separate validation set from remaining data
        val_size = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df.get('document_type')
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_evaluation_dataset(self, df: pd.DataFrame, 
                                ground_truth_columns: List[str]) -> pd.DataFrame:
        """
        Create evaluation dataset with ground truth fields.
        
        Args:
            df: Input DataFrame
            ground_truth_columns: List of column names containing ground truth data
            
        Returns:
            DataFrame with evaluation data
        """
        eval_df = df.copy()
        
        # Ensure all ground truth columns exist
        missing_columns = [col for col in ground_truth_columns if col not in eval_df.columns]
        if missing_columns:
            logger.warning(f"Missing ground truth columns: {missing_columns}")
        
        # Add evaluation metadata
        eval_df['evaluation_timestamp'] = datetime.now().isoformat()
        eval_df['has_ground_truth'] = eval_df[ground_truth_columns].notna().any(axis=1)
        
        return eval_df
    
    def validate_document_schemas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate document data against schemas.
        
        Args:
            df: DataFrame with document data
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'total_documents': len(df),
            'valid_documents': 0,
            'invalid_documents': 0,
            'validation_errors': [],
            'document_type_counts': {},
            'field_completeness': {}
        }
        
        # Count document types
        if 'document_type' in df.columns:
            validation_results['document_type_counts'] = df['document_type'].value_counts().to_dict()
        
        # Validate each document
        for idx, row in df.iterrows():
            doc_type = row.get('document_type', 'unknown')
            
            if doc_type in [dt.value for dt in DocumentType]:
                try:
                    # Convert to DocumentType enum
                    doc_type_enum = DocumentType(doc_type)
                    
                    # Extract document data (exclude metadata columns)
                    metadata_columns = ['document_id', 'text', 'document_type', 'file_path', 'evaluation_timestamp', 'has_ground_truth']
                    doc_data = {k: v for k, v in row.items() if k not in metadata_columns}
                    
                    # Validate against schema
                    validation_result = validate_document_data(doc_type_enum, doc_data)
                    
                    if validation_result['valid']:
                        validation_results['valid_documents'] += 1
                    else:
                        validation_results['invalid_documents'] += 1
                        validation_results['validation_errors'].extend([
                            f"Document {idx}: {error}" for error in validation_result['errors']
                        ])
                
                except Exception as e:
                    validation_results['invalid_documents'] += 1
                    validation_results['validation_errors'].append(f"Document {idx}: {str(e)}")
            else:
                validation_results['invalid_documents'] += 1
                validation_results['validation_errors'].append(f"Document {idx}: Unknown document type '{doc_type}'")
        
        return validation_results
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the data
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.csv':
            df.to_csv(filepath, index=False)
        elif filepath.suffix == '.json':
            df.to_json(filepath, orient='records', indent=2)
        elif filepath.suffix == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """
        Load processed data from file.
        
        Args:
            filepath: Path to the processed data file
            
        Returns:
            DataFrame with processed data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            df = pd.read_json(filepath, orient='records')
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Processed data loaded from {filepath}")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_documents': len(df),
            'columns': list(df.columns),
            'document_types': {},
            'text_length_stats': {},
            'missing_values': {}
        }
        
        # Document type distribution
        if 'document_type' in df.columns:
            summary['document_types'] = df['document_type'].value_counts().to_dict()
        
        # Text length statistics
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            summary['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': float(text_lengths.std())
            }
        
        # Missing values
        summary['missing_values'] = df.isnull().sum().to_dict()
        
        return summary
