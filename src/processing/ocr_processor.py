"""
OCR Processing Module

This module provides OCR (Optical Character Recognition) functionality
for extracting text from images and scanned documents.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor for extracting text from images and documents."""
    
    def __init__(self, use_tesseract: bool = True):
        """
        Initialize OCR processor.
        
        Args:
            use_tesseract: Whether to use Tesseract OCR (requires pytesseract)
        """
        self.use_tesseract = use_tesseract
        
        if use_tesseract:
            try:
                import pytesseract
                self.tesseract = pytesseract
                logger.info("Tesseract OCR initialized successfully")
            except ImportError:
                logger.warning("pytesseract not available, falling back to basic OCR")
                self.use_tesseract = False
                self.tesseract = None
        else:
            self.tesseract = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_tesseract(self, image: np.ndarray, config: str = '--psm 6') -> Tuple[str, float]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: Preprocessed image
            config: Tesseract configuration
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        if not self.use_tesseract or self.tesseract is None:
            return "", 0.0
        
        try:
            # Extract text with confidence
            data = self.tesseract.image_to_data(image, config=config, output_type=self.tesseract.Output.DICT)
            
            # Combine text
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Only include text with confidence > 0
                    text_parts.append(data['text'][i])
                    confidences.append(int(data['conf'][i]))
            
            extracted_text = ' '.join(text_parts).strip()
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return extracted_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return "", 0.0
    
    def extract_text_basic(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Basic text extraction (placeholder for when Tesseract is not available).
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        # This is a placeholder implementation
        # In a real scenario, you might use other OCR libraries or cloud services
        logger.warning("Basic OCR not implemented, returning empty text")
        return "", 0.0
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return "", 0.0
            
            # Preprocess if requested
            if preprocess:
                image = self.preprocess_image(image)
            
            # Extract text
            if self.use_tesseract:
                return self.extract_text_tesseract(image)
            else:
                return self.extract_text_basic(image)
                
        except Exception as e:
            logger.error(f"Text extraction failed for {image_path}: {e}")
            return "", 0.0
    
    def extract_text_from_array(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from a numpy array image.
        
        Args:
            image: Image as numpy array
            preprocess: Whether to preprocess the image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Preprocess if requested
            if preprocess:
                image = self.preprocess_image(image)
            
            # Extract text
            if self.use_tesseract:
                return self.extract_text_tesseract(image)
            else:
                return self.extract_text_basic(image)
                
        except Exception as e:
            logger.error(f"Text extraction from array failed: {e}")
            return "", 0.0
    
    def batch_extract_text(self, image_paths: List[str], preprocess: bool = True) -> List[Tuple[str, float]]:
        """
        Extract text from multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            preprocess: Whether to preprocess images
            
        Returns:
            List of (extracted_text, confidence) tuples
        """
        results = []
        for image_path in image_paths:
            text, confidence = self.extract_text(image_path, preprocess)
            results.append((text, confidence))
        
        return results
