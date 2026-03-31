"""
Layout Parsing Module

This module provides layout analysis and parsing functionality
for understanding document structure and organizing extracted text.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a block of text with position information."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    block_type: str = "text"  # text, header, footer, table, etc.


@dataclass
class DocumentLayout:
    """Represents the layout structure of a document."""
    blocks: List[TextBlock]
    page_width: int
    page_height: int
    reading_order: List[int]  # Indices of blocks in reading order


class LayoutParser:
    """Layout parser for analyzing document structure and organizing text blocks."""
    
    def __init__(self, min_block_area: int = 100):
        """
        Initialize layout parser.
        
        Args:
            min_block_area: Minimum area for a text block to be considered valid
        """
        self.min_block_area = min_block_area
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in an image using contour detection.
        
        Args:
            image: Input image (should be preprocessed)
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by area and reasonable aspect ratio
                if (area >= self.min_block_area and 
                    0.1 <= aspect_ratio <= 10.0 and
                    w >= 10 and h >= 10):
                    text_regions.append((x, y, w, h))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []
    
    def merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]], 
                                overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping text regions.
        
        Args:
            regions: List of bounding boxes
            overlap_threshold: Minimum overlap ratio to merge regions
            
        Returns:
            List of merged bounding boxes
        """
        if not regions:
            return []
        
        # Sort regions by area (largest first)
        regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
        
        merged = []
        used = set()
        
        for i, (x1, y1, w1, h1) in enumerate(regions):
            if i in used:
                continue
            
            # Find overlapping regions
            overlapping = [i]
            for j, (x2, y2, w2, h2) in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate overlap
                overlap = self._calculate_overlap((x1, y1, w1, h1), (x2, y2, w2, h2))
                if overlap > overlap_threshold:
                    overlapping.append(j)
                    used.add(j)
            
            # Merge overlapping regions
            if len(overlapping) > 1:
                merged_region = self._merge_regions([regions[k] for k in overlapping])
                merged.append(merged_region)
            else:
                merged.append((x1, y1, w1, h1))
            
            used.add(i)
        
        return merged
    
    def _calculate_overlap(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_regions(self, regions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge multiple regions into one."""
        if not regions:
            return (0, 0, 0, 0)
        
        x_min = min(r[0] for r in regions)
        y_min = min(r[1] for r in regions)
        x_max = max(r[0] + r[2] for r in regions)
        y_max = max(r[1] + r[3] for r in regions)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def determine_reading_order(self, regions: List[Tuple[int, int, int, int]]) -> List[int]:
        """
        Determine reading order of text regions (top-to-bottom, left-to-right).
        
        Args:
            regions: List of bounding boxes
            
        Returns:
            List of indices in reading order
        """
        if not regions:
            return []
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        sorted_indices = sorted(range(len(regions)), 
                              key=lambda i: (regions[i][1], regions[i][0]))
        
        return sorted_indices
    
    def classify_block_types(self, regions: List[Tuple[int, int, int, int]], 
                           image_shape: Tuple[int, int]) -> List[str]:
        """
        Classify text blocks by their position and characteristics.
        
        Args:
            regions: List of bounding boxes
            image_shape: (height, width) of the image
            
        Returns:
            List of block types
        """
        if not regions:
            return []
        
        height, width = image_shape
        block_types = []
        
        for x, y, w, h in regions:
            # Determine position-based classification
            y_ratio = y / height
            x_ratio = x / width
            
            if y_ratio < 0.1:  # Top 10% of page
                block_type = "header"
            elif y_ratio > 0.9:  # Bottom 10% of page
                block_type = "footer"
            elif w > width * 0.8:  # Wide blocks might be titles
                block_type = "title"
            elif h > height * 0.1:  # Tall blocks might be tables
                block_type = "table"
            else:
                block_type = "text"
            
            block_types.append(block_type)
        
        return block_types
    
    def parse_layout(self, image: np.ndarray, text_blocks: List[Tuple[str, float]]) -> DocumentLayout:
        """
        Parse document layout and organize text blocks.
        
        Args:
            image: Input image
            text_blocks: List of (text, confidence) tuples
            
        Returns:
            DocumentLayout object
        """
        try:
            # Detect text regions
            regions = self.detect_text_regions(image)
            
            # Merge overlapping regions
            merged_regions = self.merge_overlapping_regions(regions)
            
            # Determine reading order
            reading_order = self.determine_reading_order(merged_regions)
            
            # Classify block types
            block_types = self.classify_block_types(merged_regions, image.shape[:2])
            
            # Create text blocks
            blocks = []
            for i, (region, block_type) in enumerate(zip(merged_regions, block_types)):
                if i < len(text_blocks):
                    text, confidence = text_blocks[i]
                    block = TextBlock(
                        text=text,
                        bbox=region,
                        confidence=confidence,
                        block_type=block_type
                    )
                    blocks.append(block)
            
            return DocumentLayout(
                blocks=blocks,
                page_width=image.shape[1],
                page_height=image.shape[0],
                reading_order=reading_order
            )
            
        except Exception as e:
            logger.error(f"Layout parsing failed: {e}")
            return DocumentLayout(
                blocks=[],
                page_width=image.shape[1] if len(image.shape) > 1 else 0,
                page_height=image.shape[0] if len(image.shape) > 0 else 0,
                reading_order=[]
            )
