"""
Document Classification Module

This module provides machine learning-based document classification
for automatically identifying document types.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Machine learning-based document classifier."""
    
    def __init__(self, model_type: str = "random_forest", max_features: int = 1000):
        """
        Initialize document classifier.
        
        Args:
            model_type: Type of classifier ("random_forest" or "logistic_regression")
            max_features: Maximum number of TF-IDF features
        """
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
        self.classes_ = None
    
    def prepare_features(self, texts: List[str]) -> np.ndarray:
        """
        Prepare TF-IDF features from text.
        
        Args:
            texts: List of document texts
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.is_trained:
            # Fit vectorizer on training data
            features = self.vectorizer.fit_transform(texts)
        else:
            # Transform using fitted vectorizer
            features = self.vectorizer.transform(texts)
        
        return features.toarray()
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train the document classifier.
        
        Args:
            texts: List of document texts
            labels: List of document type labels
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_type} classifier on {len(texts)} documents")
        
        # Prepare features
        X = self.prepare_features(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Training completed. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        return metrics
    
    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Predict document types.
        
        Args:
            texts: List of document texts
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(texts)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get confidence scores (probability estimates)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
        else:
            # For models without predict_proba, use decision function
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X)
                confidence_scores = np.abs(decision_scores) / (np.abs(decision_scores).max() + 1e-8)
            else:
                confidence_scores = [0.5] * len(predictions)
        
        return predictions.tolist(), confidence_scores.tolist()
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Predict document type for a single document.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (prediction, confidence)
        """
        predictions, confidences = self.predict([text])
        return predictions[0], confidences[0]
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance for interpretability.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return {}
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get importance scores
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        importance_dict = {}
        for i in indices:
            importance_dict[feature_names[i]] = float(importances[i])
        
        return importance_dict
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and vectorizer.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'max_features': self.max_features,
            'classes': self.classes_
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and vectorizer.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.max_features = model_data['max_features']
        self.classes_ = model_data['classes']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            texts: List of test document texts
            labels: List of true document type labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions, confidences = self.predict(texts)
        
        # Calculate metrics
        accuracy = np.mean([p == l for p, l in zip(predictions, labels)])
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'confidences': confidences
        }
        
        return metrics
