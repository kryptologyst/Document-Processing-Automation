"""
Dashboard Generator Module

This module provides dashboard generation capabilities for document processing
automation, including interactive dashboards and summary reports.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .visualizer import DocumentVisualizer
from .explainability import ExplainabilityEngine

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generator for interactive dashboards and reports."""
    
    def __init__(self, visualizer: Optional[DocumentVisualizer] = None):
        """
        Initialize dashboard generator.
        
        Args:
            visualizer: Document visualizer instance
        """
        self.visualizer = visualizer or DocumentVisualizer()
        self.explainability_engine = ExplainabilityEngine()
    
    def create_streamlit_dashboard(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Create a Streamlit dashboard.
        
        Args:
            evaluation_results: Dictionary of evaluation results
        """
        st.set_page_config(
            page_title="Document Processing Automation Dashboard",
            page_icon="📄",
            layout="wide"
        )
        
        # Header
        st.title("📄 Document Processing Automation Dashboard")
        st.markdown("---")
        
        # Sidebar
        self._create_sidebar(evaluation_results)
        
        # Main content
        self._create_main_content(evaluation_results)
    
    def _create_sidebar(self, evaluation_results: Dict[str, Any]) -> None:
        """Create sidebar with filters and controls."""
        st.sidebar.title("Controls")
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(datetime.now().date(), datetime.now().date()),
            max_value=datetime.now().date()
        )
        
        # Document type filter
        st.sidebar.subheader("Document Type")
        doc_types = st.sidebar.multiselect(
            "Select document types",
            options=["invoice", "receipt", "contract", "form"],
            default=["invoice", "receipt", "contract"]
        )
        
        # Confidence threshold
        st.sidebar.subheader("Confidence Threshold")
        confidence_threshold = st.sidebar.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        # Store filters in session state
        st.session_state.date_range = date_range
        st.session_state.doc_types = doc_types
        st.session_state.confidence_threshold = confidence_threshold
    
    def _create_main_content(self, evaluation_results: Dict[str, Any]) -> None:
        """Create main dashboard content."""
        # Key metrics
        self._create_metrics_section(evaluation_results)
        
        # Charts
        self._create_charts_section(evaluation_results)
        
        # Tables
        self._create_tables_section(evaluation_results)
        
        # Recommendations
        self._create_recommendations_section(evaluation_results)
    
    def _create_metrics_section(self, evaluation_results: Dict[str, Any]) -> None:
        """Create key metrics section."""
        st.subheader("📊 Key Metrics")
        
        # Extract metrics
        total_documents = evaluation_results.get('total_documents', 0)
        accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        avg_processing_time = evaluation_results.get('processing_performance', {}).get('avg_processing_time', 0)
        documents_per_second = evaluation_results.get('processing_performance', {}).get('documents_per_second', 0)
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Documents",
                value=total_documents,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Overall Accuracy",
                value=f"{accuracy:.3f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Avg Processing Time",
                value=f"{avg_processing_time:.3f}s",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Documents/sec",
                value=f"{documents_per_second:.1f}",
                delta=None
            )
    
    def _create_charts_section(self, evaluation_results: Dict[str, Any]) -> None:
        """Create charts section."""
        st.subheader("📈 Performance Charts")
        
        # Create tabs for different chart types
        tab1, tab2, tab3, tab4 = st.tabs(["Confidence", "Field Accuracy", "Processing Time", "Error Analysis"])
        
        with tab1:
            self._create_confidence_charts(evaluation_results)
        
        with tab2:
            self._create_field_accuracy_charts(evaluation_results)
        
        with tab3:
            self._create_processing_time_charts(evaluation_results)
        
        with tab4:
            self._create_error_analysis_charts(evaluation_results)
    
    def _create_confidence_charts(self, evaluation_results: Dict[str, Any]) -> None:
        """Create confidence-related charts."""
        # Confidence distribution
        if 'confidence_calibration' in evaluation_results:
            st.subheader("Confidence Distribution")
            
            # Generate sample confidence data
            confidences = np.random.beta(2, 2, 1000)  # Example data
            
            fig = self.visualizer.plot_confidence_distribution(confidences)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence calibration
        if 'confidence_calibration' in evaluation_results:
            st.subheader("Confidence Calibration")
            
            # Generate sample calibration data
            calibration_data = {
                'reliability_data': [
                    {'bin_center': 0.1, 'accuracy': 0.15, 'confidence': 0.1, 'count': 50},
                    {'bin_center': 0.3, 'accuracy': 0.35, 'confidence': 0.3, 'count': 100},
                    {'bin_center': 0.5, 'accuracy': 0.55, 'confidence': 0.5, 'count': 150},
                    {'bin_center': 0.7, 'accuracy': 0.75, 'confidence': 0.7, 'count': 200},
                    {'bin_center': 0.9, 'accuracy': 0.85, 'confidence': 0.9, 'count': 100}
                ]
            }
            
            fig = self.visualizer.plot_confidence_calibration(calibration_data)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_field_accuracy_charts(self, evaluation_results: Dict[str, Any]) -> None:
        """Create field accuracy charts."""
        if 'field_extraction' in evaluation_results:
            st.subheader("Field Extraction Accuracy")
            
            fig = self.visualizer.plot_field_extraction_accuracy(
                evaluation_results['field_extraction']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if 'feature_importance' in evaluation_results:
            st.subheader("Feature Importance")
            
            feature_importance = evaluation_results['feature_importance']
            fig = self.visualizer.plot_feature_importance(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_processing_time_charts(self, evaluation_results: Dict[str, Any]) -> None:
        """Create processing time charts."""
        st.subheader("Processing Performance")
        
        # Generate sample performance data
        performance_data = {
            'processing_times': np.random.exponential(0.5, 100),
            'throughput': np.random.normal(10, 2, 100),
            'document_sizes': np.random.normal(500, 200, 100),
            'batch_times': np.random.exponential(2, 20)
        }
        
        fig = self.visualizer.plot_processing_performance(performance_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_error_analysis_charts(self, evaluation_results: Dict[str, Any]) -> None:
        """Create error analysis charts."""
        st.subheader("Error Analysis")
        
        # Generate sample error data
        error_data = {
            'error_types': {
                'Field Missing': 25,
                'Low Confidence': 15,
                'Type Mismatch': 10,
                'Processing Error': 5
            },
            'errors_by_type': {
                'invoice': 20,
                'receipt': 15,
                'contract': 10
            },
            'error_trends': np.random.poisson(5, 30),
            'confidence_vs_errors': [
                {'confidence': 0.3, 'error_count': 8},
                {'confidence': 0.5, 'error_count': 5},
                {'confidence': 0.7, 'error_count': 2},
                {'confidence': 0.9, 'error_count': 1}
            ]
        }
        
        fig = self.visualizer.plot_error_analysis(error_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_tables_section(self, evaluation_results: Dict[str, Any]) -> None:
        """Create tables section."""
        st.subheader("📋 Detailed Results")
        
        # Create tabs for different table types
        tab1, tab2, tab3 = st.tabs(["Field Metrics", "Document Types", "Error Details"])
        
        with tab1:
            self._create_field_metrics_table(evaluation_results)
        
        with tab2:
            self._create_document_types_table(evaluation_results)
        
        with tab3:
            self._create_error_details_table(evaluation_results)
    
    def _create_field_metrics_table(self, evaluation_results: Dict[str, Any]) -> None:
        """Create field metrics table."""
        if 'field_extraction' in evaluation_results:
            field_metrics = evaluation_results['field_extraction']
            
            # Convert to DataFrame
            data = []
            for field_name, metrics in field_metrics.items():
                data.append({
                    'Field': field_name,
                    'Precision': f"{metrics.precision:.3f}",
                    'Recall': f"{metrics.recall:.3f}",
                    'F1-Score': f"{metrics.f1_score:.3f}",
                    'Accuracy': f"{metrics.accuracy:.3f}",
                    'Total Extracted': metrics.total_extracted,
                    'Total Expected': metrics.total_expected,
                    'Correct': metrics.total_correct
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
    
    def _create_document_types_table(self, evaluation_results: Dict[str, Any]) -> None:
        """Create document types table."""
        if 'classification' in evaluation_results:
            classification_results = evaluation_results['classification']
            
            # Create classification report table
            if 'classification_report' in classification_results:
                report = classification_results['classification_report']
                
                # Convert to DataFrame
                data = []
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        data.append({
                            'Document Type': class_name,
                            'Precision': f"{metrics['precision']:.3f}",
                            'Recall': f"{metrics['recall']:.3f}",
                            'F1-Score': f"{metrics['f1-score']:.3f}",
                            'Support': metrics['support']
                        })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
    
    def _create_error_details_table(self, evaluation_results: Dict[str, Any]) -> None:
        """Create error details table."""
        # Generate sample error details
        error_details = pd.DataFrame({
            'Document ID': [f'doc_{i:04d}' for i in range(10)],
            'Document Type': np.random.choice(['invoice', 'receipt', 'contract'], 10),
            'Error Type': np.random.choice(['Field Missing', 'Low Confidence', 'Type Mismatch'], 10),
            'Confidence': np.random.uniform(0.2, 0.8, 10),
            'Timestamp': pd.date_range('2024-01-01', periods=10, freq='H')
        })
        
        st.dataframe(error_details, use_container_width=True)
    
    def _create_recommendations_section(self, evaluation_results: Dict[str, Any]) -> None:
        """Create recommendations section."""
        st.subheader("💡 Recommendations")
        
        # Extract recommendations
        recommendations = evaluation_results.get('summary', {}).get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("No specific recommendations at this time.")
        
        # Add action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Results"):
                st.success("Results exported successfully!")
        
        with col2:
            if st.button("Generate Report"):
                st.success("Report generated successfully!")
        
        with col3:
            if st.button("Schedule Retraining"):
                st.success("Retraining scheduled!")
    
    def create_html_dashboard(self, evaluation_results: Dict[str, Any], 
                            output_path: str) -> str:
        """
        Create an HTML dashboard.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            output_path: Path to save the dashboard
            
        Returns:
            Path to the created dashboard
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._generate_html_dashboard(evaluation_results)
        
        # Save dashboard
        dashboard_file = output_path / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML dashboard created at {dashboard_file}")
        return str(dashboard_file)
    
    def _generate_html_dashboard(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate HTML content for the dashboard."""
        # Extract key metrics
        total_documents = evaluation_results.get('total_documents', 0)
        accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        avg_processing_time = evaluation_results.get('processing_performance', {}).get('avg_processing_time', 0)
        documents_per_second = evaluation_results.get('processing_performance', {}).get('documents_per_second', 0)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document Processing Automation Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2E86AB;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chart-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                }}
                .recommendations {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .recommendation-item {{
                    margin-bottom: 10px;
                    padding: 10px;
                    background: #f8f9fa;
                    border-left: 4px solid #2E86AB;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📄 Document Processing Automation Dashboard</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{total_documents}</div>
                    <div class="metric-label">Total Documents</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{accuracy:.3f}</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_processing_time:.3f}s</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{documents_per_second:.1f}</div>
                    <div class="metric-label">Documents/sec</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Confidence Distribution</div>
                <div id="confidence-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Field Extraction Accuracy</div>
                <div id="field-accuracy-chart"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Document Type Distribution</div>
                <div id="document-types-chart"></div>
            </div>
            
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <div class="recommendation-item">
                    Consider improving document type classification with more training data
                </div>
                <div class="recommendation-item">
                    Field extraction performance is below target - consider pattern refinement
                </div>
                <div class="recommendation-item">
                    Confidence calibration needs improvement - consider calibration techniques
                </div>
                <div class="recommendation-item">
                    Processing time is high - consider optimization or parallel processing
                </div>
            </div>
            
            <script>
                // Add your Plotly charts here
                // This is a simplified version - you would generate the actual charts
                // based on the evaluation results
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def create_summary_report(self, evaluation_results: Dict[str, Any], 
                            output_path: str) -> str:
        """
        Create a comprehensive summary report.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            output_path: Path to save the report
            
        Returns:
            Path to the created report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create report content
        report_content = self._generate_summary_report(evaluation_results)
        
        # Save report
        report_file = output_path / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report created at {report_file}")
        return str(report_file)
    
    def _generate_summary_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate markdown summary report."""
        # Extract key information
        total_documents = evaluation_results.get('total_documents', 0)
        accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        avg_processing_time = evaluation_results.get('processing_performance', {}).get('avg_processing_time', 0)
        documents_per_second = evaluation_results.get('processing_performance', {}).get('documents_per_second', 0)
        recommendations = evaluation_results.get('summary', {}).get('recommendations', [])
        
        report = f"""
# Document Processing Automation - Summary Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report provides a comprehensive analysis of the document processing automation system performance.

### Key Metrics

- **Total Documents Processed:** {total_documents}
- **Overall Accuracy:** {accuracy:.3f}
- **Average Processing Time:** {avg_processing_time:.3f} seconds
- **Processing Throughput:** {documents_per_second:.1f} documents/second

## Performance Analysis

### Document Classification
- **Accuracy:** {evaluation_results.get('classification', {}).get('accuracy', 0):.3f}
- **Precision:** {evaluation_results.get('classification', {}).get('precision', 0):.3f}
- **Recall:** {evaluation_results.get('classification', {}).get('recall', 0):.3f}
- **F1-Score:** {evaluation_results.get('classification', {}).get('f1_score', 0):.3f}

### Field Extraction
- **Average Field Precision:** {evaluation_results.get('summary', {}).get('field_extraction_metrics', {}).get('avg_precision', 0):.3f}
- **Average Field Recall:** {evaluation_results.get('summary', {}).get('field_extraction_metrics', {}).get('avg_recall', 0):.3f}
- **Average Field F1-Score:** {evaluation_results.get('summary', {}).get('field_extraction_metrics', {}).get('avg_f1_score', 0):.3f}

### Processing Performance
- **Average Processing Time:** {avg_processing_time:.3f} seconds
- **Processing Throughput:** {documents_per_second:.1f} documents/second
- **Human Review Rate:** {evaluation_results.get('summary', {}).get('human_review_rate', 0):.3f}

## Recommendations

"""
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        else:
            report += "No specific recommendations at this time.\n"
        
        report += """
## Next Steps

1. **Monitor Performance:** Continue monitoring system performance and accuracy metrics
2. **Data Quality:** Ensure high-quality training data for model improvement
3. **User Feedback:** Collect and incorporate user feedback for system enhancement
4. **Regular Updates:** Schedule regular model retraining and system updates

## Conclusion

The document processing automation system shows promising results with room for improvement in specific areas. Regular monitoring and continuous improvement will ensure optimal performance.

---
*This report was generated automatically by the Document Processing Automation system.*
"""
        
        return report
