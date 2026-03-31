"""
Document Visualizer Module

This module provides visualization capabilities for document processing
automation, including result visualization, performance charts, and
interactive dashboards.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentVisualizer:
    """Visualizer for document processing automation results."""
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize document visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
        # Set color palette
        self.colors = px.colors.qualitative.Set3
    
    def plot_confidence_distribution(self, confidences: List[float], 
                                   title: str = "Confidence Score Distribution") -> go.Figure:
        """
        Plot confidence score distribution.
        
        Args:
            confidences: List of confidence scores
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name="Confidence Distribution",
            marker_color=self.colors[0],
            opacity=0.7
        ))
        
        # Add mean line
        mean_confidence = np.mean(confidences)
        fig.add_vline(
            x=mean_confidence,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_confidence:.3f}"
        )
        
        # Add threshold lines
        fig.add_vline(x=0.7, line_dash="dot", line_color="orange", annotation_text="Auto Threshold")
        fig.add_vline(x=0.5, line_dash="dot", line_color="red", annotation_text="Review Threshold")
        
        fig.update_layout(
            title=title,
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def plot_field_extraction_accuracy(self, field_metrics: Dict[str, Any], 
                                     title: str = "Field Extraction Accuracy") -> go.Figure:
        """
        Plot field extraction accuracy metrics.
        
        Args:
            field_metrics: Dictionary of field metrics
            title: Plot title
            
        Returns:
            Plotly figure
        """
        field_names = list(field_metrics.keys())
        precisions = [metrics.precision for metrics in field_metrics.values()]
        recalls = [metrics.recall for metrics in field_metrics.values()]
        f1_scores = [metrics.f1_score for metrics in field_metrics.values()]
        
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            name="Precision",
            x=field_names,
            y=precisions,
            marker_color=self.colors[0]
        ))
        
        fig.add_trace(go.Bar(
            name="Recall",
            x=field_names,
            y=recalls,
            marker_color=self.colors[1]
        ))
        
        fig.add_trace(go.Bar(
            name="F1-Score",
            x=field_names,
            y=f1_scores,
            marker_color=self.colors[2]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Field Name",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white"
        )
        
        return fig
    
    def plot_processing_performance(self, performance_data: Dict[str, Any], 
                                  title: str = "Processing Performance") -> go.Figure:
        """
        Plot processing performance metrics.
        
        Args:
            performance_data: Dictionary of performance data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Processing Time", "Throughput", "Document Size vs Time", "Batch Performance"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Processing time over time
        if 'processing_times' in performance_data:
            times = performance_data['processing_times']
            fig.add_trace(
                go.Scatter(x=list(range(len(times))), y=times, mode='lines+markers', name="Processing Time"),
                row=1, col=1
            )
        
        # Throughput over time
        if 'throughput' in performance_data:
            throughput = performance_data['throughput']
            fig.add_trace(
                go.Scatter(x=list(range(len(throughput))), y=throughput, mode='lines+markers', name="Throughput"),
                row=1, col=2
            )
        
        # Document size vs processing time
        if 'document_sizes' in performance_data and 'processing_times' in performance_data:
            sizes = performance_data['document_sizes']
            times = performance_data['processing_times']
            fig.add_trace(
                go.Scatter(x=sizes, y=times, mode='markers', name="Size vs Time"),
                row=2, col=1
            )
        
        # Batch performance
        if 'batch_times' in performance_data:
            batch_times = performance_data['batch_times']
            fig.add_trace(
                go.Bar(x=list(range(len(batch_times))), y=batch_times, name="Batch Time"),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_error_analysis(self, error_data: Dict[str, Any], 
                           title: str = "Error Analysis") -> go.Figure:
        """
        Plot error analysis charts.
        
        Args:
            error_data: Dictionary of error data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Error Types", "Error by Document Type", "Error Trends", "Confidence vs Errors"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Error types pie chart
        if 'error_types' in error_data:
            error_types = error_data['error_types']
            fig.add_trace(
                go.Pie(labels=list(error_types.keys()), values=list(error_types.values()), name="Error Types"),
                row=1, col=1
            )
        
        # Error by document type
        if 'errors_by_type' in error_data:
            doc_types = list(error_data['errors_by_type'].keys())
            error_counts = list(error_data['errors_by_type'].values())
            fig.add_trace(
                go.Bar(x=doc_types, y=error_counts, name="Errors by Type"),
                row=1, col=2
            )
        
        # Error trends over time
        if 'error_trends' in error_data:
            trends = error_data['error_trends']
            fig.add_trace(
                go.Scatter(x=list(range(len(trends))), y=trends, mode='lines+markers', name="Error Trend"),
                row=2, col=1
            )
        
        # Confidence vs errors scatter
        if 'confidence_vs_errors' in error_data:
            conf_errors = error_data['confidence_vs_errors']
            confidences = [item['confidence'] for item in conf_errors]
            errors = [item['error_count'] for item in conf_errors]
            fig.add_trace(
                go.Scatter(x=confidences, y=errors, mode='markers', name="Confidence vs Errors"),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_confidence_calibration(self, calibration_data: Dict[str, Any], 
                                  title: str = "Confidence Calibration") -> go.Figure:
        """
        Plot confidence calibration chart.
        
        Args:
            calibration_data: Dictionary of calibration data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Reliability diagram
        if 'reliability_data' in calibration_data:
            reliability = calibration_data['reliability_data']
            bin_centers = [item['bin_center'] for item in reliability]
            accuracies = [item['accuracy'] for item in reliability]
            confidences = [item['confidence'] for item in reliability]
            
            # Perfect calibration line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='red')
            ))
            
            # Actual calibration
            fig.add_trace(go.Scatter(
                x=confidences, y=accuracies,
                mode='markers+lines',
                name='Actual Calibration',
                marker=dict(size=8, color=self.colors[0])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Confidence",
            yaxis_title="Accuracy",
            template="plotly_white"
        )
        
        return fig
    
    def plot_document_type_distribution(self, document_types: List[str], 
                                      title: str = "Document Type Distribution") -> go.Figure:
        """
        Plot document type distribution.
        
        Args:
            document_types: List of document types
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Count document types
        type_counts = pd.Series(document_types).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.3
        )])
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              title: str = "Feature Importance") -> go.Figure:
        """
        Plot feature importance chart.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            title: Plot title
            
        Returns:
            Plotly figure
        """
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        fig = go.Figure(data=[go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=self.colors[0]
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_white"
        )
        
        return fig
    
    def plot_processing_timeline(self, timeline_data: List[Dict[str, Any]], 
                               title: str = "Processing Timeline") -> go.Figure:
        """
        Plot processing timeline.
        
        Args:
            timeline_data: List of timeline data points
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Extract data
        timestamps = [item['timestamp'] for item in timeline_data]
        processing_times = [item['processing_time'] for item in timeline_data]
        confidences = [item['confidence'] for item in timeline_data]
        
        # Create scatter plot with color-coded confidence
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=processing_times,
            mode='markers',
            marker=dict(
                size=8,
                color=confidences,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f"Confidence: {c:.3f}" for c in confidences],
            hovertemplate="<b>%{text}</b><br>Time: %{y:.3f}s<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Timestamp",
            yaxis_title="Processing Time (seconds)",
            template="plotly_white"
        )
        
        return fig
    
    def create_dashboard(self, evaluation_results: Dict[str, Any], 
                        output_path: str) -> str:
        """
        Create a comprehensive dashboard.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            output_path: Path to save the dashboard
            
        Returns:
            Path to the created dashboard
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create HTML dashboard
        html_content = self._generate_dashboard_html(evaluation_results)
        
        # Save dashboard
        dashboard_file = output_path / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard created at {dashboard_file}")
        return str(dashboard_file)
    
    def _generate_dashboard_html(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate HTML content for the dashboard."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Processing Automation Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .chart-container { margin: 20px 0; }
                .metric-card { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    text-align: center;
                    min-width: 150px;
                }
                .metric-value { font-size: 24px; font-weight: bold; color: #2E86AB; }
                .metric-label { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Processing Automation Dashboard</h1>
                <p>Generated on: {timestamp}</p>
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
                <div id="confidence-distribution"></div>
            </div>
            
            <div class="chart-container">
                <div id="field-accuracy"></div>
            </div>
            
            <div class="chart-container">
                <div id="document-types"></div>
            </div>
            
            <script>
                // Add your Plotly charts here
                // This is a simplified version - you would generate the actual charts
                // based on the evaluation results
            </script>
        </body>
        </html>
        """
        
        # Extract metrics
        total_documents = evaluation_results.get('total_documents', 0)
        accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        avg_processing_time = evaluation_results.get('processing_performance', {}).get('avg_processing_time', 0)
        documents_per_second = evaluation_results.get('processing_performance', {}).get('documents_per_second', 0)
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_documents=total_documents,
            accuracy=accuracy,
            avg_processing_time=avg_processing_time,
            documents_per_second=documents_per_second
        )
    
    def save_plots(self, plots: Dict[str, go.Figure], output_dir: str) -> None:
        """
        Save plots to files.
        
        Args:
            plots: Dictionary of plot names and figures
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_name, fig in plots.items():
            # Save as HTML
            html_file = output_dir / f"{plot_name}.html"
            fig.write_html(str(html_file))
            
            # Save as PNG
            png_file = output_dir / f"{plot_name}.png"
            fig.write_image(str(png_file), width=1200, height=800)
        
        logger.info(f"Plots saved to {output_dir}")
    
    def create_summary_report(self, evaluation_results: Dict[str, Any], 
                            output_path: str) -> str:
        """
        Create a summary report with visualizations.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            output_path: Path to save the report
            
        Returns:
            Path to the created report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        plots = {}
        
        # Confidence distribution
        if 'confidence_calibration' in evaluation_results:
            confidences = [0.7, 0.8, 0.9, 0.6, 0.5]  # Example data
            plots['confidence_distribution'] = self.plot_confidence_distribution(confidences)
        
        # Field extraction accuracy
        if 'field_extraction' in evaluation_results:
            plots['field_accuracy'] = self.plot_field_extraction_accuracy(
                evaluation_results['field_extraction']
            )
        
        # Document type distribution
        if 'classification' in evaluation_results:
            doc_types = ['invoice', 'receipt', 'contract']  # Example data
            plots['document_types'] = self.plot_document_type_distribution(doc_types)
        
        # Save plots
        self.save_plots(plots, output_path / "plots")
        
        # Create dashboard
        dashboard_path = self.create_dashboard(evaluation_results, output_path)
        
        return dashboard_path
