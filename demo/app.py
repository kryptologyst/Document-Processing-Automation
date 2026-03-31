"""
Streamlit Demo Application

This module provides an interactive Streamlit demo for the document processing
automation system, showcasing key features and capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json
from datetime import datetime
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.processing.document_processor import DocumentProcessor, DocumentField, DocumentResult
from src.data.data_generator import DocumentDataGenerator
from src.data.schema import DocumentType
from src.eval.evaluator import DocumentProcessingEvaluator
from src.viz.explainability import ExplainabilityEngine
from src.viz.dashboard import DashboardGenerator

# Page configuration
st.set_page_config(
    page_title="Document Processing Automation Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor(confidence_threshold=0.7)
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DocumentDataGenerator(seed=42)
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = DocumentProcessingEvaluator()
if 'explainability_engine' not in st.session_state:
    st.session_state.explainability_engine = ExplainabilityEngine()
if 'dashboard_generator' not in st.session_state:
    st.session_state.dashboard_generator = DashboardGenerator()

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📄 Document Processing Automation Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This is a research and educational demonstration system.</strong> 
        It is not intended for automated decision-making without human review. 
        All results should be validated by qualified professionals before use in production environments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page",
        ["Document Processing", "Batch Processing", "Evaluation", "Explainability", "Dashboard", "About"]
    )
    
    # Route to appropriate page
    if page == "Document Processing":
        document_processing_page()
    elif page == "Batch Processing":
        batch_processing_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Explainability":
        explainability_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "About":
        about_page()

def document_processing_page():
    """Document processing page."""
    st.header("📄 Document Processing")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "Sample Documents", "File Upload"]
    )
    
    if input_method == "Text Input":
        # Text input
        st.subheader("Enter Document Text")
        document_text = st.text_area(
            "Document Text",
            height=200,
            placeholder="Enter your document text here..."
        )
        
        if st.button("Process Document"):
            if document_text.strip():
                process_single_document(document_text)
            else:
                st.error("Please enter document text.")
    
    elif input_method == "Sample Documents":
        # Sample documents
        st.subheader("Sample Documents")
        
        # Document type selection
        doc_type = st.selectbox(
            "Select document type:",
            [dt.value for dt in DocumentType]
        )
        
        if st.button("Generate Sample Document"):
            # Generate sample document
            doc_type_enum = DocumentType(doc_type)
            sample_text = st.session_state.data_generator.generate_document_text(doc_type_enum)
            
            st.text_area("Generated Sample Document", value=sample_text, height=200)
            
            if st.button("Process Sample Document"):
                process_single_document(sample_text)
    
    elif input_method == "File Upload":
        # File upload
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a document file for processing"
        )
        
        if uploaded_file is not None:
            # Read file content
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
                st.text_area("File Content", value=content, height=200)
                
                if st.button("Process Uploaded Document"):
                    process_single_document(content)
            else:
                st.warning("File type not supported. Please upload a .txt file.")

def process_single_document(text: str):
    """Process a single document and display results."""
    with st.spinner("Processing document..."):
        # Process document
        result = st.session_state.processor.process_document(text)
        
        # Display results
        st.subheader("Processing Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document Information**")
            st.write(f"**Document ID:** {result.document_id}")
            st.write(f"**Document Type:** {result.document_type}")
            st.write(f"**Overall Confidence:** {result.overall_confidence:.3f}")
            st.write(f"**Processing Time:** {result.processing_time:.3f}s")
        
        with col2:
            st.markdown("**Extracted Fields**")
            if result.fields:
                for field in result.fields:
                    st.write(f"**{field.name}:** {field.value} (confidence: {field.confidence:.3f})")
            else:
                st.write("No fields extracted.")
        
        # Confidence visualization
        st.subheader("Confidence Analysis")
        confidence_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.overall_confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Confidence"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        confidence_fig.update_layout(height=300)
        st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Field confidence chart
        if result.fields:
            st.subheader("Field Confidence Scores")
            field_names = [field.name for field in result.fields]
            field_confidences = [field.confidence for field in result.fields]
            
            field_fig = go.Figure(go.Bar(
                x=field_names,
                y=field_confidences,
                marker_color=['green' if c >= 0.7 else 'orange' if c >= 0.5 else 'red' for c in field_confidences]
            ))
            field_fig.update_layout(
                title="Field Extraction Confidence",
                xaxis_title="Field Name",
                yaxis_title="Confidence Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(field_fig, use_container_width=True)

def batch_processing_page():
    """Batch processing page."""
    st.header("📦 Batch Processing")
    
    # Batch size selection
    batch_size = st.slider("Number of documents to generate", 5, 100, 20)
    
    # Document type selection
    doc_types = st.multiselect(
        "Select document types:",
        [dt.value for dt in DocumentType],
        default=["invoice", "receipt"]
    )
    
    if st.button("Generate and Process Batch"):
        if doc_types:
            process_batch(doc_types, batch_size)
        else:
            st.error("Please select at least one document type.")

def process_batch(doc_types: list, batch_size: int):
    """Process a batch of documents."""
    with st.spinner("Generating and processing batch..."):
        # Generate documents
        all_documents = []
        for doc_type in doc_types:
            doc_type_enum = DocumentType(doc_type)
            count = batch_size // len(doc_types)
            documents = st.session_state.data_generator.generate_text_batch(doc_type_enum, count)
            all_documents.extend(documents)
        
        # Process documents
        results = st.session_state.processor.process_batch(all_documents)
        
        # Display results
        st.subheader("Batch Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", len(results))
        
        with col2:
            avg_confidence = np.mean([r.overall_confidence for r in results])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            avg_time = np.mean([r.processing_time for r in results])
            st.metric("Avg Processing Time", f"{avg_time:.3f}s")
        
        with col4:
            total_time = sum([r.processing_time for r in results])
            st.metric("Total Processing Time", f"{total_time:.3f}s")
        
        # Results table
        st.subheader("Detailed Results")
        results_data = []
        for result in results:
            results_data.append({
                'Document ID': result.document_id,
                'Document Type': result.document_type,
                'Confidence': f"{result.overall_confidence:.3f}",
                'Processing Time': f"{result.processing_time:.3f}s",
                'Fields Extracted': len(result.fields)
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        st.subheader("Batch Analysis")
        
        # Document type distribution
        doc_type_counts = pd.Series([r.document_type for r in results]).value_counts()
        type_fig = px.pie(values=doc_type_counts.values, names=doc_type_counts.index, title="Document Type Distribution")
        st.plotly_chart(type_fig, use_container_width=True)
        
        # Confidence distribution
        confidences = [r.overall_confidence for r in results]
        conf_fig = px.histogram(x=confidences, nbins=20, title="Confidence Score Distribution")
        st.plotly_chart(conf_fig, use_container_width=True)

def evaluation_page():
    """Evaluation page."""
    st.header("📊 Evaluation")
    
    # Evaluation type selection
    eval_type = st.radio(
        "Select evaluation type:",
        ["Quick Evaluation", "Comprehensive Evaluation", "Custom Evaluation"]
    )
    
    if eval_type == "Quick Evaluation":
        quick_evaluation()
    elif eval_type == "Comprehensive Evaluation":
        comprehensive_evaluation()
    elif eval_type == "Custom Evaluation":
        custom_evaluation()

def quick_evaluation():
    """Quick evaluation."""
    st.subheader("Quick Evaluation")
    
    if st.button("Run Quick Evaluation"):
        with st.spinner("Running evaluation..."):
            # Generate test data
            test_documents = []
            test_ground_truth = []
            
            for doc_type in DocumentType:
                # Generate documents
                texts = st.session_state.data_generator.generate_text_batch(doc_type, 10)
                data = st.session_state.data_generator.generate_batch(doc_type, 10)
                
                # Process documents
                results = st.session_state.processor.process_batch(texts)
                
                # Prepare predictions and ground truth
                for result, gt_data in zip(results, data):
                    predictions = {
                        'document_type': result.document_type,
                        'fields': [{'name': f.name, 'value': f.value} for f in result.fields],
                        'confidence': result.overall_confidence,
                        'processing_time': result.processing_time
                    }
                    test_documents.append(predictions)
                    
                    ground_truth = {
                        'document_type': doc_type.value,
                        'fields': [{'name': k, 'value': v} for k, v in gt_data.items() if v is not None]
                    }
                    test_ground_truth.append(ground_truth)
            
            # Run evaluation
            evaluation_results = st.session_state.evaluator.evaluate_system(test_documents, test_ground_truth)
            
            # Display results
            display_evaluation_results(evaluation_results)

def comprehensive_evaluation():
    """Comprehensive evaluation."""
    st.subheader("Comprehensive Evaluation")
    
    # Evaluation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test size per document type", 10, 100, 50)
    
    with col2:
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)
    
    if st.button("Run Comprehensive Evaluation"):
        with st.spinner("Running comprehensive evaluation..."):
            # This would run a more thorough evaluation
            st.info("Comprehensive evaluation would be implemented here.")
            st.info("This would include cross-validation, multiple metrics, and detailed analysis.")

def custom_evaluation():
    """Custom evaluation."""
    st.subheader("Custom Evaluation")
    
    # Custom parameters
    st.write("Configure custom evaluation parameters:")
    
    # This would allow users to configure custom evaluation parameters
    st.info("Custom evaluation configuration would be implemented here.")

def display_evaluation_results(evaluation_results: dict):
    """Display evaluation results."""
    st.subheader("Evaluation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", evaluation_results.get('total_documents', 0))
    
    with col2:
        accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        st.metric("Overall Accuracy", f"{accuracy:.3f}")
    
    with col3:
        cls_accuracy = evaluation_results.get('classification', {}).get('accuracy', 0)
        st.metric("Classification Accuracy", f"{cls_accuracy:.3f}")
    
    with col4:
        avg_time = evaluation_results.get('processing_performance', {}).get('avg_processing_time', 0)
        st.metric("Avg Processing Time", f"{avg_time:.3f}s")
    
    # Detailed results
    st.subheader("Detailed Results")
    
    # Classification results
    if 'classification' in evaluation_results:
        st.write("**Classification Results:**")
        cls_results = evaluation_results['classification']
        st.write(f"- Accuracy: {cls_results.get('accuracy', 0):.3f}")
        st.write(f"- Precision: {cls_results.get('precision', 0):.3f}")
        st.write(f"- Recall: {cls_results.get('recall', 0):.3f}")
        st.write(f"- F1-Score: {cls_results.get('f1_score', 0):.3f}")
    
    # Field extraction results
    if 'field_extraction' in evaluation_results:
        st.write("**Field Extraction Results:**")
        field_results = evaluation_results['field_extraction']
        
        field_data = []
        for field_name, metrics in field_results.items():
            field_data.append({
                'Field': field_name,
                'Precision': f"{metrics.precision:.3f}",
                'Recall': f"{metrics.recall:.3f}",
                'F1-Score': f"{metrics.f1_score:.3f}",
                'Accuracy': f"{metrics.accuracy:.3f}"
            })
        
        df = pd.DataFrame(field_data)
        st.dataframe(df, use_container_width=True)

def explainability_page():
    """Explainability page."""
    st.header("🔍 Explainability")
    
    # Explainability type selection
    exp_type = st.radio(
        "Select explainability type:",
        ["Single Document", "Batch Analysis", "Confidence Analysis"]
    )
    
    if exp_type == "Single Document":
        single_document_explainability()
    elif exp_type == "Batch Analysis":
        batch_explainability()
    elif exp_type == "Confidence Analysis":
        confidence_analysis()

def single_document_explainability():
    """Single document explainability."""
    st.subheader("Single Document Explainability")
    
    # Document input
    document_text = st.text_area(
        "Enter document text for explainability analysis:",
        height=200,
        placeholder="Enter your document text here..."
    )
    
    if st.button("Analyze Explainability"):
        if document_text.strip():
            with st.spinner("Analyzing explainability..."):
                # Process document
                result = st.session_state.processor.process_document(document_text)
                
                # Convert to dictionary format
                result_dict = {
                    'document_id': result.document_id,
                    'document_type': result.document_type,
                    'fields': [{'name': f.name, 'value': f.value, 'confidence': f.confidence} for f in result.fields],
                    'confidence': result.overall_confidence,
                    'processing_time': result.processing_time
                }
                
                # Generate explanation
                explanation = st.session_state.explainability_engine.explain_document_processing(result_dict)
                
                # Display explanation
                st.subheader("Explanation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Decision:**")
                    st.write(explanation.decision)
                    
                    st.markdown("**Confidence:**")
                    st.write(f"{explanation.confidence:.3f}")
                    
                    st.markdown("**Requires Human Review:**")
                    st.write("Yes" if explanation.requires_human_review else "No")
                
                with col2:
                    st.markdown("**Explanation:**")
                    st.write(explanation.explanation_text)
                
                # Factors
                st.subheader("Confidence Factors")
                factors_df = pd.DataFrame([
                    {'Factor': factor, 'Score': f"{score:.3f}"}
                    for factor, score in explanation.factors.items()
                ])
                st.dataframe(factors_df, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommendations")
                for i, rec in enumerate(explanation.recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Human review request
                if explanation.requires_human_review:
                    st.subheader("Human Review Request")
                    review_request = st.session_state.explainability_engine.create_human_review_request(
                        result_dict, explanation
                    )
                    
                    st.write(f"**Priority:** {review_request.priority}")
                    st.write(f"**Created:** {review_request.created_at}")
                    st.write(f"**Explanation:** {review_request.explanation}")
        else:
            st.error("Please enter document text.")

def batch_explainability():
    """Batch explainability analysis."""
    st.subheader("Batch Explainability Analysis")
    
    # Batch parameters
    batch_size = st.slider("Batch size", 5, 50, 20)
    doc_type = st.selectbox("Document type", [dt.value for dt in DocumentType])
    
    if st.button("Analyze Batch Explainability"):
        with st.spinner("Analyzing batch explainability..."):
            # Generate and process batch
            doc_type_enum = DocumentType(doc_type)
            texts = st.session_state.data_generator.generate_text_batch(doc_type_enum, batch_size)
            results = st.session_state.processor.process_batch(texts)
            
            # Generate explanations
            explanations = []
            for result in results:
                result_dict = {
                    'document_id': result.document_id,
                    'document_type': result.document_type,
                    'fields': [{'name': f.name, 'value': f.value, 'confidence': f.confidence} for f in result.fields],
                    'confidence': result.overall_confidence,
                    'processing_time': result.processing_time
                }
                
                explanation = st.session_state.explainability_engine.explain_document_processing(result_dict)
                explanations.append(explanation)
            
            # Generate report
            report = st.session_state.explainability_engine.generate_explainability_report(explanations)
            
            # Display report
            st.subheader("Batch Explainability Report")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", report.get('total_documents', 0))
            
            with col2:
                auto_approved = report.get('decision_distribution', {}).get('auto_approved', 0)
                st.metric("Auto Approved", auto_approved)
            
            with col3:
                human_review = report.get('decision_distribution', {}).get('human_review_required', 0)
                st.metric("Human Review", human_review)
            
            with col4:
                review_rate = report.get('human_review_rate', 0)
                st.metric("Review Rate", f"{review_rate:.3f}")
            
            # Confidence statistics
            st.subheader("Confidence Statistics")
            conf_stats = report.get('confidence_statistics', {})
            st.write(f"- Mean: {conf_stats.get('mean', 0):.3f}")
            st.write(f"- Std: {conf_stats.get('std', 0):.3f}")
            st.write(f"- Min: {conf_stats.get('min', 0):.3f}")
            st.write(f"- Max: {conf_stats.get('max', 0):.3f}")
            
            # Common recommendations
            st.subheader("Common Recommendations")
            common_recs = report.get('common_recommendations', [])
            for rec, count in common_recs[:5]:
                st.write(f"- {rec} ({count} times)")

def confidence_analysis():
    """Confidence analysis."""
    st.subheader("Confidence Analysis")
    
    # Analysis parameters
    analysis_type = st.selectbox(
        "Analysis type:",
        ["Confidence Distribution", "Calibration Analysis", "Threshold Analysis"]
    )
    
    if st.button("Run Confidence Analysis"):
        with st.spinner("Running confidence analysis..."):
            # Generate sample data
            confidences = np.random.beta(2, 2, 1000)
            
            if analysis_type == "Confidence Distribution":
                # Confidence distribution
                fig = px.histogram(x=confidences, nbins=20, title="Confidence Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.write(f"**Mean:** {np.mean(confidences):.3f}")
                st.write(f"**Median:** {np.median(confidences):.3f}")
                st.write(f"**Std:** {np.std(confidences):.3f}")
            
            elif analysis_type == "Calibration Analysis":
                # Calibration analysis
                st.info("Calibration analysis would be implemented here.")
            
            elif analysis_type == "Threshold Analysis":
                # Threshold analysis
                thresholds = np.arange(0.1, 1.0, 0.1)
                auto_approved = [np.sum(confidences >= t) for t in thresholds]
                human_review = [np.sum(confidences < t) for t in thresholds]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=thresholds, y=auto_approved, name="Auto Approved"))
                fig.add_trace(go.Scatter(x=thresholds, y=human_review, name="Human Review"))
                
                fig.update_layout(
                    title="Threshold Analysis",
                    xaxis_title="Confidence Threshold",
                    yaxis_title="Number of Documents"
                )
                st.plotly_chart(fig, use_container_width=True)

def dashboard_page():
    """Dashboard page."""
    st.header("📊 Dashboard")
    
    # Dashboard type selection
    dashboard_type = st.radio(
        "Select dashboard type:",
        ["Performance Dashboard", "Evaluation Dashboard", "Explainability Dashboard"]
    )
    
    if dashboard_type == "Performance Dashboard":
        performance_dashboard()
    elif dashboard_type == "Evaluation Dashboard":
        evaluation_dashboard()
    elif dashboard_type == "Explainability Dashboard":
        explainability_dashboard()

def performance_dashboard():
    """Performance dashboard."""
    st.subheader("Performance Dashboard")
    
    if st.button("Generate Performance Dashboard"):
        with st.spinner("Generating dashboard..."):
            # Generate sample performance data
            performance_data = {
                'processing_times': np.random.exponential(0.5, 100),
                'throughput': np.random.normal(10, 2, 100),
                'document_sizes': np.random.normal(500, 200, 100),
                'batch_times': np.random.exponential(2, 20)
            }
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Processing time distribution
                fig1 = px.histogram(x=performance_data['processing_times'], title="Processing Time Distribution")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Throughput over time
                fig2 = px.line(y=performance_data['throughput'], title="Throughput Over Time")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Processing Time", f"{np.mean(performance_data['processing_times']):.3f}s")
            
            with col2:
                st.metric("Avg Throughput", f"{np.mean(performance_data['throughput']):.1f} docs/sec")
            
            with col3:
                st.metric("Max Processing Time", f"{np.max(performance_data['processing_times']):.3f}s")
            
            with col4:
                st.metric("Min Processing Time", f"{np.min(performance_data['processing_times']):.3f}s")

def evaluation_dashboard():
    """Evaluation dashboard."""
    st.subheader("Evaluation Dashboard")
    
    if st.button("Generate Evaluation Dashboard"):
        with st.spinner("Generating evaluation dashboard..."):
            # Generate sample evaluation data
            doc_types = ['invoice', 'receipt', 'contract']
            type_counts = [50, 30, 20]
            
            # Document type distribution
            fig1 = px.pie(values=type_counts, names=doc_types, title="Document Type Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Accuracy metrics
            accuracy_data = {
                'Document Type': doc_types,
                'Accuracy': [0.95, 0.92, 0.88],
                'Precision': [0.94, 0.91, 0.87],
                'Recall': [0.96, 0.93, 0.89],
                'F1-Score': [0.95, 0.92, 0.88]
            }
            
            df = pd.DataFrame(accuracy_data)
            fig2 = px.bar(df, x='Document Type', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                         title="Accuracy Metrics by Document Type")
            st.plotly_chart(fig2, use_container_width=True)

def explainability_dashboard():
    """Explainability dashboard."""
    st.subheader("Explainability Dashboard")
    
    if st.button("Generate Explainability Dashboard"):
        with st.spinner("Generating explainability dashboard..."):
            # Generate sample explainability data
            confidences = np.random.beta(2, 2, 1000)
            
            # Confidence distribution
            fig1 = px.histogram(x=confidences, nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Decision distribution
            decisions = ['auto_approved', 'human_review_required', 'auto_rejected']
            decision_counts = [700, 250, 50]
            
            fig2 = px.pie(values=decision_counts, names=decisions, title="Decision Distribution")
            st.plotly_chart(fig2, use_container_width=True)

def about_page():
    """About page."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Document Processing Automation Demo
    
    This is a comprehensive demonstration of a document processing automation system
    designed for research and educational purposes.
    
    ### Features
    
    - **Document Processing**: Extract structured data from unstructured documents
    - **Batch Processing**: Process multiple documents efficiently
    - **Evaluation**: Comprehensive evaluation metrics and analysis
    - **Explainability**: Understand how decisions are made
    - **Dashboard**: Visualize performance and results
    
    ### Supported Document Types
    
    - **Invoices**: Extract invoice numbers, dates, amounts, customers
    - **Receipts**: Extract receipt numbers, dates, amounts, merchants
    - **Contracts**: Extract contract IDs, dates, parties, values
    
    ### Technology Stack
    
    - **Python 3.10+**: Core programming language
    - **Streamlit**: Interactive web interface
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation and analysis
    - **NumPy**: Numerical computing
    - **Scikit-learn**: Machine learning utilities
    
    ### Disclaimer
    
    This system is designed for research and educational purposes only.
    It is not intended for automated decision-making without human review.
    All results should be validated by qualified professionals before use
    in production environments.
    
    ### Contact
    
    For questions or feedback, please contact the development team.
    """)
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Version:** 1.0.0")
        st.write("**Python Version:** 3.10+")
        st.write("**Streamlit Version:** 1.25.0+")
    
    with col2:
        st.write("**Last Updated:** 2024-01-01")
        st.write("**License:** MIT")
        st.write("**Repository:** [GitHub](https://github.com/example/document-processing-automation)")

if __name__ == "__main__":
    main()
