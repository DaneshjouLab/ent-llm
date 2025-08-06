# Evaluation
import os
import io
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from contextlib import redirect_stdout
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score)
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

warnings.filterwarnings('ignore')

def build_evaluation_df(processed_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build evaluation DataFrame by merging processed_df with LLM results.
    
    Args:
        processed_df: Original DataFrame with surgery outcomes
        results_df: DataFrame with LLM results including llm_caseID, decision, confidence, reasoning
    
    Returns:
        DataFrame ready for evaluation with binary columns
    """
    
    # Merge the DataFrames
    processed_llm = pd.merge(
        processed_df,
        results_df[['llm_caseID', 'decision', 'confidence', 'reasoning']],
        on='llm_caseID',
        how='left'
    )
    
    # Convert LLM decision to binary (1 for "Yes", 0 for "No")
    processed_llm['llm_surgery'] = processed_llm['decision'].map({'Yes': 1, 'No': 0})
    
    # Convert had_surgery to binary (1 for True, 0 for False)
    processed_llm['actual_surgery'] = processed_llm['had_surgery'].map({True: 1, False: 0})
    
    # Only keep rows where we have both predictions and actual outcomes
    processed_llm = processed_llm.dropna(subset=['llm_surgery', 'actual_surgery'])
        
    print(f"Evaluation dataset: {len(processed_llm)} cases with complete data")
    print(f"Actual surgery rate: {processed_llm['actual_surgery'].mean():.1%}")
    print(f"LLM predicted surgery rate: {processed_llm['llm_surgery'].mean():.1%}")
    
    # Show the value counts for verification
    print(f"\nActual surgery distribution:")
    print(f"  No surgery (False): {(processed_llm['actual_surgery'] == 0).sum()}")
    print(f"  Surgery (True): {(processed_llm['actual_surgery'] == 1).sum()}")
    
    print(f"\nLLM predicted surgery distribution:")
    print(f"  No surgery (No): {(processed_llm['llm_surgery'] == 0).sum()}")
    print(f"  Surgery (Yes): {(processed_llm['llm_surgery'] == 1).sum()}")
    
    return processed_llm

def calculate_performance_metrics(eval_df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        eval_df: DataFrame with actual_surgery and llm_surgery columns (binary)
    
    Returns:
        Dictionary of performance metrics
    """
    
    y_true = eval_df['actual_surgery']
    y_pred = eval_df['llm_surgery']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # AUC-ROC (if we have confidence scores)
    auc_roc = None
    if 'confidence' in eval_df.columns and eval_df['confidence'].notna().sum() > 0:
        # Use confidence as probability for AUC calculation
        confidence_scores = eval_df['confidence'].fillna(5) / 10  # Convert to 0-1 scale
        auc_roc = roc_auc_score(y_true, confidence_scores)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'negative_predictive_value': npv,
        'auc_roc': auc_roc,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_cases': len(eval_df)
    }
    
    return metrics

def analyze_confidence_accuracy_relationship(eval_df: pd.DataFrame):
    """
    Analyze the relationship between LLM confidence and accuracy.
    
    Args:
        eval_df: DataFrame with predictions, outcomes, and confidence scores
    """
    
    if 'confidence' not in eval_df.columns or eval_df['confidence'].isna().all():
        print("No confidence data available for analysis")
        return None
    
    # Calculate accuracy by confidence level
    eval_df['correct_prediction'] = (eval_df['actual_surgery'] == eval_df['llm_surgery']).astype(int)
    
    # Group by confidence level
    confidence_analysis = eval_df.groupby('confidence').agg({
        'correct_prediction': ['count', 'mean'],
        'llm_surgery': 'mean',
        'actual_surgery': 'mean'
    }).round(3)
    
    confidence_analysis.columns = ['n_cases', 'accuracy', 'predicted_surgery_rate', 'actual_surgery_rate']
    
    print("\n=== Accuracy by Confidence Level ===")
    print(confidence_analysis)
    
    # Correlation between confidence and accuracy
    correlation = eval_df['confidence'].corr(eval_df['correct_prediction'])
    print(f"\nCorrelation between confidence and accuracy: {correlation:.3f}")
    
    # Statistical test
    if len(eval_df) > 10:
        corr_stat, p_value = stats.pearsonr(eval_df['confidence'].fillna(5), eval_df['correct_prediction'])
        print(f"Pearson correlation: {corr_stat:.3f} (p-value: {p_value:.3f})")
    
    return confidence_analysis

def create_evaluation_visualizations(eval_df: pd.DataFrame, metrics: dict):
    """
    Create visualizations for model evaluation.
    
    Args:
        eval_df: DataFrame with evaluation data
        metrics: Dictionary of performance metrics
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(eval_df['actual_surgery'], eval_df['llm_surgery'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['No Surgery', 'Surgery'], 
                yticklabels=['No Surgery', 'Surgery'])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('LLM Predicted')
    axes[0,0].set_ylabel('Actual Outcome')
    
    # 2. Confidence Distribution by Prediction Correctness
    if 'confidence' in eval_df.columns and eval_df['confidence'].notna().sum() > 0:
        eval_df['correct_prediction'] = (eval_df['actual_surgery'] == eval_df['llm_surgery'])
        
        # Get unique confidence values and count correct/incorrect for each
        confidence_counts = eval_df.groupby(['confidence', 'correct_prediction']).size().unstack(fill_value=0)
        
        # Create side-by-side bar chart
        confidence_values = confidence_counts.index
        width = 0.35
        x_pos = np.arange(len(confidence_values))
        
        if False in confidence_counts.columns:  # Incorrect predictions
            axes[0,1].bar(x_pos - width/2, confidence_counts[False], width, 
                         label='Incorrect', color='red', alpha=0.7)
        
        if True in confidence_counts.columns:   # Correct predictions
            axes[0,1].bar(x_pos + width/2, confidence_counts[True], width, 
                         label='Correct', color='green', alpha=0.7)
        
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Confidence Distribution by Correctness')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(confidence_values)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Confidence Analysis (No Data)')
    
    # 3. Performance Metrics Bar Chart
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    values = [metrics[m] for m in metrics_to_plot]
    
    bars = axes[1,0].bar(metrics_to_plot, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Performance Metrics')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Prediction Distribution
    pred_dist = eval_df['llm_surgery'].value_counts()
    actual_dist = eval_df['actual_surgery'].value_counts()
    
    x = np.arange(2)
    width = 0.35
    
    axes[1,1].bar(x - width/2, [pred_dist.get(0, 0), pred_dist.get(1, 0)], width, 
                  label='LLM Predicted', alpha=0.7)
    axes[1,1].bar(x + width/2, [actual_dist.get(0, 0), actual_dist.get(1, 0)], width, 
                  label='Actual', alpha=0.7)
    
    axes[1,1].set_xlabel('Surgery Decision')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Prediction vs Actual Distribution')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(['No Surgery', 'Surgery'])
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # Return the figure instead of showing it
    return fig

def generate_evaluation_report(eval_df: pd.DataFrame, filename: str = None) -> str:
    """
    Generate evaluation report with embedded plots as PDF.

    Args:
        eval_df: DataFrame with evaluation data
        filename: Optional filename for the PDF

    Returns:
        String path of the saved PDF file
    """
    
    # Ensure output directory exists
    output_dir = "/content/ent-llm/outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_surgery_eval_report_{timestamp}.pdf"
    
    if not filename.endswith('.pdf'):
        filename += '.pdf'
        
    full_path = os.path.join(output_dir, filename)

    # Temporary directory for saving plots
    temp_dir = tempfile.mkdtemp()

    # Buffer to capture printed outputs
    output_buffer = io.StringIO()

    try:
        # Step 1: Run analysis and capture output
        metrics = calculate_performance_metrics(eval_df)
        
        with redirect_stdout(output_buffer):
            print(f"Dataset Overview:")
            print(f"Total cases evaluated: {len(eval_df)}")
            print(f"Actual surgery rate: {eval_df['actual_surgery'].mean():.1%}")
            print(f"LLM predicted surgery rate: {eval_df['llm_surgery'].mean():.1%}")
            print()

            print(f"Performance Metrics:")
            for key in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'negative_predictive_value']:
                val = metrics[key]
                if val is not None:
                    print(f"  {key.replace('_', ' ').title()}: {val:.3f}")
            
            if metrics['auc_roc'] is not None:
                print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print()

            print(f"Confusion Matrix Counts:")
            print(f"  True Positives: {metrics['true_positives']}")
            print(f"  True Negatives: {metrics['true_negatives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
            print()

        # Run confidence analysis
        confidence_analysis_df = analyze_confidence_accuracy_relationship(eval_df)
        
        with redirect_stdout(output_buffer):
            print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_text = output_buffer.getvalue()

        # Step 2: Create visualizations
        fig = create_evaluation_visualizations(eval_df, metrics)
        plot_path = os.path.join(temp_dir, "eval_plots.png")
        fig.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # Step 3: Create PDF
        doc = SimpleDocTemplate(full_path, pagesize=letter, 
                               leftMargin=50, rightMargin=50, 
                               topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        elements = []

        # Add title
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center alignment
        elements.append(Paragraph("LLM Surgery Prediction Evaluation Report", title_style))
        elements.append(Spacer(1, 20))

        # Add report text
        for line in report_text.strip().split('\n'):
            if line.strip():  # Skip empty lines
                if line.startswith('='):
                    # Header line
                    elements.append(Paragraph(line, styles["Heading2"]))
                elif line.startswith('  '):
                    # Indented line
                    elements.append(Paragraph(line, styles["Code"]))
                else:
                    elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 6))

        # Add confidence analysis table if it exists
        if confidence_analysis_df is not None and not confidence_analysis_df.empty:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Accuracy by Confidence Level", styles["Heading2"]))
            elements.append(Spacer(1, 10))
            
            # Convert DataFrame to table data
            table_data = [['Confidence'] + list(confidence_analysis_df.columns)]
            for idx, row in confidence_analysis_df.iterrows():
                table_data.append([str(idx)] + [f"{val:.3f}" if isinstance(val, (int, float)) else str(val) 
                                              for val in row.values])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)

        # Add plot image to PDF
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Performance Visualizations", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        
        # Scale image to fit page
        from reportlab.lib.utils import ImageReader
        img = ImageReader(plot_path)
        img_width, img_height = img.getSize()
        aspect = img_height / float(img_width)
        
        # Fit to page width with some margin
        page_width = letter[0] - 100  # Leave margins
        scaled_width = min(page_width, 500)
        scaled_height = scaled_width * aspect
        
        elements.append(Image(plot_path, width=scaled_width, height=scaled_height))

        # Build the PDF
        doc.build(elements)

        print(f"Evaluation report with plots saved as: {full_path}")

    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    return full_path

def save_llm_eval_csv(eval_df: pd.DataFrame, filename: str = None) -> str:
    """
    Save detailed evaluation data as CSV.
    
    Args:
        eval_df: DataFrame with evaluation data
        filename: Optional filename for the CSV
    
    Returns:
        String path of the saved CSV file
    """
    # Ensure output directory exists
    output_dir = "/content/ent-llm/outputs/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_detailed_eval_{timestamp}.csv"
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    full_path = os.path.join(output_dir, filename)
    
    # Define required columns
    required_columns = ['llm_caseID', 'confidence', 'llm_surgery', 'actual_surgery', 'reasoning']
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in required_columns if col in eval_df.columns]
    missing_columns = [col for col in required_columns if col not in eval_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in dataframe: {missing_columns}")
    
    if not available_columns:
        raise ValueError("None of the required columns found in dataframe")
    
    # Create subset dataframe
    printable_data = eval_df[available_columns].copy()
    
    # Save to CSV
    printable_data.to_csv(full_path, index=False)
    
    print(f"Detailed evaluation data saved to: {full_path}")
    print(f"Saved {len(printable_data)} rows with columns: {list(printable_data.columns)}")
    
    return full_path
