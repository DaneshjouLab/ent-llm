# Radiology Report Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, roc_auc_score)
from scipy import stats
import os
import io
import tempfile
import shutil
from datetime import datetime
from contextlib import redirect_stdout

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import warnings

warnings.filterwarnings('ignore')

def calculate_performance_metrics(eval_df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive performance metrics segmented by radiology availability.

    Args:
        eval_df: DataFrame with actual_surgery, llm_surgery, and has_radiology columns

    Returns:
        Dictionary of performance metrics for each group
    """

    results = {}

    # Overall metrics
    results['overall'] = calculate_single_group_metrics(eval_df)

    # Metrics by radiology availability
    for has_rad, group_name in [(True, 'with_radiology'), (False, 'without_radiology')]:
        subset = eval_df[eval_df['has_radiology'] == has_rad]
        if len(subset) > 0:
            results[group_name] = calculate_single_group_metrics(subset)
        else:
            results[group_name] = None

    return results

def calculate_single_group_metrics(df: pd.DataFrame) -> dict:
    """Calculate metrics for a single group."""

    y_true = df['actual_surgery']
    y_pred = df['llm_surgery']

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
    if 'confidence' in df.columns and df['confidence'].notna().sum() > 0:
        # Use confidence as probability for AUC calculation
        confidence_scores = df['confidence'].fillna(5) / 10  # Convert to 0-1 scale
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
        'total_cases': len(df),
        'actual_surgery_rate': y_true.mean(),
        'predicted_surgery_rate': y_pred.mean()
    }

    return metrics

def analyze_confidence_by_radiology(eval_df: pd.DataFrame):
    """
    Analyze the relationship between LLM confidence, accuracy, and radiology availability.

    Args:
        eval_df: DataFrame with predictions, outcomes, confidence scores, and radiology flag
    """

    if 'confidence' not in eval_df.columns or eval_df['confidence'].isna().all():
        print("No confidence data available for analysis")
        return None

    eval_df['correct_prediction'] = (eval_df['actual_surgery'] == eval_df['llm_surgery']).astype(int)

    confidence_results = {}

    print("\n=== Confidence Analysis by Radiology Availability ===")

    for has_rad, group_name in [(True, 'With Radiology'), (False, 'Without Radiology')]:
        subset = eval_df[eval_df['has_radiology'] == has_rad]

        if len(subset) == 0:
            print(f"\n{group_name}: No data available")
            continue

        print(f"\n{group_name} (n={len(subset)}):")

        # Group by confidence level
        confidence_analysis = subset.groupby('confidence').agg({
            'correct_prediction': ['count', 'mean'],
            'llm_surgery': 'mean',
            'actual_surgery': 'mean'
        }).round(3)

        confidence_analysis.columns = ['n_cases', 'accuracy', 'predicted_surgery_rate', 'actual_surgery_rate']
        print(confidence_analysis)

        # Store results
        confidence_results[group_name] = confidence_analysis

        # Correlation between confidence and accuracy
        if len(subset) > 5:
            correlation = subset['confidence'].corr(subset['correct_prediction'])
            print(f"Correlation between confidence and accuracy: {correlation:.3f}")

    return confidence_results

def create_radiology_segmented_visualizations(eval_df: pd.DataFrame, metrics_dict: dict):
    """
    Create comprehensive visualizations comparing performance with and without radiology.
    Returns two figures: main analysis and detailed confidence analysis.
    """

    # Figure 1: Main Performance Analysis
    fig1 = plt.figure(figsize=(20, 16))
    gs1 = fig1.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Confusion Matrices (Top row)
    for i, (has_rad, title) in enumerate([(True, 'With Radiology'), (False, 'Without Radiology')]):
        ax = fig1.add_subplot(gs1[0, i])
        subset = eval_df[eval_df['has_radiology'] == has_rad]

        if len(subset) > 0:
            cm = confusion_matrix(subset['actual_surgery'], subset['llm_surgery'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Surgery', 'Surgery'],
                       yticklabels=['No Surgery', 'Surgery'])
            ax.set_title(f'Confusion Matrix - {title}\n(n={len(subset)})')
            ax.set_xlabel('LLM Predicted')
            ax.set_ylabel('Actual Outcome')
        else:
            ax.text(0.5, 0.5, f'No data\nfor {title}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Confusion Matrix - {title}')

    # 2. Performance Metrics Comparison (Top right)
    ax = fig1.add_subplot(gs1[0, 2])
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    with_rad_values = [metrics_dict['with_radiology'][m] if metrics_dict['with_radiology'] else 0 for m in metrics_to_plot]
    without_rad_values = [metrics_dict['without_radiology'][m] if metrics_dict['without_radiology'] else 0 for m in metrics_to_plot]

    ax.bar(x - width/2, with_rad_values, width, label='With Radiology', alpha=0.8)
    ax.bar(x + width/2, without_rad_values, width, label='Without Radiology', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for i, (with_val, without_val) in enumerate(zip(with_rad_values, without_rad_values)):
        if with_val > 0:
            ax.text(i - width/2, with_val + 0.02, f'{with_val:.3f}', ha='center', va='bottom', fontsize=8)
        if without_val > 0:
            ax.text(i + width/2, without_val + 0.02, f'{without_val:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Surgery Rate Comparison (Second row, left)
    ax = fig1.add_subplot(gs1[1, 0])
    categories = ['Actual', 'LLM Predicted']

    with_rad_subset = eval_df[eval_df['has_radiology'] == True]
    without_rad_subset = eval_df[eval_df['has_radiology'] == False]

    with_rad_rates = [
        with_rad_subset['actual_surgery'].mean() if len(with_rad_subset) > 0 else 0,
        with_rad_subset['llm_surgery'].mean() if len(with_rad_subset) > 0 else 0
    ]
    without_rad_rates = [
        without_rad_subset['actual_surgery'].mean() if len(without_rad_subset) > 0 else 0,
        without_rad_subset['llm_surgery'].mean() if len(without_rad_subset) > 0 else 0
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, with_rad_rates, width, label='With Radiology', alpha=0.8)
    ax.bar(x + width/2, without_rad_rates, width, label='Without Radiology', alpha=0.8)

    ax.set_ylabel('Surgery Rate')
    ax.set_title('Surgery Rates Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add percentage labels on bars
    for i, (with_val, without_val) in enumerate(zip(with_rad_rates, without_rad_rates)):
        if with_val > 0:
            ax.text(i - width/2, with_val + 0.02, f'{with_val:.1%}', ha='center', va='bottom')
        if without_val > 0:
            ax.text(i + width/2, without_val + 0.02, f'{without_val:.1%}', ha='center', va='bottom')

    # 4. Case Count Breakdown (Second row, middle and right)
    ax = fig1.add_subplot(gs1[1, 1:])

    with_rad_subset = eval_df[eval_df['has_radiology'] == True]
    without_rad_subset = eval_df[eval_df['has_radiology'] == False]

    categories = ['Total Cases', 'Actual Surgery', 'Predicted Surgery', 'Correct Predictions']
    with_rad_counts = [
        len(with_rad_subset),
        with_rad_subset['actual_surgery'].sum(),
        with_rad_subset['llm_surgery'].sum(),
        (with_rad_subset['actual_surgery'] == with_rad_subset['llm_surgery']).sum()
    ]
    without_rad_counts = [
        len(without_rad_subset),
        without_rad_subset['actual_surgery'].sum(),
        without_rad_subset['llm_surgery'].sum(),
        (without_rad_subset['actual_surgery'] == without_rad_subset['llm_surgery']).sum()
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, with_rad_counts, width, label='With Radiology', alpha=0.8)
    bars2 = ax.bar(x + width/2, without_rad_counts, width, label='Without Radiology', alpha=0.8)

    ax.set_ylabel('Count')
    ax.set_title('Case Counts Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom')

    # 5. Error Analysis (Third row, spanning all columns)
    ax = fig1.add_subplot(gs1[2, :])

    error_data = []
    for has_rad, group_name in [(True, 'With Radiology'), (False, 'Without Radiology')]:
        subset = eval_df[eval_df['has_radiology'] == has_rad]
        if len(subset) > 0:
            tp = ((subset['actual_surgery'] == 1) & (subset['llm_surgery'] == 1)).sum()
            tn = ((subset['actual_surgery'] == 0) & (subset['llm_surgery'] == 0)).sum()
            fp = ((subset['actual_surgery'] == 0) & (subset['llm_surgery'] == 1)).sum()
            fn = ((subset['actual_surgery'] == 1) & (subset['llm_surgery'] == 0)).sum()

            error_data.append({
                'Group': group_name,
                'True Positive': tp,
                'True Negative': tn,
                'False Positive': fp,
                'False Negative': fn,
                'Total': len(subset)
            })

    if error_data:
        groups = [d['Group'] for d in error_data]
        tp_vals = [d['True Positive'] for d in error_data]
        tn_vals = [d['True Negative'] for d in error_data]
        fp_vals = [d['False Positive'] for d in error_data]
        fn_vals = [d['False Negative'] for d in error_data]

        x = np.arange(len(groups))

        ax.bar(x, tp_vals, label='True Positive', color='green', alpha=0.8)
        ax.bar(x, tn_vals, bottom=tp_vals, label='True Negative', color='lightgreen', alpha=0.8)
        ax.bar(x, fp_vals, bottom=np.array(tp_vals) + np.array(tn_vals), label='False Positive', color='orange', alpha=0.8)
        ax.bar(x, fn_vals, bottom=np.array(tp_vals) + np.array(tn_vals) + np.array(fp_vals), label='False Negative', color='red', alpha=0.8)

        ax.set_ylabel('Count')
        ax.set_title('Prediction Outcomes by Radiology Availability')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add total counts on top of bars
        for i, total in enumerate([d['Total'] for d in error_data]):
            ax.text(i, total + 1, f'n={total}', ha='center', va='bottom', fontweight='bold')

    # 6. Metrics Summary Table (Fourth row)
    ax = fig1.add_subplot(gs1[3, :])
    ax.axis('tight')
    ax.axis('off')

    # Create summary table data
    table_data = []
    headers = ['Metric', 'With Radiology', 'Without Radiology', 'Difference']

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
        with_val = metrics_dict['with_radiology'][metric] if metrics_dict['with_radiology'] else 0
        without_val = metrics_dict['without_radiology'][metric] if metrics_dict['without_radiology'] else 0
        diff = with_val - without_val

        table_data.append([
            metric.replace('_', ' ').title(),
            f'{with_val:.3f}',
            f'{without_val:.3f}',
            f'{diff:+.3f}'
        ])

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Metrics Summary', pad=20)

    plt.suptitle('LLM Performance Analysis: With vs Without Radiology Reports', fontsize=16, y=0.98)

    # Figure 2: Confidence Analysis (if available)
    fig2 = None
    if 'confidence' in eval_df.columns and eval_df['confidence'].notna().sum() > 0:
        fig2 = plt.figure(figsize=(16, 10))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        eval_df['correct_prediction'] = (eval_df['actual_surgery'] == eval_df['llm_surgery'])

        for i, (has_rad, title) in enumerate([(True, 'With Radiology'), (False, 'Without Radiology')]):
            ax = fig2.add_subplot(gs2[0, i])
            subset = eval_df[eval_df['has_radiology'] == has_rad]

            if len(subset) > 0 and subset['confidence'].notna().sum() > 0:
                confidence_counts = subset.groupby(['confidence', 'correct_prediction']).size().unstack(fill_value=0)

                confidence_values = confidence_counts.index
                width = 0.35
                x_pos = np.arange(len(confidence_values))

                if False in confidence_counts.columns:
                    ax.bar(x_pos - width/2, confidence_counts[False], width,
                          label='Incorrect', color='red', alpha=0.7)

                if True in confidence_counts.columns:
                    ax.bar(x_pos + width/2, confidence_counts[True], width,
                          label='Correct', color='green', alpha=0.7)

                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Count')
                ax.set_title(f'Confidence vs Correctness\n{title} (n={len(subset)})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(confidence_values)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No confidence data\nfor {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Confidence Analysis - {title}')

        # # Overall confidence distribution
        # ax = fig2.add_subplot(gs2[1, :])
        # if eval_df['confidence'].notna().sum() > 0:
        #     for has_rad, label, color in [(True, 'With Radiology', 'blue'), (False, 'Without Radiology', 'red')]:
        #         subset = eval_df[eval_df['has_radiology'] == has_rad]
        #         if len(subset) > 0:
        #             confidence_acc = subset.groupby('confidence')['correct_prediction'].agg(['count', 'mean']).reset_index()
        #             ax.scatter(confidence_acc['confidence'], confidence_acc['mean'],
        #                      s=confidence_acc['count']*10, alpha=0.6, label=f'{label} (accuracy)', color=color)

        #     ax.set_xlabel('Confidence Score')
        #     ax.set_ylabel('Accuracy')
        #     ax.set_title('Accuracy vs Confidence by Radiology Availability\n(Bubble size = number of cases)')
        #     ax.legend()
        #     ax.grid(True, alpha=0.3)
        #     ax.set_ylim(-0.1, 1.1)

        # plt.suptitle('Confidence Analysis by Radiology Availability', fontsize=14, y=0.98)

    return fig1, fig2

def generate_radiology_evaluation_report(eval_df: pd.DataFrame, filename: str = None) -> str:
    """
    Generate comprehensive radiology evaluation report as PDF.

    Args:
        eval_df: DataFrame with evaluation data including has_radiology column
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
        filename = f"llm_radiology_eval_report_{timestamp}.pdf"

    if not filename.endswith('.pdf'):
        filename += '.pdf'

    full_path = os.path.join(output_dir, filename)

    # Temporary directory for saving plots
    temp_dir = tempfile.mkdtemp()

    # Buffer to capture printed outputs
    output_buffer = io.StringIO()

    try:
        # Step 1: Calculate metrics and run analysis
        metrics_dict = calculate_performance_metrics(eval_df)

        with redirect_stdout(output_buffer):
            print(f"Dataset Overview:")
            print(f"Total cases evaluated: {len(eval_df)}")
            print(f"Cases with radiology: {(eval_df['has_radiology'] == True).sum()} ({(eval_df['has_radiology'] == True).mean():.1%})")
            print(f"Cases without radiology: {(eval_df['has_radiology'] == False).sum()} ({(eval_df['has_radiology'] == False).mean():.1%})")
            print()

            # Overall performance
            print(f"Overall Performance:")
            overall = metrics_dict['overall']
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'negative_predictive_value']:
                if key in overall:
                    print(f"  {key.replace('_', ' ').title()}: {overall[key]:.3f}")
            if overall.get('auc_roc'):
                print(f"  AUC-ROC: {overall['auc_roc']:.3f}")
            print()

            # With radiology performance
            if metrics_dict['with_radiology']:
                print(f"Performance WITH Radiology Reports:")
                with_rad = metrics_dict['with_radiology']
                print(f"  Cases: {with_rad['total_cases']}")
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                    print(f"  {key.replace('_', ' ').title()}: {with_rad[key]:.3f}")
                if with_rad.get('auc_roc'):
                    print(f"  AUC-ROC: {with_rad['auc_roc']:.3f}")
                print(f"  Actual Surgery Rate: {with_rad['actual_surgery_rate']:.1%}")
                print(f"  Predicted Surgery Rate: {with_rad['predicted_surgery_rate']:.1%}")
                print()

            # Without radiology performance
            if metrics_dict['without_radiology']:
                print(f"Performance WITHOUT Radiology Reports:")
                without_rad = metrics_dict['without_radiology']
                print(f"  Cases: {without_rad['total_cases']}")
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                    print(f"  {key.replace('_', ' ').title()}: {without_rad[key]:.3f}")
                if without_rad.get('auc_roc'):
                    print(f"  AUC-ROC: {without_rad['auc_roc']:.3f}")
                print(f"  Actual Surgery Rate: {without_rad['actual_surgery_rate']:.1%}")
                print(f"  Predicted Surgery Rate: {without_rad['predicted_surgery_rate']:.1%}")
                print()

            # Performance comparison
            if metrics_dict['with_radiology'] and metrics_dict['without_radiology']:
                print(f"Performance Comparison (With - Without Radiology):")
                with_rad = metrics_dict['with_radiology']
                without_rad = metrics_dict['without_radiology']

                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                    diff = with_rad[key] - without_rad[key]
                    print(f"  {key.replace('_', ' ').title()} Difference: {diff:+.3f}")
                print()

        # Run confidence analysis
        confidence_results = analyze_confidence_by_radiology(eval_df)

        with redirect_stdout(output_buffer):
            print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_text = output_buffer.getvalue()

        # Step 2: Create visualizations
        fig1, fig2 = create_radiology_segmented_visualizations(eval_df, metrics_dict)

        # Save main plot
        plot1_path = os.path.join(temp_dir, "radiology_analysis_main.png")
        fig1.savefig(plot1_path, bbox_inches='tight', dpi=150)
        plt.close(fig1)

        # Save confidence plot if it exists
        plot2_path = None
        if fig2 is not None:
            plot2_path = os.path.join(temp_dir, "confidence_analysis.png")
            fig2.savefig(plot2_path, bbox_inches='tight', dpi=150)
            plt.close(fig2)

        # Step 3: Create PDF
        doc = SimpleDocTemplate(full_path, pagesize=A4,
                               leftMargin=50, rightMargin=50,
                               topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        elements = []

        # Add title
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center alignment
        elements.append(Paragraph("LLM Surgery Prediction: Radiology Impact Analysis", title_style))
        elements.append(Spacer(1, 20))

        # Add report text
        for line in report_text.strip().split('\n'):
            if line.strip():  # Skip empty lines
                if line.startswith('===') or line.endswith(':') and not line.startswith('  '):
                    # Header line
                    elements.append(Paragraph(line.replace('=', ''), styles["Heading2"]))
                elif line.startswith('  '):
                    # Indented line
                    elements.append(Paragraph(line, styles["Code"]))
                else:
                    elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 6))

        # Add confidence analysis tables if they exist
        if confidence_results:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Detailed Confidence Analysis", styles["Heading2"]))

            for group_name, confidence_df in confidence_results.items():
                if confidence_df is not None and not confidence_df.empty:
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"{group_name}", styles["Heading3"]))
                    elements.append(Spacer(1, 5))

                    # Convert DataFrame to table data
                    table_data = [['Confidence'] + list(confidence_df.columns)]
                    for idx, row in confidence_df.iterrows():
                        table_data.append([str(idx)] + [f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
                                                      for val in row.values])

                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)

        # Add main analysis plots
        elements.append(PageBreak())
        elements.append(Paragraph("Performance Analysis Visualizations", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        # Scale main image to fit page
        img1 = ImageReader(plot1_path)
        img1_width, img1_height = img1.getSize()
        aspect1 = img1_height / float(img1_width)
        page_width = A4[0] - 100  # Leave margins
        scaled_width1 = min(page_width, 500)
        scaled_height1 = scaled_width1 * aspect1

        elements.append(Image(plot1_path, width=scaled_width1, height=scaled_height1))

        # Add confidence analysis plot if it exists
        if plot2_path:
            elements.append(PageBreak())
            elements.append(Paragraph("Confidence Analysis Visualizations", styles["Heading2"]))
            elements.append(Spacer(1, 10))

            img2 = ImageReader(plot2_path)
            img2_width, img2_height = img2.getSize()
            aspect2 = img2_height / float(img2_width)
            scaled_width2 = min(page_width, 500)
            scaled_height2 = scaled_width2 * aspect2

            elements.append(Image(plot2_path, width=scaled_width2, height=scaled_height2))

        # Build the PDF
        doc.build(elements)

        print(f"Radiology evaluation report saved as: {full_path}")

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

def radiology_strat_analysis(processed_llm: pd.DataFrame, filename: str = None):
    """
    Run the complete radiology-segmented analysis and generate PDF report.

    Args:
        processed_llm: DataFrame that already has llm_surgery, actual_surgery columns
        filename: Optional filename for the PDF report

    Returns:
        Tuple of (processed_df, metrics_dict, report_path)
    """

    # Ensure has_radiology column exists
    if 'has_radiology' not in processed_llm.columns:
        processed_llm['has_radiology'] = processed_llm['radiology_reports'].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else False
        )

    print("="*80)
    print("RUNNING COMPREHENSIVE RADIOLOGY IMPACT ANALYSIS")
    print("="*80)

    # Calculate metrics by radiology availability
    metrics_dict = calculate_performance_metrics(processed_llm)

    # Analyze confidence by radiology availability
    confidence_analysis = analyze_confidence_by_radiology(processed_llm)

    # Print summary to console
    print_radiology_summary(metrics_dict)

    # Generate PDF report
    report_path = generate_radiology_evaluation_report(processed_llm, filename)

    return processed_llm, metrics_dict, report_path

def print_radiology_summary(metrics_dict: dict):
    """Print a comprehensive summary of the radiology-segmented analysis."""

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS: LLM PERFORMANCE WITH vs WITHOUT RADIOLOGY")
    print("="*80)

    # Overall statistics
    print(f"\nOVERALL PERFORMANCE:")
    overall = metrics_dict['overall']
    print(f"  Total Cases: {overall['total_cases']}")
    print(f"  Accuracy: {overall['accuracy']:.3f}")
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall: {overall['recall']:.3f}")
    print(f"  F1-Score: {overall['f1_score']:.3f}")
    if overall.get('auc_roc'):
        print(f"  AUC-ROC: {overall['auc_roc']:.3f}")

    # With radiology performance
    if metrics_dict['with_radiology']:
        print(f"\nWITH RADIOLOGY REPORTS:")
        with_rad = metrics_dict['with_radiology']
        print(f"  Cases: {with_rad['total_cases']}")
        print(f"  Accuracy: {with_rad['accuracy']:.3f}")
        print(f"  Precision: {with_rad['precision']:.3f}")
        print(f"  Recall: {with_rad['recall']:.3f}")
        print(f"  F1-Score: {with_rad['f1_score']:.3f}")
        print(f"  Specificity: {with_rad['specificity']:.3f}")
        if with_rad.get('auc_roc'):
            print(f"  AUC-ROC: {with_rad['auc_roc']:.3f}")
        print(f"  Actual Surgery Rate: {with_rad['actual_surgery_rate']:.1%}")
        print(f"  Predicted Surgery Rate: {with_rad['predicted_surgery_rate']:.1%}")

    # Without radiology performance
    if metrics_dict['without_radiology']:
        print(f"\nWITHOUT RADIOLOGY REPORTS:")
        without_rad = metrics_dict['without_radiology']
        print(f"  Cases: {without_rad['total_cases']}")
        print(f"  Accuracy: {without_rad['accuracy']:.3f}")
        print(f"  Precision: {without_rad['precision']:.3f}")
        print(f"  Recall: {without_rad['recall']:.3f}")
        print(f"  F1-Score: {without_rad['f1_score']:.3f}")
        print(f"  Specificity: {without_rad['specificity']:.3f}")
        if without_rad.get('auc_roc'):
            print(f"  AUC-ROC: {without_rad['auc_roc']:.3f}")
        print(f"  Actual Surgery Rate: {without_rad['actual_surgery_rate']:.1%}")
        print(f"  Predicted Surgery Rate: {without_rad['predicted_surgery_rate']:.1%}")

    # Performance comparison
    if metrics_dict['with_radiology'] and metrics_dict['without_radiology']:
        print(f"\nPERFORMANCE COMPARISON (With - Without Radiology):")
        with_rad = metrics_dict['with_radiology']
        without_rad = metrics_dict['without_radiology']

        print(f"  Accuracy Difference: {with_rad['accuracy'] - without_rad['accuracy']:+.3f}")
        print(f"  Precision Difference: {with_rad['precision'] - without_rad['precision']:+.3f}")
        print(f"  Recall Difference: {with_rad['recall'] - without_rad['recall']:+.3f}")
        print(f"  F1-Score Difference: {with_rad['f1_score'] - without_rad['f1_score']:+.3f}")
        print(f"  Specificity Difference: {with_rad['specificity'] - without_rad['specificity']:+.3f}")

        # Statistical significance indicators
        print(f"\nINSIGHTS:")
        if with_rad['total_cases'] < 30 or without_rad['total_cases'] < 30:
            print(f"  - Small sample size may limit statistical significance")

        accuracy_diff = with_rad['accuracy'] - without_rad['accuracy']
        if abs(accuracy_diff) > 0.1:
            direction = "positively" if accuracy_diff > 0 else "negatively"
            print(f"  - Radiology reports appear to {direction} impact LLM performance")
        elif abs(accuracy_diff) < 0.05:
            print(f"  - Minimal difference suggests limited impact of radiology reports")

        if with_rad['recall'] > without_rad['recall'] + 0.05:
            print(f"  - LLM shows better sensitivity (recall) when radiology is available")
        elif without_rad['recall'] > with_rad['recall'] + 0.05:
            print(f"  - LLM shows better sensitivity (recall) without radiology reports")

        if with_rad['specificity'] > without_rad['specificity'] + 0.05:
            print(f"  - LLM shows better specificity when radiology is available")
        elif without_rad['specificity'] > with_rad['specificity'] + 0.05:
            print(f"  - LLM shows better specificity without radiology reports")

    print("\n" + "="*80)