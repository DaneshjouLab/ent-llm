# Formatting for LLM Input
import pandas as pd
from typing import Dict, Any, Union, List

def format_medical_data(progress_note: Union[Dict, None], radiology_reports: List[Dict]) -> Dict[str, Any]:
    """Format medical data from note dictionary and reports list into readable text."""
   
    # Format single progress note (not a list)
    progress_text = ""
    if progress_note and isinstance(progress_note, dict):
        date = progress_note.get('date', 'Unknown date')
        note_type = progress_note.get('type', 'Unknown type')
        title = progress_note.get('title', 'No title')
        text = progress_note.get('text', 'No text')

        # Add original date info if it was recursively censored
        original_date = progress_note.get('original_date')
        date_info = f"{date}"
        if original_date and original_date != date:
            date_info += f" (originally from {original_date})"

        progress_text = f"Date: {date_info}, Type: {note_type}, Title: {title}\nContent: {text}"
    else:
        progress_text = "No progress notes available."

    # Format radiology reports
    radiology_text = ""
    has_radiology = bool(radiology_reports and len(radiology_reports) > 0)

    if radiology_reports and isinstance(radiology_reports, list):
        for report in radiology_reports:
            if isinstance(report, dict):
                date = report.get('date', 'Unknown date')
                report_type = report.get('type', 'Unknown type')
                title = report.get('title', 'No title')
                text = report.get('text', 'No text')
                radiology_text += f"Date: {date}, Type: {report_type}, Title: {title}\nContent: {text}\n\n"

    if not radiology_text:
        radiology_text = "No radiology reports available."

    return {
        'progress_text': progress_text.strip(),
        'radiology_text': radiology_text.strip(),
        'has_radiology_report': has_radiology
    }

def create_llm_dataframe(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean 3-column DataFrame for LLM queries.

    Args:
        processed_df: DataFrame with 'radiology_reports' and 'last_progress_note_censored' columns

    Returns:
        DataFrame with columns: llm_caseID, formatted_radiology_text, formatted_progress_text
    """
    # Initialize columns for formatted text
    formatted_radiology = []
    formatted_progress = []

    # Process each row
    for idx, row in processed_df.iterrows():
        # Get the single progress note (not a list)
        progress_note = row.get('last_progress_note_censored')
        radiology_reports = row.get('radiology_reports', [])

        # Ensure radiology_reports is a list
        if not isinstance(radiology_reports, list):
            radiology_reports = []

        # Format the medical data
        formatted_data = format_medical_data(
            progress_note=progress_note,  # Single note, not list
            radiology_reports=radiology_reports
        )

        formatted_radiology.append(formatted_data['radiology_text'])
        formatted_progress.append(formatted_data['progress_text'])

    # Build the new DataFrame
    llm_df = pd.DataFrame({
        'llm_caseID': processed_df['llm_caseID'].values,  # Use original case IDs!
        'formatted_radiology_text': formatted_radiology,
        'formatted_progress_text': formatted_progress
    })

    return llm_df