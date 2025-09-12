# Helper functions for processing patient data

# ENT Note and Procedure Extraction
def extract_ent_notes_ref(clinical_notes_df, note_types, note_titles):
    """Extract relevant ENT notes."""

    df = clinical_notes_df.copy()

    # Normalize string fields
    df['author'] = df['author'].astype(str).str.lower()
    df['text'] = df['text'].astype(str).str.lower()
    df['type'] = df['type'].astype(str)
    df['title'] = df['title'].astype(str)

    # Match full word "ent" or "otolaryngology"
    keyword = r'\b(?:ent|otolaryngology)\b'

    ent_filter = (
        df['author'].str.contains(keyword, na=False, regex=True) |
        df['linked_author'].str.contains(keyword, na=False, regex=True)
    )

    key_filter = df['type'].isin(note_types) | df['title'].isin(note_titles)

    ent_df = df[ent_filter & key_filter].copy()

    return ent_df

def extract_radiology_reports(radiology_df, types, titles):
    """Extract relevant radiology reports."""
    df = radiology_df.copy()

    # Normalize string fields
    df['type'] = df['type'].astype(str)
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str).str.lower()

    # Filter by type or title match
    type_filter = df['type'].isin(types)
    title_filter = df['title'].isin(titles)

    filtered_df = df[type_filter & title_filter].copy()
    
    return filtered_df

def procedures_df(procedures_df, surgery_cpt_codes, endoscopy_cpt_codes):
    """Returns a dataframe with surgery/endoscopy flags and their earliest CPT dates."""
    import pandas as pd

    # Ensure proper data types
    procedures_df['code'] = procedures_df['code'].astype(str)
    procedures_df['code_type'] = procedures_df['code_type'].astype(str)

    # Results dictionary to collect per-patient results
    results = {}

    for patient_id in procedures_df['patient_id'].unique():
        patient_procs = procedures_df[procedures_df['patient_id'] == patient_id]
        patient_cpt = patient_procs[patient_procs['code_type'].str.upper() == 'CPT']

        had_surgery = False
        had_endoscopy = False
        surgery_dates = []
        endoscopy_dates = []

        for _, row in patient_cpt.iterrows():
            code = row['code']
            date = row.get('date')
            parsed_date = pd.to_datetime(date) if pd.notna(date) else None

            if code in surgery_cpt_codes:
                had_surgery = True
                if parsed_date:
                    surgery_dates.append(parsed_date)

            if code in endoscopy_cpt_codes:
                had_endoscopy = True
                if parsed_date:
                    endoscopy_dates.append(parsed_date)

        first_surgery_date = min(surgery_dates) if surgery_dates else pd.NaT
        first_endoscopy_date = min(endoscopy_dates) if endoscopy_dates else pd.NaT

        results[patient_id] = {
            'had_surgery': had_surgery,
            'had_endoscopy': had_endoscopy,
            'first_surgery_date': first_surgery_date,
            'first_endoscopy_date': first_endoscopy_date
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'patient_id'}, inplace=True)

    return results_df


def build_patient_df(ent_df, radiology_df, surgery_df):
    """Builds a patient-level DataFrame with ENT notes, radiology reports, and surgery data."""
    import pandas as pd
    # Normalize and clean dates
    ent_df['date'] = pd.to_datetime(ent_df['date'], errors='coerce')
    radiology_df['date'] = pd.to_datetime(radiology_df['date'], errors='coerce')

    # Ensure columns are strings
    for df in [ent_df, radiology_df]:
        df['text'] = df['text'].astype(str)
        df['type'] = df['type'].astype(str)
        df['title'] = df['title'].astype(str)

    # Group ENT notes by patient
    ent_grouped = ent_df.groupby('patient_id').apply(
        lambda x: sorted(
            [
                {
                    'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
                    'type': typ,
                    'title': ttl,
                    'text': t
                }
                for d, t, typ, ttl in zip(x['date'], x['text'], x['type'], x['title'])
            ],
            key=lambda note: note['date'] if note['date'] else ''
        ),
        include_groups=False
    ).reset_index(name='ent_notes')

    # Merge with surgery data first to get surgery dates
    patient_data = pd.merge(ent_grouped, surgery_df, on='patient_id', how='left')
    patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)

    # Group radiology reports by patient, filtering by surgery date
    def filter_radiology_by_surgery(group):
        patient_id = group.name
        # Get surgery date for this patient
        surgery_date = patient_data[patient_data['patient_id'] == patient_id]['first_surgery_date'].iloc[0]

        # Filter radiology reports
        if pd.notna(surgery_date):
            # Only include reports before surgery date
            group = group[group['date'] < surgery_date]
        # If no surgery date (NaT), keep all reports

        return sorted(
            [
                {
                    'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
                    'type': typ,
                    'title': ttl,
                    'text': t
                }
                for d, t, typ, ttl in zip(group['date'], group['text'], group['type'], group['title'])
            ],
            key=lambda note: note['date'] if note['date'] else ''
        )

    rad_grouped = radiology_df.groupby('patient_id').apply(filter_radiology_by_surgery, include_groups=False).reset_index(name='radiology_reports')

    # Merge radiology data
    patient_data = pd.merge(patient_data, rad_grouped, on='patient_id', how='left')

    # Handle missing values
    patient_data['ent_notes'] = patient_data['ent_notes'].apply(lambda x: x if isinstance(x, list) else [])
    patient_data['radiology_reports'] = patient_data['radiology_reports'].apply(lambda x: x if isinstance(x, list) else [])

    return patient_data

def add_last_progress_note(merged_df):
    import pandas as pd

    """
    Adds a column 'last_progress_note' to the merged_df:
    - For patients with surgery: returns last 'Progress Notes' before surgery.
    - For patients without surgery: returns last 'Progress Notes' overall.
    """
    def get_last_progress_note(row):
        ent_notes = row.get('ent_notes', [])
        had_surgery = row.get('had_surgery', False)
        surgery_date = pd.to_datetime(row.get('first_surgery_date'), errors='coerce')

        if not isinstance(ent_notes, list):
            return None

        # Filter to only 'Progress Notes'
        progress_notes = [
            note for note in ent_notes
            if isinstance(note, dict)
            and note.get('title') == 'Progress Notes'
            and 'date' in note
        ]

        # Sort by date
        sorted_notes = sorted(
            progress_notes,
            key=lambda note: pd.to_datetime(note['date'], errors='coerce')
        )

        if had_surgery and pd.notna(surgery_date):
            # Return last progress note before surgery
            notes_before_surgery = [
                note for note in sorted_notes
                if pd.to_datetime(note['date'], errors='coerce') < surgery_date
            ]
            return notes_before_surgery[-1] if notes_before_surgery else None
        else:
            # Return last progress note overall
            return sorted_notes[-1] if sorted_notes else None

    # Apply row-wise and return modified df
    merged_df['last_progress_note'] = merged_df.apply(get_last_progress_note, axis=1)
    return merged_df

# Patient Note Processing 
import re
from copy import deepcopy
import pandas as pd
from datetime import datetime
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def safe_regex_search(pattern, text, timeout_seconds=5):
    """Safely search with regex pattern with timeout protection."""
    try:
        with timeout(timeout_seconds):
            return pattern.search(text)
    except TimeoutError:
        print(f"Regex search timed out for pattern: {pattern.pattern[:50]}...")
        return None
    except Exception as e:
        print(f"Regex search error: {str(e)}")
        return None


def remove_surg_plan(text, timeout_seconds=10):
    """
    Removes surgical planning text starting at the best match:
    - Highest strong match (earliest in text), if any
    - Else, lowest weak match (latest in text)

    Includes timeout protection and error handling for problematic texts.

    Returns:
    - cleaned_text: note with surgical plan removed
    - matched_pattern: regex pattern used
    - matched_text: exact matched string
    """
    if not isinstance(text, str):
        return text, None, None

    # Handle short texts by removing entirely
    if len(text.strip()) < 500:
        print(f"Text too short ({len(text)} chars), truncating for safety")
        return "", None, None

    # Check for extremely long texts that might cause issues
    if len(text) > 100000:  # 100KB limit
        print(f"Text too long ({len(text)} chars), truncating for safety")
        text = text[:100000]

    try:
        with timeout(timeout_seconds):
            parts = re.split(r'\s{5,}', text)
            delimiter = ' ' * 5

            best_strong = {
                'index': None,
                'pattern': None,
                'matched_text': None,
            }

            best_weak = {
                'index': None,
                'pattern': None,
                'matched_text': None,
            }

            for i in reversed(range(len(parts))):
                # Skip extremely long parts that might cause regex issues
                if len(parts[i]) > 50000:
                    print(f"Skipping part {i} (too long: {len(parts[i])} chars)")
                    continue

                for pattern in all_patterns:
                    m = safe_regex_search(pattern, parts[i], timeout_seconds=2)
                    if m:
                        if pattern in strong_patterns:
                            best_strong.update({
                                'index': i,
                                'pattern': pattern.pattern,
                                'matched_text': m.group(0),
                            })
                            # Keep updating, we want the HIGHEST (earliest) strong match
                        else:
                            if best_weak['index'] is None or i > best_weak['index']:
                                best_weak.update({
                                    'index': i,
                                    'pattern': pattern.pattern,
                                    'matched_text': m.group(0),
                                })
                        break  # Only first match per chunk

            # Use best strong match if it exists
            if best_strong['index'] is not None:
                cleaned_text = delimiter.join(parts[:best_strong['index']])
                return cleaned_text, best_strong['pattern'], best_strong['matched_text']
            elif best_weak['index'] is not None:
                cleaned_text = delimiter.join(parts[:best_weak['index']])
                return cleaned_text, best_weak['pattern'], best_weak['matched_text']
            else:
                return text, None, None

    except TimeoutError:
        print(f"remove_surg_plan timed out after {timeout_seconds} seconds")
        return text, None, None
    except Exception as e:
        print(f"Error in remove_surg_plan: {str(e)}")
        return text, None, None


def add_censored_last_progress_note(merged_df):
    """
    Adds:
      - 'last_progress_note_censored': dict with surgical plan removed (if matched)
      - 'matched_text': the specific text that matched
      - 'removed_surgical_text': the full portion of text that was cut
    """
    merged_df = deepcopy(merged_df)

    def censor_note(note):
        if isinstance(note, dict) and 'text' in note:
            original_text = note['text']
            cleaned_text, match_pattern, match_text = remove_surg_plan(original_text)
            removed_text = original_text[len(cleaned_text):].lstrip() if cleaned_text != original_text else None
            note_copy = note.copy()
            note_copy['text'] = cleaned_text
            return note_copy, match_text, removed_text
        return note, None, None

    results = merged_df['last_progress_note'].apply(censor_note)

    merged_df['last_progress_note_censored'] = results.apply(lambda x: x[0])
    merged_df['matched_text'] = results.apply(lambda x: x[1])
    merged_df['censored_text'] = results.apply(lambda x: x[2])

    return merged_df


def get_previous_note(patient_notes_list, current_date, max_attempts=10):
    """
    Find the most recent 'Progress Notes' before current_date from a list of patient notes.
    Uses the same filtering logic as add_last_progress_note function.

    Args:
        patient_notes_list: List of note dictionaries with 'date', 'type', 'title', 'text'
        current_date: Current note date to find previous note before
        max_attempts: Maximum number of previous notes to try

    Returns:
        dict: Previous note with 'text' and 'date' keys, or None if not found
    """
    if not isinstance(patient_notes_list, list):
        return None

    # Filter to only 'Progress Notes'
    progress_notes = [
        note for note in patient_notes_list
        if isinstance(note, dict)
        and note.get('title') == 'Progress Notes'
        and 'date' in note
    ]

    if not progress_notes:
        return None

    # Sort by date
    sorted_notes = sorted(
        progress_notes,
        key=lambda note: pd.to_datetime(note['date'], errors='coerce')
    )

    # Filter notes before current date
    current_dt = pd.to_datetime(current_date, errors='coerce')
    if pd.isna(current_dt):
        return None

    notes_before_current = [
        note for note in sorted_notes
        if pd.to_datetime(note['date'], errors='coerce') < current_dt
    ]

    if not notes_before_current:
        return None

    # Limit to max_attempts to prevent infinite loops
    if len(notes_before_current) > max_attempts:
        notes_before_current = notes_before_current[-max_attempts:]

    # Return the most recent note (last in sorted list)
    return notes_before_current[-1] if notes_before_current else None


def is_text_empty_or_whitespace(text):
    """Check if text is empty, None, or only whitespace."""
    if text is None:
        return True
    if isinstance(text, str):
        return len(text.strip()) == 0
    return False


def recursive_censor_notes(merged_df, max_iterations=5, max_patients_to_skip=100, patient_timeout_seconds=60):
    """
    Recursively censor progress notes, falling back to previous notes when completely censored.
    FIXED: Now properly updates last_progress_note and re-runs add_censored_last_progress_note

    Args:
        merged_df: DataFrame with patient data including 'ent_notes' and 'last_progress_note_censored'
        max_iterations: Maximum number of recursive attempts per patient
        max_patients_to_skip: Maximum number of patients to skip before stopping
        patient_timeout_seconds: Maximum time to spend on each patient (now just for monitoring)

    Returns:
        tuple: (processed_df, skipped_patient_ids)
    """
    import time

    merged_df = deepcopy(merged_df)
    skipped_patients = []

    # First run the initial censoring if not already done
    if 'last_progress_note_censored' not in merged_df.columns:
        merged_df = add_censored_last_progress_note(merged_df)

    patients_to_process = []

    # Identify patients that need recursive processing
    for idx, row in merged_df.iterrows():
        censored_note = row.get('last_progress_note_censored', {})
        if isinstance(censored_note, dict):
            text = censored_note.get('text', '')
            if is_text_empty_or_whitespace(text):
                patients_to_process.append(idx)

    print(f"Found {len(patients_to_process)} patients requiring recursive censoring")

    # Process each patient that needs recursive censoring
    for patient_idx in patients_to_process:
        if len(skipped_patients) >= max_patients_to_skip:
            print(f"Reached maximum skip limit ({max_patients_to_skip}), stopping processing")
            break

        patient_start_time = time.time()

        try:
            print(f"Processing patient {patient_idx}...")

            # Get patient's ENT notes list and current censored note
            try:
                patient_notes_list = merged_df.loc[patient_idx, 'ent_notes']
                current_note = merged_df.loc[patient_idx, 'last_progress_note_censored']
                original_last_progress_note = merged_df.loc[patient_idx, 'last_progress_note']
                print(f"  - Retrieved patient data successfully")
            except Exception as e:
                print(f"Patient {patient_idx}: Error accessing data - {str(e)}, skipping")
                skipped_patients.append(patient_idx)
                continue

            current_date = current_note.get('date') if isinstance(current_note, dict) else None

            if not isinstance(patient_notes_list, list):
                print(f"Patient {patient_idx}: Invalid ent_notes format (not a list), skipping")
                skipped_patients.append(patient_idx)
                continue

            iteration_count = 0
            found_valid_note = False

            while iteration_count < max_iterations and not found_valid_note:
                iteration_count += 1

                # Check if we've been processing this patient too long
                elapsed = time.time() - patient_start_time
                if elapsed > patient_timeout_seconds:
                    print(f"Patient {patient_idx}: Taking too long ({elapsed:.1f}s), skipping")
                    break

                print(f"  - Iteration {iteration_count}: Looking for previous note...")

                # Get previous note from the patient's notes list
                previous_note = get_previous_note(patient_notes_list, current_date)

                if previous_note is None:
                    print(f"Patient {patient_idx}: No previous note found after {iteration_count} iterations, skipping")
                    break

                print(f"  - Found previous note dated {previous_note.get('date')}, testing censoring...")

                # Test censoring on the previous note
                prev_text = previous_note.get('text', '')

                try:
                    cleaned_text, match_pattern, match_text = remove_surg_plan(prev_text, timeout_seconds=10)
                except Exception as e:
                    print(f"Patient {patient_idx}: Error censoring note in iteration {iteration_count}: {str(e)}")
                    # Skip to next note or give up
                    current_date = previous_note.get('date')
                    continue

                # Check if this note has content after censoring
                if not is_text_empty_or_whitespace(cleaned_text):
                    print(f"  - Previous note has valid content after censoring, updating last_progress_note...")

                    # Update the last_progress_note with the previous note that works
                    # Store original date for reference
                    updated_note = previous_note.copy()
                    updated_note['original_date'] = original_last_progress_note.get('date') if isinstance(original_last_progress_note, dict) else None

                    try:
                        # Update the last_progress_note with the previous note
                        merged_df.at[patient_idx, 'last_progress_note'] = updated_note

                        print(f"  - Re-running censoring on updated note...")

                        # Create a single-row dataframe to re-run censoring
                        single_patient_df = merged_df.loc[[patient_idx]].copy()
                        censored_single_df = add_censored_last_progress_note(single_patient_df)

                        # Update the original dataframe with the newly censored results
                        merged_df.at[patient_idx, 'last_progress_note_censored'] = censored_single_df.loc[patient_idx, 'last_progress_note_censored']
                        merged_df.at[patient_idx, 'matched_text'] = censored_single_df.loc[patient_idx, 'matched_text']
                        merged_df.at[patient_idx, 'censored_text'] = censored_single_df.loc[patient_idx, 'censored_text']

                        print(f"  - Successfully updated with censored previous note")

                    except Exception as assignment_error:
                        print(f"Patient {patient_idx}: Error updating dataframe - {str(assignment_error)}")
                        # Try to continue to next iteration
                        current_date = previous_note.get('date')
                        continue

                    found_valid_note = True
                    print(f"Patient {patient_idx}: Successfully found valid note after {iteration_count} iterations")
                else:
                    # This note was also completely censored, try the next one
                    current_date = previous_note.get('date')
                    print(f"Patient {patient_idx}, iteration {iteration_count}: Previous note also completely censored, trying next previous note")

            if not found_valid_note:
                print(f"Patient {patient_idx}: Exhausted all attempts ({max_iterations} iterations), skipping")
                skipped_patients.append(patient_idx)

        except Exception as e:
            print(f"Error processing patient {patient_idx}: {str(e)}, skipping")
            skipped_patients.append(patient_idx)

    # Remove skipped patients from dataframe
    if skipped_patients:
        print(f"Removing {len(skipped_patients)} skipped patients from dataframe")
        merged_df = merged_df.drop(index=skipped_patients, errors='ignore')

    print(f"Recursive censoring complete. Processed {len(merged_df)} patients, skipped {len(skipped_patients)} patients")

    return merged_df, skipped_patients



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
        'llm_caseID': range(1, len(processed_df) + 1),
        'formatted_radiology_text': formatted_radiology,
        'formatted_progress_text': formatted_progress
    })

    return llm_df


# LLM Prompting
import openai
import pandas as pd
import json
import logging
import time
from typing import Dict, Any

def query_openai(prompt: str, client) -> str:
    """Query GPT-4 for surgical decision based on input prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are an expert otolaryngologist. "
                    "Provide a surgical recommendation in the requested JSON format."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return None

def generate_prompt(case_id: str, progress_text: str, radiology_text: str) -> str:
    """Generates a structured prompt for the LLM."""

    # Check if radiology text is available
    has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
    radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."

    prompt = f"""
    You are an expert otolaryngologist evaluating an ENT case.
    Based ONLY on the information provided below, make a recommendation on surgery.

    --- Case Details ---
    - Case ID: {case_id}
    - Clinical Summary from ENT Notes: {progress_text}
    {radiology_section}

    ---

    Provide your response as a JSON object with three keys:
    1. "decision": Your recommendation, either "Yes" or "No".
    2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
    3. "reasoning": A brief, 2-4 sentence explanation for your decision.

    Return ONLY the JSON object, no additional text.
    """
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract decision, confidence, and reasoning."""
    default_response = {
        'decision': None,
        'confidence': None,
        'reasoning': 'Failed to parse response'
    }

    if not response:
        return default_response

    try:
        # Try to find JSON in the response
        response = response.strip()
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        elif response.startswith('```'):
            response = response.replace('```', '').strip()

        parsed = json.loads(response)

        return {
            'decision': parsed.get('decision'),
            'confidence': parsed.get('confidence'),
            'reasoning': parsed.get('reasoning', 'No reasoning provided')
        }
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        logging.error(f"Response was: {response}")
        return default_response
    except Exception as e:
        logging.error(f"Unexpected error parsing response: {e}")
        return default_response

def process_llm_cases(llm_df: pd.DataFrame, api_key: str, delay_seconds: float = 1.0) -> pd.DataFrame:
    """
    Process a clean LLM DataFrame through OpenAI API.

    Args:
        llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
        api_key: OpenAI API key (hardcoded)
        delay_seconds: Delay between API calls to avoid rate limiting

    Returns:
        DataFrame with additional columns: 'decision', 'confidence', 'reasoning', 'api_response'
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        raise

    # Create a copy of the dataframe
    result_df = llm_df.copy()

    # Initialize new columns
    result_df['decision'] = None
    result_df['confidence'] = None
    result_df['reasoning'] = None
    result_df['api_response'] = None  # Store raw response for debugging

    total_rows = len(result_df)
    logging.info(f"Processing {total_rows} cases...")

    for idx, row in result_df.iterrows():
        try:
            case_id = row['llm_caseID']
            logging.info(f"Processing case {idx + 1}/{total_rows}: Case ID {case_id}")

            # Generate prompt using the formatted text columns
            prompt = generate_prompt(
                case_id=case_id,
                progress_text=row['formatted_progress_text'],
                radiology_text=row['formatted_radiology_text']
            )

            # Query OpenAI
            response = query_openai(prompt, client)
            result_df.at[idx, 'api_response'] = response

            if response:
                # Parse response
                parsed = parse_llm_response(response)
                result_df.at[idx, 'decision'] = parsed['decision']
                result_df.at[idx, 'confidence'] = parsed['confidence']
                result_df.at[idx, 'reasoning'] = parsed['reasoning']

                logging.info(f"✓ Case {case_id}: {parsed['decision']} (confidence: {parsed['confidence']})")
            else:
                logging.warning(f"✗ No response for case {case_id}")

            # Add delay to avoid rate limiting
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        except Exception as e:
            logging.error(f"Error processing case {case_id}: {e}")
            result_df.at[idx, 'reasoning'] = f"Error: {str(e)}"

    logging.info("Processing complete!")
    return result_df

def run_llm_analysis(llm_df, api_key):
    """
    Main function to run the LLM analysis on your DataFrame.

    Args:
        llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
        api_key: Your OpenAI API key

    Returns:
        DataFrame with LLM analysis results
    """

    print(f"Starting analysis of {len(llm_df)} cases...")
    print(f"DataFrame columns: {list(llm_df.columns)}")

    # Process the cases
    results_df = process_llm_cases(llm_df, api_key, delay_seconds=1.0)

    # Show summary
    total_cases = len(results_df)
    successful_cases = results_df['decision'].notna().sum()
    yes_decisions = (results_df['decision'] == 'Yes').sum()
    no_decisions = (results_df['decision'] == 'No').sum()

    print(f"\n=== Analysis Complete ===")
    print(f"Total cases processed: {total_cases}")
    print(f"Successful responses: {successful_cases}")
    print(f"Surgery recommended: {yes_decisions}")
    print(f"Surgery not recommended: {no_decisions}")
    print(f"Failed responses: {total_cases - successful_cases}")

    return results_df

# Evaluation
def evaluate_predictions(eval_df):
    
    # Map LLM decision ('Yes'/'No') to boolean for comparison. Using .lower() is robust.
    eval_df['llm_decision_bool'] = eval_df['llm_decision'].str.lower() == 'yes'

    #  Use 'Had_Surgery' (the correct column name) for comparison
    y_true = eval_df['Had_Surgery']
    y_pred = eval_df['llm_decision_bool']
    
     # Accuracy, Classification Report, and Confusion Matrix
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Surgery', 'Surgery'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # --- Confidence Analysis ---
    eval_df['is_correct'] = (y_true == y_pred)
    
    print("\n--- Confidence Analysis ---")
    print(f"Average confidence on CORRECT predictions: {eval_df[eval_df['is_correct']]['llm_confidence'].mean():.2f}")
    print(f"Average confidence on INCORRECT predictions: {eval_df[~eval_df['is_correct']]['llm_confidence'].mean():.2f}")

    # Break down confidence by prediction outcome (TP, TN, FP, FN)
    tp_mask = (eval_df['is_correct']) & (eval_df['Had_Surgery'])
    tn_mask = (eval_df['is_correct']) & (~eval_df['Had_Surgery'])
    fp_mask = (~eval_df['is_correct']) & (eval_df['llm_decision_bool'])
    fn_mask = (~eval_df['is_correct']) & (~eval_df['llm_decision_bool'])

    print("\n--- Average Confidence by Prediction Outcome ---")
    print(f"True Positives (Correctly recommended surgery):  {eval_df.loc[tp_mask, 'llm_confidence'].mean():.2f}")
    print(f"True Negatives (Correctly recommended no surgery): {eval_df.loc[tn_mask, 'llm_confidence'].mean():.2f}")
    print(f"False Positives (Wrongly recommended surgery):   {eval_df.loc[fp_mask, 'llm_confidence'].mean():.2f}")
    print(f"False Negatives (Missed needed surgery):       {eval_df.loc[fn_mask, 'llm_confidence'].mean():.2f}")