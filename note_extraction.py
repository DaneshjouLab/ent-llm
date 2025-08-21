# Patient Note Processing 
import re
from copy import deepcopy
import pandas as pd
from datetime import datetime
import signal
from contextlib import contextmanager

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

# Define "strong" and "weak" surgical planning phrases
STRONG_SURGICAL_PHRASES = [
    r'surgical\s+intervention(?:\s+for\s+\w+\s+sinusitis)?',
    r'surgical\s+treatment(?:\s+of\s+\w+\s+sinusitis)?',
    r'proceed\s+with\s+surgical\s+intervention(?:\s+for\s+\w+\s+sinusitis)?',
    r'surgical\s+management(?:\s+of\s+(?:\w+\s+)*(?:sinusitis|crs))?',
    r'plan\s+for(?:\s+\w+)*\s+sinus\s+surgery',
    r'scheduled?\s+for(?:\s+\w+)*\s+sinus\s+surgery',
    r'candidate\s+for(?:\s+\w+)*\s+(?:nasal|sinus)\s+surgery',
    r'patient\s+(?:was\s+)?(?:agreed?|elected|opted)(?:\s+to\s+proceed)?(?:\s+with)?(?:\s+\w+)*\s+sinus\s+surgery',
    r'(?:plan|proceed|scheduled?|recommended?|candidate).{0,50}\b(?:FESS|ESS)\b',
    r'\b(?:FESS|ESS)\b\s+(?:is\s+)?(?:recommended|planned|scheduled|indicated)',
    r'(?:considering?|planning\s+for)\s+(?:FESS|ESS)',
]

WEAK_SURGICAL_PHRASES = [
    r'surgical\s+planning',
    r'surgical.{0,20}(?:planning|plan|discussion)',
    r'^(?:assessment\s+and\s+plan|plan):',  # anchored to start of chunk

    # Discussion/decision-making 
    r'surgery.{0,20}(?:discussed?|discussion)',
    r'(?:sinus\s+)?surgery\s+was\s+discussed',
    r'consider\s+(?:endoscopic\s+)?surgery',
    r'patient\s+agrees?\s+(?:with\s+(?:the\s+)?plan)',
    r'we\s+(?:have\s+)?discussed\s+(?:sinus\s+)?surgery',
    r'consented?\s+(?:to|for)\s+(?:sinus\s+)?surgery',
    r'referred\s+to\s+ENT\s+for\s+(?:evaluation\s+and\s+)?surgery',

    # Abbreviations and procedure types
    r'\bESS\b',
    r'\bFESS\b',
    r'\bSEPT\b',
    r'\bESS/FESS\b',
    r'endoscopic\s+sinus\s+surgery',
    r'functional\s+endoscopic\s+sinus\s+surgery',
    r'\bseptoplasty\b',
    r'\bturbinate\s+reduction\b',
    r'\bturbinectomy\b',
    r'\bballoon\s*sinuplasty\b',
    r'\bpolypectomy\b',
]

# Compile regex patterns
strong_patterns = [re.compile(p, re.IGNORECASE) for p in STRONG_SURGICAL_PHRASES]
weak_patterns = [re.compile(p, re.IGNORECASE) for p in WEAK_SURGICAL_PHRASES]
all_patterns = strong_patterns + weak_patterns

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