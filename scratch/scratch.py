#### For when there was an issue with functions not being pulled in

# import importlib
# import sys

# modules_to_reload = [
#     'raw_data_parsing',
#     'note_extraction',
#     'LLM_input',
#     'LLM_analysis',
#     'config',
# ]

# for module in modules_to_reload:
#     if module in sys.modules:
#         importlib.reload(sys.modules[module])
#     else:SC
#       import module


# just the libraries 

import pandas as pd
import numpy as np
import os
import openai
import re
import json
import sys
import importlib
import time
from typing import List, Dict, Iterator, Tuple
from google.cloud import bigquery
import gc


# # Formatting for LLM Input
# import pandas as pd
# from typing import Dict, Any, Union, List

# def format_medical_data(progress_note: Union[Dict, None], radiology_reports: List[Dict]) -> Dict[str, Any]:
#     """Format medical data from note dictionary and reports list into readable text."""

#     # Format single progress note (not a list)
#     progress_text = ""
#     if progress_note and isinstance(progress_note, dict):
#         date = progress_note.get('date', 'Unknown date')
#         note_type = progress_note.get('type', 'Unknown type')
#         title = progress_note.get('title', 'No title')
#         text = progress_note.get('text', 'No text')

#         # Add original date info if it was recursively censored
#         original_date = progress_note.get('original_date')
#         date_info = f"{date}"
#         if original_date and original_date != date:
#             date_info += f" (originally from {original_date})"

#         progress_text = f"Date: {date_info}, Type: {note_type}, Title: {title}\nContent: {text}"
#     else:
#         progress_text = "No progress notes available."

#     # Format radiology reports
#     radiology_text = ""
#     has_radiology = bool(radiology_reports and len(radiology_reports) > 0)

#     if radiology_reports and isinstance(radiology_reports, list):
#         for report in radiology_reports:
#             if isinstance(report, dict):
#                 date = report.get('date', 'Unknown date')
#                 report_type = report.get('type', 'Unknown type')
#                 title = report.get('title', 'No title')
#                 text = report.get('text', 'No text')
#                 radiology_text += f"Date: {date}, Type: {report_type}, Title: {title}\nContent: {text}\n\n"

#     if not radiology_text:
#         radiology_text = "No radiology reports available."

#     return {
#         'progress_text': progress_text.strip(),
#         'radiology_text': radiology_text.strip(),
#         'has_radiology_report': has_radiology
#     }

# def create_llm_dataframe(processed_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create a clean 3-column DataFrame for LLM queries.

#     Args:
#         processed_df: DataFrame with 'radiology_reports' and 'last_progress_note_censored' columns

#     Returns:
#         DataFrame with columns: llm_caseID, formatted_radiology_text, formatted_progress_text
#     """
#     # Initialize columns for formatted text
#     formatted_radiology = []
#     formatted_progress = []

#     # Process each row
#     for idx, row in processed_df.iterrows():
#         # Get the single progress note (not a list)
#         progress_note = row.get('last_progress_note_censored')
#         radiology_reports = row.get('radiology_reports', [])

#         # Ensure radiology_reports is a list
#         if not isinstance(radiology_reports, list):
#             radiology_reports = []

#         # Format the medical data
#         formatted_data = format_medical_data(
#             progress_note=progress_note,  # Single note, not list
#             radiology_reports=radiology_reports
#         )

#         formatted_radiology.append(formatted_data['radiology_text'])
#         formatted_progress.append(formatted_data['progress_text'])

#     # Build the new DataFrame
#     llm_df = pd.DataFrame({
#         'llm_caseID': processed_df['llm_caseID'].values,  # Use original case IDs!
#         'formatted_radiology_text': formatted_radiology,
#         'formatted_progress_text': formatted_progress
#     })

#     return llm_df

# def build_patient_df(ent_df, radiology_df, surgery_df):
#     """Builds a patient-level DataFrame with ENT notes, radiology reports, and surgery data."""
#     import pandas as pd
#     # Normalize and clean dates
#     ent_df['date'] = pd.to_datetime(ent_df['date'], errors='coerce')
#     radiology_df['date'] = pd.to_datetime(radiology_df['date'], errors='coerce')

#     # Ensure columns are strings
#     for df in [ent_df, radiology_df]:
#         df['text'] = df['text'].astype(str)
#         df['type'] = df['type'].astype(str)
#         df['title'] = df['title'].astype(str)

#     # Group ENT notes by patient
#     ent_grouped = ent_df.groupby('patient_id').apply(
#         lambda x: sorted(
#             [
#                 {
#                     'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
#                     'type': typ,
#                     'title': ttl,
#                     'text': t
#                 }
#                 for d, t, typ, ttl in zip(x['date'], x['text'], x['type'], x['title'])
#             ],
#             key=lambda note: note['date'] if note['date'] else ''
#         ),
#         include_groups=False
#     ).reset_index(name='ent_notes')

#     # Merge with surgery data first to get surgery dates
#     patient_data = pd.merge(ent_grouped, surgery_df, on='patient_id', how='left')
#     patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)

#     # Group radiology reports by patient, filtering by surgery date
#     def filter_radiology_by_surgery(group):
#         patient_id = group.name

#         # Check if this patient exists in patient_data (i.e., has ENT notes)
#         patient_match = patient_data[patient_data['patient_id'] == patient_id]

#         if len(patient_match) == 0:
#             # Patient has radiology but no ENT notes - check surgery data directly
#             surgery_match = surgery_df[surgery_df['patient_id'] == patient_id]
#             if len(surgery_match) > 0:
#                 surgery_date = surgery_match['first_surgery_date'].iloc[0]
#             else:
#                 surgery_date = pd.NaT  # No surgery data
#         else:
#             # Patient has ENT notes, get surgery date from patient_data
#             surgery_date = patient_match['first_surgery_date'].iloc[0]

#         # Filter radiology reports
#         if pd.notna(surgery_date):
#             # Only include reports before surgery date
#             group = group[group['date'] < surgery_date]
#         # If no surgery date (NaT), keep all reports

#         return sorted(
#             [
#                 {
#                     'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
#                     'type': typ,
#                     'title': ttl,
#                     'text': t
#                 }
#                 for d, t, typ, ttl in zip(group['date'], group['text'], group['type'], group['title'])
#             ],
#             key=lambda note: note['date'] if note['date'] else ''
#         )

#     rad_grouped = radiology_df.groupby('patient_id').apply(filter_radiology_by_surgery, include_groups=False).reset_index(name='radiology_reports')

#     # Merge radiology data - use outer join to include patients with radiology but no ENT notes
#     patient_data = pd.merge(patient_data, rad_grouped, on='patient_id', how='outer')

#     # Handle missing values for patients who have radiology but no ENT notes
#     patient_data['ent_notes'] = patient_data['ent_notes'].apply(lambda x: x if isinstance(x, list) else [])
#     patient_data['radiology_reports'] = patient_data['radiology_reports'].apply(lambda x: x if isinstance(x, list) else [])
#     patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)

#     return patient_data



# import gc
# import pandas as pd
# from tqdm import tqdm

# def run_batch_llm_analysis(llm_data: pd.DataFrame,
#                           batch_num: int,
#                           api_key: str,
#                           sub_batch_size: int = 50) -> pd.DataFrame:
#     """Run LLM analysis on a batch with memory management and progress tracking."""

#     if llm_data.empty:
#         print(f"Batch {batch_num} is empty, skipping...")
#         return pd.DataFrame()

#     print(f"\n=== Starting LLM Analysis for Batch {batch_num} ===")
#     print(f"Total cases in batch: {len(llm_data)}")
#     print(f"Sub-batch size: {sub_batch_size}")

#     try:
#         results = []
#         num_sub_batches = (len(llm_data) + sub_batch_size - 1) // sub_batch_size

#         # Process in smaller sub-batches with progress bar
#         for i in tqdm(range(0, len(llm_data), sub_batch_size),
#                      desc=f"Batch {batch_num} sub-batches"):

#             sub_batch_num = i // sub_batch_size + 1
#             sub_batch = llm_data.iloc[i:i + sub_batch_size].copy()

#             print(f"\nProcessing sub-batch {sub_batch_num}/{num_sub_batches} "
#                   f"(cases {i+1}-{min(i+sub_batch_size, len(llm_data))})")

#             # Check case ID range for debugging
#             case_ids = sub_batch['llm_caseID'].tolist()
#             print(f"Case ID range: {min(case_ids)}-{max(case_ids)}")

#             try:
#                 # Use your existing LLM analysis function
#                 sub_results = run_llm_analysis_4(sub_batch, api_key)

#                 if not sub_results.empty:
#                     results.append(sub_results)
#                     print(f"✓ Sub-batch {sub_batch_num} completed: {len(sub_results)} cases processed")
#                 else:
#                     print(f"✗ Sub-batch {sub_batch_num} returned empty results")

#             except Exception as e:
#                 print(f"✗ Error in sub-batch {sub_batch_num}: {e}")
#                 # Continue with next sub-batch instead of failing entirely
#                 continue

#             # Clean up memory between sub-batches
#             del sub_batch
#             if 'sub_results' in locals():
#                 del sub_results
#             gc.collect()

#         # Combine all results
#         if results:
#             print(f"\nCombining {len(results)} sub-batch results...")
#             batch_results = pd.concat(results, ignore_index=True)

#             # Verify case IDs are preserved
#             original_case_ids = set(llm_data['llm_caseID'])
#             result_case_ids = set(batch_results['llm_caseID'])
#             missing_cases = original_case_ids - result_case_ids

#             if missing_cases:
#                 print(f"WARNING: {len(missing_cases)} cases missing from results")

#             print(f"✓ Batch {batch_num} completed: {len(batch_results)} total cases")
#             return batch_results
#         else:
#             print(f"✗ No results from batch {batch_num}")
#             return pd.DataFrame()

#     except Exception as e:
#         print(f"✗ Critical error in batch {batch_num}: {e}")
#         import traceback
#         traceback.print_exc()
#         return pd.DataFrame()


# def process_all_llm_data(final_llm_df: pd.DataFrame,
#                         api_key: str,
#                         batch_size: int = 500,
#                         sub_batch_size: int = 50) -> pd.DataFrame:
#     """
#     Process the entire final_llm_df in batches for memory efficiency.

#     Args:
#         final_llm_df: Your complete LLM dataframe
#         api_key: OpenAI API key
#         batch_size: Size of main batches (for memory management)
#         sub_batch_size: Size of LLM processing sub-batches

#     Returns:
#         Complete results dataframe
#     """

#     print(f"=== Starting Complete LLM Analysis ===")
#     print(f"Total cases to process: {len(final_llm_df)}")
#     print(f"Batch size: {batch_size}, Sub-batch size: {sub_batch_size}")

#     all_results = []
#     num_batches = (len(final_llm_df) + batch_size - 1) // batch_size

#     for i in range(0, len(final_llm_df), batch_size):
#         batch_num = i // batch_size + 1
#         batch_data = final_llm_df.iloc[i:i + batch_size].copy()

#         print(f"\n{'='*60}")
#         print(f"PROCESSING MAIN BATCH {batch_num}/{num_batches}")
#         print(f"{'='*60}")

#         # Process this batch
#         batch_results = run_batch_llm_analysis(
#             batch_data, batch_num, api_key, sub_batch_size
#         )

#         if not batch_results.empty:
#             all_results.append(batch_results)

#             # Save intermediate results
#             print(f"Saving intermediate results for batch {batch_num}...")
#             batch_results.to_pickle(f'llm_results_batch_{batch_num}.pkl')

#         # Clean up
#         del batch_data
#         if 'batch_results' in locals():
#             del batch_results
#         gc.collect()

#         print(f"Batch {batch_num} memory cleanup completed")

#     # Combine all results
#     if all_results:
#         print(f"\n=== Combining all {len(all_results)} batches ===")
#         final_results = pd.concat(all_results, ignore_index=True)

#         # Final verification
#         print(f"Final results: {len(final_results)} cases")
#         print(f"Success rate: {final_results['decision'].notna().sum()}/{len(final_results)} "
#               f"({final_results['decision'].notna().mean()*100:.1f}%)")

#         return final_results
#     else:
#         print("No results obtained from any batch")
#         return pd.DataFrame()


# def quick_status_check(final_llm_df: pd.DataFrame, save_directory: str = '.'):
#     """Quick check of what's completed vs remaining."""
#     completed_ids = get_completed_case_ids(save_directory)
#     total_cases = len(final_llm_df)
#     completed_cases = len(completed_ids)
#     remaining_cases = total_cases - completed_cases

#     print(f"\n=== STATUS CHECK ===")
#     print(f"Total cases: {total_cases}")
#     print(f"Completed: {completed_cases} ({completed_cases/total_cases*100:.1f}%)")
#     print(f"Remaining: {remaining_cases} ({remaining_cases/total_cases*100:.1f}%)")

#     if completed_cases > 0:
#         # Show some sample completed case IDs
#         sample_completed = sorted(list(completed_ids))[:10]
#         print(f"Sample completed case IDs: {sample_completed}")




# def process_llm_cases(llm_df: pd.DataFrame, api_key: str, delay_seconds: float = 0.2) -> pd.DataFrame:
#     """
#     Process a clean LLM DataFrame through OpenAI API.

#     Args:
#         llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
#         api_key: OpenAI API key (hardcoded)
#         delay_seconds: Delay between API calls to avoid rate limiting

#     Returns:
#         DataFrame with additional columns: 'decision', 'confidence', 'reasoning', 'api_response'
#     """

#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     # Initialize OpenAI client
#     client = openai.OpenAI(api_key=api_key)
#     logging.info("OpenAI client initialized successfully")

#     # Create a copy of the dataframe
#     result_df = llm_df.copy()

#     # Initialize new columns
#     result_df['decision'] = None
#     result_df['confidence'] = None
#     result_df['reasoning'] = None
#     result_df['api_response'] = None  # Store raw response for debugging

#     total_rows = len(result_df)
#     logging.info(f"Processing {total_rows} cases...")
#     start_time = time.time()

#     for idx, row in tqdm(result_df.iterrows(), total=total_rows, desc="Processing cases"):
#         try:
#             case_id = row['llm_caseID']
#             logging.info(f"Processing case {idx + 1}/{total_rows}: Case ID {case_id}")

#             # Generate prompt using the formatted text columns
#             prompt = generate_prompt(
#                 case_id=case_id,
#                 progress_text=row['formatted_progress_text'],
#                 radiology_text=row['formatted_radiology_text']
#             )

#             # Query OpenAI
#             response = query_openai(prompt, client)
#             result_df.at[idx, 'api_response'] = response

#             if response:
#                 # Parse response
#                 parsed = parse_llm_response(response)
#                 result_df.at[idx, 'decision'] = parsed['decision']
#                 result_df.at[idx, 'confidence'] = parsed['confidence']
#                 result_df.at[idx, 'reasoning'] = parsed['reasoning']

#                 logging.info(f"✓ Case {case_id}: {parsed['decision']} (confidence: {parsed['confidence']})")
#             else:
#                 logging.warning(f"✗ No response for case {case_id}")

#             # Add delay to avoid rate limiting
#             if delay_seconds > 0:
#                 time.sleep(delay_seconds)

#             # Progress updates every 100 cases
#             if (idx + 1) % 100 == 0:
#                 elapsed = time.time() - start_time
#                 rate = (idx + 1) / elapsed * 60  # cases per minute
#                 remaining = total_rows - (idx + 1)
#                 eta_minutes = remaining / (rate / 60) if rate > 0 else 0
#                 print(f"Processed {idx + 1}/{total_rows} cases. Rate: {rate:.1f}/min, ETA: {eta_minutes:.1f}min")


#         except Exception as e:
#             logging.error(f"Error processing case {case_id}: {e}")
#             result_df.at[idx, 'reasoning'] = f"Error: {str(e)}"

#     elapsed = time.time() - start_time
#     final_rate = total_rows / elapsed * 60
#     logging.info(f"Processing complete! {total_rows} cases in {elapsed:.1f}s ({final_rate:.1f} cases/min)")
#     return result_df


# def run_llm_analysis(llm_df, api_key):
#     """
#     Main function to run the LLM analysis on your DataFrame.

#     Args:
#         llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
#         api_key: Your OpenAI API key

#     Returns:
#         DataFrame with LLM analysis results
#     """

#     print(f"Starting analysis of {len(llm_df)} cases...")
#     print(f"DataFrame columns: {list(llm_df.columns)}")

#     # Process the cases
#     results_df = process_llm_cases(llm_df, api_key, delay_seconds=0.2)

#     # Show summary
#     total_cases = len(results_df)
#     successful_cases = results_df['decision'].notna().sum()
#     yes_decisions = (results_df['decision'] == 'Yes').sum()
#     no_decisions = (results_df['decision'] == 'No').sum()

#     print(f"\n=== Analysis Complete ===")
#     print(f"Total cases processed: {total_cases}")
#     print(f"Successful responses: {successful_cases}")
#     print(f"Surgery recommended: {yes_decisions}")
#     print(f"Surgery not recommended: {no_decisions}")
#     print(f"Failed responses: {total_cases - successful_cases}")

#     return results_df