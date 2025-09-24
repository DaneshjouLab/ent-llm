
#openAI
import openai
import pandas as pd
import json
import logging
import time
from typing import Dict, Any
from tqdm import tqdm

def query_openai(prompt: str, client) -> str:
    """Query GPT-4omini for surgical decision based on input prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        # Search JSON in the response
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

def process_llm_cases(llm_df: pd.DataFrame, api_key: str, delay_seconds: float = 0.2) -> pd.DataFrame:
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
    client = openai.OpenAI(api_key=api_key)
    logging.info("OpenAI client initialized successfully")

    # Create a copy of the dataframe
    result_df = llm_df.copy()

    # Initialize new columns
    result_df['decision'] = None
    result_df['confidence'] = None
    result_df['reasoning'] = None
    result_df['api_response'] = None  # Store raw response for debugging

    total_rows = len(result_df)
    logging.info(f"Processing {total_rows} cases...")
    start_time = time.time()

    for idx, row in tqdm(result_df.iterrows(), total=total_rows, desc="Processing cases"):
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

            # Progress updates every 100 cases
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed * 60  # cases per minute
                remaining = total_rows - (idx + 1)
                eta_minutes = remaining / (rate / 60) if rate > 0 else 0
                print(f"Processed {idx + 1}/{total_rows} cases. Rate: {rate:.1f}/min, ETA: {eta_minutes:.1f}min")


        except Exception as e:
            logging.error(f"Error processing case {case_id}: {e}")
            result_df.at[idx, 'reasoning'] = f"Error: {str(e)}"

    elapsed = time.time() - start_time
    final_rate = total_rows / elapsed * 60
    logging.info(f"Processing complete! {total_rows} cases in {elapsed:.1f}s ({final_rate:.1f} cases/min)")
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
    results_df = process_llm_cases(llm_df, api_key, delay_seconds=0.2)

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


def fast_batch_processing(final_llm_df: pd.DataFrame,
                          gcs_bucket_path: str = 'gs://starr-sinusitis_2016_2025',
                          local_save_directory: str = './llm_results',
                          batch_size: int = 200,
                          max_workers: int = 5,
                          save_every_n_batches: int = 5) -> pd.DataFrame:
    """
    Fast batch processing with parallel execution for any LLM provider.

    Args:
        final_llm_df: DataFrame with cases to process
        llm_provider: LLM provider instance
        gcs_bucket_path: GCS bucket path for saving
        batch_size: Cases per batch
        max_workers: Parallel workers per batch
        save_every_n_batches: Save to GCS every N batches (recommended: 5-10)
    """
    # Create model-specific save directory
    model_save_dir = os.path.join(local_save_directory, f"{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace(':', '_').replace('@', '_')}")
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"Using model-specific save directory: {model_save_dir}")
    print(f"GCS bucket path: {gcs_bucket_path}")
    print(f"Using LLM Provider: {llm_provider.provider_name} with model: {llm_provider.model_name}")

    # Load existing results for this specific model
    completed_ids = get_completed_case_ids(model_save_dir)
    remaining_df = final_llm_df[~final_llm_df['llm_caseID'].isin(completed_ids)].copy()

    print(f"Processing {len(remaining_df)} remaining cases...")
    print(f"Using {max_workers} parallel workers per batch")

    if remaining_df.empty:
        print("All cases already completed for this model!")
        return load_existing_results(model_save_dir)

    all_results = []
    num_batches = (len(remaining_df) + batch_size - 1) // batch_size

    # Create timestamped session folder for this run
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_folder = f"llm_processing_{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace('@', '_')}_{session_timestamp}"

    # Reset index for clean batch numbering
    remaining_df = remaining_df.reset_index(drop=True)

    for i in range(0, len(remaining_df), batch_size):
        batch_num = i // batch_size + 1
        batch_data = remaining_df.iloc[i:i + batch_size].copy()

        # Check if this specific batch was already processed for this model
        save_file = os.path.join(model_save_dir, f'llm_results_{llm_provider.provider_name}_batch_{batch_num}.pkl')
        if os.path.exists(save_file):
            print(f"Batch {batch_num} already exists for {llm_provider.provider_name}, loading from {save_file}")
            try:
                batch_results = pd.read_pickle(save_file)
                all_results.append(batch_results)
                continue
            except Exception as e:
                print(f"Error loading {save_file}, will reprocess: {e}")

        print(f"\nProcessing batch {batch_num}/{num_batches} ({len(batch_data)} cases) with {llm_provider.provider_name}...")
        case_ids = batch_data['llm_caseID'].tolist()
        print(f"Case ID range: {min(case_ids)}-{max(case_ids)}")

        # Process batch in parallel
        start_time = time.time()
        batch_results = parallel_process_llm_cases(batch_data, llm_provider, max_workers)
        end_time = time.time()

        if not batch_results.empty:
            batch_time = end_time - start_time
            cases_per_minute = len(batch_data) / (batch_time / 60)

            print(f"✓ Batch {batch_num} completed in {batch_time:.1f}s ({cases_per_minute:.1f} cases/min)")

            # Save intermediate results locally
            try:
                batch_results.to_pickle(save_file)
                print(f"Saved locally to {save_file}")
                all_results.append(batch_results)
            except Exception as e:
                print(f"  Error saving batch {batch_num} locally: {e}")
                all_results.append(batch_results)

            # Save to GCS
            try:
                gcs_batch_path = f"{gcs_bucket_path}/{session_folder}/batch_{batch_num:03d}.parquet"
                batch_results.to_parquet(gcs_batch_path)
                print(f"Batch {batch_num} saved to GCS: {gcs_batch_path}")
            except Exception as e:
                print(f"  Error saving batch {batch_num} to GCS: {e}")

            # Save combined results to GCS every N batches
            if batch_num % save_every_n_batches == 0 and all_results:
                print(f"\n=== Saving progress to GCS (after {batch_num} batches) ===")
                combined_so_far = pd.concat(all_results, ignore_index=True)

                # Save intermediate combined results
                gcs_progress_path = f"{gcs_bucket_path}/{session_folder}/progress_after_batch_{batch_num:03d}.parquet"
                try:
                    combined_so_far.to_parquet(gcs_progress_path)
                    print(f"Progress saved to GCS: {gcs_progress_path}")
                except Exception as e:
                    print(f"Error saving progress to GCS: {e}")

                # Also save as "latest" for easy recovery
                gcs_latest_path = f"{gcs_bucket_path}/final_llm_results_{llm_provider.provider_name}_latest.parquet"
                try:
                    combined_so_far.to_parquet(gcs_latest_path)
                    print(f"Latest results saved to GCS: {gcs_latest_path}")
                except Exception as e:
                    print(f"Error saving latest to GCS: {e}")

                print(f"Progress saved: {len(combined_so_far)} cases processed so far")

        else:
            print(f"✗ Batch {batch_num} returned no results")

        # Clean up memory
        del batch_data
        if 'batch_results' in locals():
            del batch_results
        gc.collect()

        # Show progress
        total_processed = sum(len(result) for result in all_results)
        print(f"Progress: {total_processed} cases completed so far")

    # Final save to GCS
    if all_results:
        print(f"\n=== Saving FINAL results to GCS ===")
        final_results = pd.concat(all_results, ignore_index=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Main final results with model name
        final_gcs_path = f"{gcs_bucket_path}/final_llm_results_{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace('@', '_')}_{timestamp}.parquet"
        try:
            final_results.to_parquet(final_gcs_path)
            print(f"✓ Final results saved to GCS: {final_gcs_path}")
        except Exception as e:
            print(f"✗ Error saving final results to GCS: {e}")

        # Save local copy
        local_final_file = os.path.join(model_save_dir, f'final_parallel_llm_results_{llm_provider.provider_name}_{timestamp}.pkl')
        try:
            final_results.to_pickle(local_final_file)
            print(f"✓ Final results also saved locally to {local_final_file}")
        except Exception as e:
            print(f"✗ Error saving locally: {e}")

        print(f"\nPROCESSING COMPLETE!")
        print(f"Total cases processed: {len(final_results)}")
        print(f"Model used: {llm_provider.provider_name} ({llm_provider.model_name})")
        print(f"Final results: {final_gcs_path}")

        return final_results
    else:
        print("No results to combine")
        return pd.DataFrame()
    
    

    # def fast_batch_processing_open_ai(final_llm_df: pd.DataFrame,
#                           gcs_bucket_path: str = 'gs://starr-sinusitis_2016_2025',
#                           local_save_directory: str = './llm_results',
#                           batch_size: int = 200,
#                           max_workers: int = 5,
#                           save_every_n_batches: int = 5) -> pd.DataFrame:
#     """
#     Fast batch processing with parallel execution for any LLM provider.

#     Args:
#         final_llm_df: DataFrame with cases to process
#         llm_provider: LLM provider instance
#         gcs_bucket_path: GCS bucket path for saving
#         batch_size: Cases per batch
#         max_workers: Parallel workers per batch
#         save_every_n_batches: Save to GCS every N batches (recommended: 5-10)
#     """
#     # Create model-specific save directory
#     model_save_dir = os.path.join(local_save_directory, f"{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace(':', '_').replace('@', '_')}")
#     os.makedirs(model_save_dir, exist_ok=True)

#     print(f"Using model-specific save directory: {model_save_dir}")
#     print(f"GCS bucket path: {gcs_bucket_path}")
#     print(f"Using LLM Provider: {llm_provider.provider_name} with model: {llm_provider.model_name}")

#     # Load existing results for this specific model
#     completed_ids = get_completed_case_ids(model_save_dir)
#     remaining_df = final_llm_df[~final_llm_df['llm_caseID'].isin(completed_ids)].copy()

#     print(f"Processing {len(remaining_df)} remaining cases...")
#     print(f"Using {max_workers} parallel workers per batch")

#     if remaining_df.empty:
#         print("All cases already completed for this model!")
#         return load_existing_results(model_save_dir)

#     all_results = []
#     num_batches = (len(remaining_df) + batch_size - 1) // batch_size

#     # Create timestamped session folder for this run
#     session_timestamp = time.strftime("%Y%m%d_%H%M%S")
#     session_folder = f"llm_processing_{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace('@', '_')}_{session_timestamp}"

#     # Reset index for clean batch numbering
#     remaining_df = remaining_df.reset_index(drop=True)

#     for i in range(0, len(remaining_df), batch_size):
#         batch_num = i // batch_size + 1
#         batch_data = remaining_df.iloc[i:i + batch_size].copy()

#         # Check if this specific batch was already processed for this model
#         save_file = os.path.join(model_save_dir, f'llm_results_{llm_provider.provider_name}_batch_{batch_num}.pkl')
#         if os.path.exists(save_file):
#             print(f"Batch {batch_num} already exists for {llm_provider.provider_name}, loading from {save_file}")
#             try:
#                 batch_results = pd.read_pickle(save_file)
#                 all_results.append(batch_results)
#                 continue
#             except Exception as e:
#                 print(f"Error loading {save_file}, will reprocess: {e}")

#         print(f"\nProcessing batch {batch_num}/{num_batches} ({len(batch_data)} cases) with {llm_provider.provider_name}...")
#         case_ids = batch_data['llm_caseID'].tolist()
#         print(f"Case ID range: {min(case_ids)}-{max(case_ids)}")

#         # Process batch in parallel
#         start_time = time.time()
#         batch_results = parallel_process_llm_cases(batch_data, llm_provider, max_workers)
#         end_time = time.time()

#         if not batch_results.empty:
#             batch_time = end_time - start_time
#             cases_per_minute = len(batch_data) / (batch_time / 60)

#             print(f"✓ Batch {batch_num} completed in {batch_time:.1f}s ({cases_per_minute:.1f} cases/min)")

#             # Save intermediate results locally
#             try:
#                 batch_results.to_pickle(save_file)
#                 print(f"Saved locally to {save_file}")
#                 all_results.append(batch_results)
#             except Exception as e:
#                 print(f"  Error saving batch {batch_num} locally: {e}")
#                 all_results.append(batch_results)

#             # Save to GCS
#             try:
#                 gcs_batch_path = f"{gcs_bucket_path}/{session_folder}/batch_{batch_num:03d}.parquet"
#                 batch_results.to_parquet(gcs_batch_path)
#                 print(f"Batch {batch_num} saved to GCS: {gcs_batch_path}")
#             except Exception as e:
#                 print(f"  Error saving batch {batch_num} to GCS: {e}")

#             # Save combined results to GCS every N batches
#             if batch_num % save_every_n_batches == 0 and all_results:
#                 print(f"\n=== Saving progress to GCS (after {batch_num} batches) ===")
#                 combined_so_far = pd.concat(all_results, ignore_index=True)

#                 # Save intermediate combined results
#                 gcs_progress_path = f"{gcs_bucket_path}/{session_folder}/progress_after_batch_{batch_num:03d}.parquet"
#                 try:
#                     combined_so_far.to_parquet(gcs_progress_path)
#                     print(f"Progress saved to GCS: {gcs_progress_path}")
#                 except Exception as e:
#                     print(f"Error saving progress to GCS: {e}")

#                 # Also save as "latest" for easy recovery
#                 gcs_latest_path = f"{gcs_bucket_path}/final_llm_results_{llm_provider.provider_name}_latest.parquet"
#                 try:
#                     combined_so_far.to_parquet(gcs_latest_path)
#                     print(f"Latest results saved to GCS: {gcs_latest_path}")
#                 except Exception as e:
#                     print(f"Error saving latest to GCS: {e}")

#                 print(f"Progress saved: {len(combined_so_far)} cases processed so far")

#         else:
#             print(f"✗ Batch {batch_num} returned no results")

#         # Clean up memory
#         del batch_data
#         if 'batch_results' in locals():
#             del batch_results
#         gc.collect()

#         # Show progress
#         total_processed = sum(len(result) for result in all_results)
#         print(f"Progress: {total_processed} cases completed so far")

#     # Final save to GCS
#     if all_results:
#         print(f"\n=== Saving FINAL results to GCS ===")
#         final_results = pd.concat(all_results, ignore_index=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")

#         # Main final results with model name
#         final_gcs_path = f"{gcs_bucket_path}/final_llm_results_{llm_provider.provider_name}_{llm_provider.model_name.replace('/', '_').replace('@', '_')}_{timestamp}.parquet"
#         try:
#             final_results.to_parquet(final_gcs_path)
#             print(f"✓ Final results saved to GCS: {final_gcs_path}")
#         except Exception as e:
#             print(f"✗ Error saving final results to GCS: {e}")

#         # Save local copy
#         local_final_file = os.path.join(model_save_dir, f'final_parallel_llm_results_{llm_provider.provider_name}_{timestamp}.pkl')
#         try:
#             final_results.to_pickle(local_final_file)
#             print(f"✓ Final results also saved locally to {local_final_file}")
#         except Exception as e:
#             print(f"✗ Error saving locally: {e}")

#         print(f"\nPROCESSING COMPLETE!")
#         print(f"Total cases processed: {len(final_results)}")
#         print(f"Model used: {llm_provider.provider_name} ({llm_provider.model_name})")
#         print(f"Final results: {final_gcs_path}")

#         return final_results
#     else:
#         print("No results to combine")
#         return pd.DataFrame()

# import openai
# import pandas as pd
# import json
# import logging
# import time
# from typing import Dict, Any
# from tqdm import tqdm


# def generate_prompt(case_id: str, progress_text: str, radiology_text: str) -> str:
#     """Generates a structured prompt for the LLM."""

#     # Check if radiology text is available
#     has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
#     radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."

#     prompt = f"""
#     You are an expert otolaryngologist evaluating an ENT case.
#     Based ONLY on the information provided below, make a recommendation on surgery.

#     --- Case Details ---
#     - Case ID: {case_id}
#     - Clinical Summary from ENT Notes: {progress_text}
#     {radiology_section}

#     ---

#     Provide your response as a JSON object with three keys:
#     1. "decision": Your recommendation, either "Yes" or "No".
#     2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
#     3. "reasoning": A brief, 2-4 sentence explanation for your decision.

#     Return ONLY the JSON object, no additional text.
#     """
#     return prompt

# def parse_llm_response(response: str) -> Dict[str, Any]:
#     """Parse LLM response and extract decision, confidence, and reasoning."""
#     default_response = {
#         'decision': None,
#         'confidence': None,
#         'reasoning': 'Failed to parse response'
#     }

#     if not response:
#         return default_response

#     try:
#         # Try to find JSON in the response
#         response = response.strip()
#         if response.startswith('```json'):
#             response = response.replace('```json', '').replace('```', '').strip()
#         elif response.startswith('```'):
#             response = response.replace('```', '').strip()

#         parsed = json.loads(response)

#         return {
#             'decision': parsed.get('decision'),
#             'confidence': parsed.get('confidence'),
#             'reasoning': parsed.get('reasoning', 'No reasoning provided')
#         }
#     except json.JSONDecodeError as e:
#         logging.error(f"JSON parsing error: {e}")
#         logging.error(f"Response was: {response}")
#         return default_response
#     except Exception as e:
#         logging.error(f"Unexpected error parsing response: {e}")
#         return default_response

# import asyncio
# import aiohttp
# import openai
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time

# def parallel_process_llm_cases(llm_df: pd.DataFrame,
#                              llm_provider: LLMProvider,
#                              max_workers: int = 5,
#                              delay_seconds: float = 0.1) -> pd.DataFrame:
#     """
#     Process LLM cases in parallel using ThreadPoolExecutor.

#     Args:
#         llm_df: DataFrame with cases to process
#         llm_provider: LLM provider instance
#         max_workers: Number of parallel workers (start with 5)
#         delay_seconds: Delay between requests (can be smaller with parallel)
#     """

#     def process_single_case(row_data):
#         """Process a single case - designed for parallel execution."""
#         idx, row = row_data

#         try:
#             case_id = row['llm_caseID']

#             # Generate prompt
#             prompt = generate_prompt(
#                 case_id=case_id,
#                 progress_text=row['formatted_progress_text'],
#                 radiology_text=row['formatted_radiology_text']
#             )

#             # Query LLM
#             response = llm_provider.query(prompt)

#             result = {
#                 'index': idx,
#                 'llm_caseID': case_id,
#                 'api_response': response,
#                 'decision': None,
#                 'confidence': None,
#                 'reasoning': None,
#                 'model_name': llm_provider.model_name,
#                 'provider': llm_provider.provider_name
#             }

#             if response:
#                 parsed = parse_llm_response(response)
#                 result.update(parsed)

#             # Small delay to avoid rate limiting
#             time.sleep(delay_seconds)

#             return result

#         except Exception as e:
#             return {
#                 'index': idx,
#                 'llm_caseID': row['llm_caseID'],
#                 'api_response': None,
#                 'decision': None,
#                 'confidence': None,
#                 'reasoning': f"Error: {str(e)}",
#                 'model_name': llm_provider.model_name,
#                 'provider': llm_provider.provider_name
#             }

#     print(f"Processing {len(llm_df)} cases with {max_workers} parallel workers using {llm_provider.provider_name} ({llm_provider.model_name})...")

#     # Prepare results dataframe
#     result_df = llm_df.copy()
#     result_df['decision'] = None
#     result_df['confidence'] = None
#     result_df['reasoning'] = None
#     result_df['api_response'] = None
#     result_df['model_name'] = llm_provider.model_name
#     result_df['provider'] = llm_provider.provider_name

#     # Process in parallel
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all tasks
#         future_to_row = {
#             executor.submit(process_single_case, (idx, row)): idx
#             for idx, row in llm_df.iterrows()
#         }

#         # Collect results with progress bar
#         completed = 0
#         for future in tqdm(as_completed(future_to_row), total=len(llm_df), desc="Processing"):
#             try:
#                 result = future.result()
#                 idx = result['index']

#                 # Update result dataframe
#                 result_df.at[idx, 'decision'] = result['decision']
#                 result_df.at[idx, 'confidence'] = result['confidence']
#                 result_df.at[idx, 'reasoning'] = result['reasoning']
#                 result_df.at[idx, 'api_response'] = result['api_response']
#                 result_df.at[idx, 'model_name'] = result['model_name']
#                 result_df.at[idx, 'provider'] = result['provider']

#                 completed += 1
#                 if completed % 50 == 0:  # Progress updates
#                     print(f"Completed {completed}/{len(llm_df)} cases")

#             except Exception as e:
#                 print(f"Error processing future: {e}")

#     return result_df

# def fast_batch_processing(final_llm_df: pd.DataFrame,
#                           api_key: str,
#                           gcs_bucket_path: str = 'gs://starr-sinusitis_2016_2025',
#                           local_save_directory: str = './llm_results',
#                           batch_size: int = 200,
#                           max_workers: int = 5,
#                           save_every_n_batches: int = 5) -> pd.DataFrame:
#     """
#     Fast batch processing with parallel execution.

#     Args:
#         final_llm_df: DataFrame with cases to process
#         api_key: OpenAI API key
#         gcs_bucket_path: GCS bucket path for saving
#         batch_size: Cases per batch
#         max_workers: Parallel workers per batch
#         save_every_n_batches: Save to GCS every N batches (recommended: 5-10)

#     """
#     # Check save directory exists
#     os.makedirs(local_save_directory, exist_ok=True)
#     print(f"Using local save directory: {local_save_directory}")
#     print(f"GCS bucket path: {gcs_bucket_path}")

#     # Load existing results
#     completed_ids = get_completed_case_ids(local_save_directory)
#     remaining_df = final_llm_df[~final_llm_df['llm_caseID'].isin(completed_ids)].copy()

#     print(f"Processing {len(remaining_df)} remaining cases...")
#     print(f"Using {max_workers} parallel workers per batch")

#     if remaining_df.empty:
#         print("All cases already completed!")
#         return load_existing_results(local_save_directory)

#     all_results = []
#     num_batches = (len(remaining_df) + batch_size - 1) // batch_size

#     # Create timestamped session folder for this run
#     session_timestamp = time.strftime("%Y%m%d_%H%M%S")
#     session_folder = f"llm_processing_{session_timestamp}"

#     # Reset index for clean batch numbering
#     remaining_df = remaining_df.reset_index(drop=True)

#     for i in range(0, len(remaining_df), batch_size):
#         batch_num = i // batch_size + 1
#         batch_data = remaining_df.iloc[i:i + batch_size].copy()

#         # Check if this specific batch was already processed
#         save_file = os.path.join(local_save_directory, f'llm_results_parallel_batch_{batch_num}.pkl')
#         if os.path.exists(save_file):
#             print(f"Batch {batch_num} already exists, loading from {save_file}")
#             try:
#                 batch_results = pd.read_pickle(save_file)
#                 all_results.append(batch_results)
#                 continue
#             except Exception as e:
#                 print(f"Error loading {save_file}, will reprocess: {e}")

#         print(f"\nProcessing batch {batch_num}/{num_batches} ({len(batch_data)} cases)...")
#         case_ids = batch_data['llm_caseID'].tolist()
#         print(f"Case ID range: {min(case_ids)}-{max(case_ids)}")

#         # Process batch in parallel
#         start_time = time.time()
#         batch_results = parallel_process_llm_cases(batch_data, api_key, max_workers)
#         end_time = time.time()

#         if not batch_results.empty:
#             batch_time = end_time - start_time
#             cases_per_minute = len(batch_data) / (batch_time / 60)

#             print(f"✓ Batch {batch_num} completed in {batch_time:.1f}s ({cases_per_minute:.1f} cases/min)")

#             # Save intermediate results locally
#             try:
#                 batch_results.to_pickle(save_file)
#                 print(f"Saved locally to {save_file}")
#                 all_results.append(batch_results)
#             except Exception as e:
#                 print(f"  Error saving batch {batch_num} locally: {e}")
#                 all_results.append(batch_results)

#             # Save to GCS
#             try:
#                 gcs_batch_path = f"{gcs_bucket_path}/{session_folder}/batch_{batch_num:03d}.parquet"
#                 batch_results.to_parquet(gcs_batch_path)
#                 print(f"Batch {batch_num} saved to GCS: {gcs_batch_path}")
#             except Exception as e:
#                 print(f"  Error saving batch {batch_num} to GCS: {e}")

#             # Save combined results to GCS every N batches
#             if batch_num % save_every_n_batches == 0 and all_results:
#                 print(f"\n=== Saving progress to GCS (after {batch_num} batches) ===")
#                 combined_so_far = pd.concat(all_results, ignore_index=True)

#                 # Save intermediate combined results
#                 gcs_progress_path = f"{gcs_bucket_path}/{session_folder}/progress_after_batch_{batch_num:03d}.parquet"
#                 try:
#                     combined_so_far.to_parquet(gcs_progress_path)
#                     print(f"Progress saved to GCS: {gcs_progress_path}")
#                 except Exception as e:
#                     print(f"Error saving progress to GCS: {e}")

#                 # Also save as "latest" for easy recovery
#                 gcs_latest_path = f"{gcs_bucket_path}/final_llm_results_latest.parquet"
#                 try:
#                     combined_so_far.to_parquet(gcs_latest_path)
#                     print(f"Latest results saved to GCS: {gcs_latest_path}")
#                 except Exception as e:
#                     print(f"Error saving latest to GCS: {e}")

#                 print(f"Progress saved: {len(combined_so_far)} cases processed so far")

#         else:
#             print(f"✗ Batch {batch_num} returned no results")

#         # Clean up memory
#         del batch_data
#         if 'batch_results' in locals():
#             del batch_results
#         gc.collect()

#         # Show progress
#         total_processed = sum(len(result) for result in all_results)
#         print(f"Progress: {total_processed} cases completed so far")

#     # Final save to GCS
#     if all_results:
#         print(f"\n=== Saving FINAL results to GCS ===")
#         final_results = pd.concat(all_results, ignore_index=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")

#         # Main final results
#         final_gcs_path = f"{gcs_bucket_path}/final_llm_results_{timestamp}.parquet"
#         try:
#             final_results.to_parquet(final_gcs_path)
#             print(f"✓ Final results saved to GCS: {final_gcs_path}")
#         except Exception as e:
#             print(f"✗ Error saving final results to GCS: {e}")

#         # Save local copy
#         local_final_file = os.path.join(local_save_directory, f'final_parallel_llm_results_{timestamp}.pkl')
#         try:
#             final_results.to_pickle(local_final_file)
#             print(f"✓ Final results also saved locally to {local_final_file}")
#         except Exception as e:
#             print(f"✗ Error saving locally: {e}")

#         print(f"\nPROCESSING COMPLETE!")
#         print(f"Total cases processed: {len(final_results)}")
#         print(f"Final results: {final_gcs_path}")

#         return final_results
#     else:
#         print("No results to combine")
#         return pd.DataFrame()

# final_results = fast_batch_processing(
#     final_llm_df=llm_df_filtered,
#     api_key=API_KEY,
#     gcs_bucket_path='gs://starr-sinusitis_2016_2025',
#     local_save_directory='./llm_results',
#     batch_size=200,
#     max_workers=5,
#     save_every_n_batches=5 # Save progress every 1000 cases
# )

# processed_final_results=final_results[final_results["llm_caseID"] <= 3837].copy()
# processed_final_results.tail()