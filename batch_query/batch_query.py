import glob

def load_existing_results(save_directory: str = '.') -> pd.DataFrame:
    """Load all existing intermediate results."""
    pattern = os.path.join(save_directory, 'llm_results_batch_*.pkl')
    result_files = glob.glob(pattern)

    if not result_files:
        print("No existing results found")
        return pd.DataFrame()

    print(f"Found {len(result_files)} existing result files")

    all_results = []
    for file in sorted(result_files):
        try:
            batch_results = pd.read_pickle(file)
            all_results.append(batch_results)
            batch_num = file.split('_')[-1].replace('.pkl', '')
            print(f"  Loaded batch {batch_num}: {len(batch_results)} cases")
        except Exception as e:
            print(f"  Error loading {file}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"Total existing results: {len(combined)} cases")
        return combined
    else:
        return pd.DataFrame()

def get_completed_case_ids(save_directory: str = '.') -> set:
    """Get set of case IDs that have already been processed."""
    existing_results = load_existing_results(save_directory)
    if existing_results.empty:
        return set()

    completed_ids = set(existing_results['llm_caseID'].unique())
    print(f"Found {len(completed_ids)} already completed cases")
    return completed_ids

def process_remaining_llm_data(final_llm_df: pd.DataFrame,
                             api_key: str,
                             batch_size: int = 500,
                             sub_batch_size: int = 25,
                             save_directory: str = '.') -> pd.DataFrame:
    """
    Process only the remaining cases that haven't been completed yet.

    Args:
        final_llm_df: Your complete LLM dataframe
        api_key: OpenAI API key
        batch_size: Size of main batches
        sub_batch_size: Size of LLM processing sub-batches
        save_directory: Where to save/load intermediate results

    Returns:
        Complete results dataframe (existing + new)
    """

    print(f"=== Resuming LLM Analysis ===")

    # Load existing results
    existing_results = load_existing_results(save_directory)
    completed_case_ids = get_completed_case_ids(save_directory)

    # Filter out already completed cases
    remaining_df = final_llm_df[~final_llm_df['llm_caseID'].isin(completed_case_ids)].copy()

    print(f"Original cases: {len(final_llm_df)}")
    print(f"Already completed: {len(completed_case_ids)}")
    print(f"Remaining to process: {len(remaining_df)}")

    if remaining_df.empty:
        print("All cases already completed!")
        return existing_results

    # Process remaining cases
    print(f"\nProcessing {len(remaining_df)} remaining cases...")
    new_results = []
    num_batches = (len(remaining_df) + batch_size - 1) // batch_size

    # Reset index for clean batch processing
    remaining_df = remaining_df.reset_index(drop=True)

    for i in range(0, len(remaining_df), batch_size):
        batch_num = i // batch_size + 1
        batch_data = remaining_df.iloc[i:i + batch_size].copy()

        print(f"\n{'='*60}")
        print(f"PROCESSING REMAINING BATCH {batch_num}/{num_batches}")
        print(f"{'='*60}")

        # Check if this batch was already processed
        batch_case_ids = set(batch_data['llm_caseID'])
        if batch_case_ids.issubset(completed_case_ids):
            print(f"Batch {batch_num} already completed, skipping...")
            continue

        # Process this batch
        batch_results = run_batch_llm_analysis(
            batch_data, batch_num, api_key, sub_batch_size
        )

        if not batch_results.empty:
            new_results.append(batch_results)

            # Save intermediate results with unique filename
            save_file = os.path.join(save_directory, f'llm_results_batch_resume_{batch_num}.pkl')
            batch_results.to_pickle(save_file)
            print(f"Saved results to {save_file}")

        # Clean up
        del batch_data
        if 'batch_results' in locals():
            del batch_results
        gc.collect()

    # Combine all results (existing + new)
    all_results = []

    if not existing_results.empty:
        all_results.append(existing_results)

    if new_results:
        all_results.extend(new_results)

    if all_results:
        print(f"\n=== Combining all results ===")
        final_results = pd.concat(all_results, ignore_index=True)

        # Remove duplicates just in case
        final_results = final_results.drop_duplicates(subset=['llm_caseID'], keep='first')

        print(f"Final combined results: {len(final_results)} cases")
        print(f"Success rate: {final_results['decision'].notna().sum()}/{len(final_results)} "
              f"({final_results['decision'].notna().mean()*100:.1f}%)")

        # Save final combined results
        final_save_file = os.path.join(save_directory, 'final_llm_results_complete.pkl')
        final_results.to_pickle(final_save_file)
        print(f"Saved final results to {final_save_file}")

        return final_results
    else:
        print("No results to combine")
        return pd.DataFrame()

def quick_status_check(final_llm_df: pd.DataFrame, save_directory: str = '.'):
    """Quick check of what's completed vs remaining."""
    completed_ids = get_completed_case_ids(save_directory)
    total_cases = len(final_llm_df)
    completed_cases = len(completed_ids)
    remaining_cases = total_cases - completed_cases

    print(f"\n=== STATUS CHECK ===")
    print(f"Total cases: {total_cases}")
    print(f"Completed: {completed_cases} ({completed_cases/total_cases*100:.1f}%)")
    print(f"Remaining: {remaining_cases} ({remaining_cases/total_cases*100:.1f}%)")

    if completed_cases > 0:
        # Show some sample completed case IDs
        sample_completed = sorted(list(completed_ids))[:10]
        print(f"Sample completed case IDs: {sample_completed}")

import asyncio
import aiohttp
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import asyncio
import aiohttp
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def parallel_process_llm_cases(llm_df: pd.DataFrame,
                             api_key: str,
                             max_workers: int = 5,
                             delay_seconds: float = 0.1) -> pd.DataFrame:
    """
    Process LLM cases in parallel using ThreadPoolExecutor.

    Args:
        llm_df: DataFrame with cases to process
        api_key: OpenAI API key
        max_workers: Number of parallel workers (start with 5)
        delay_seconds: Delay between requests (can be smaller with parallel)
    """

    def process_single_case(row_data):
        """Process a single case - designed for parallel execution."""
        idx, row = row_data

        try:
            client = openai.OpenAI(api_key=api_key)

            case_id = row['llm_caseID']

            # Generate prompt
            prompt = generate_prompt(
                case_id=case_id,
                progress_text=row['formatted_progress_text'],
                radiology_text=row['formatted_radiology_text']
            )

            # Query OpenAI
            response = query_openai(prompt, client)

            result = {
                'index': idx,
                'llm_caseID': case_id,
                'api_response': response,
                'decision': None,
                'confidence': None,
                'reasoning': None
            }

            if response:
                parsed = parse_llm_response(response)
                result.update(parsed)

            # Small delay to avoid rate limiting
            time.sleep(delay_seconds)

            return result

        except Exception as e:
            return {
                'index': idx,
                'llm_caseID': row['llm_caseID'],
                'api_response': None,
                'decision': None,
                'confidence': None,
                'reasoning': f"Error: {str(e)}"
            }

    print(f"Processing {len(llm_df)} cases with {max_workers} parallel workers...")

    # Prepare results dataframe
    result_df = llm_df.copy()
    result_df['decision'] = None
    result_df['confidence'] = None
    result_df['reasoning'] = None
    result_df['api_response'] = None

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_case, (idx, row)): idx
            for idx, row in llm_df.iterrows()
        }

        # Collect results with progress bar
        completed = 0
        for future in tqdm(as_completed(future_to_row), total=len(llm_df), desc="Processing"):
            try:
                result = future.result()
                idx = result['index']

                # Update result dataframe
                result_df.at[idx, 'decision'] = result['decision']
                result_df.at[idx, 'confidence'] = result['confidence']
                result_df.at[idx, 'reasoning'] = result['reasoning']
                result_df.at[idx, 'api_response'] = result['api_response']

                completed += 1
                if completed % 50 == 0:  # Progress updates
                    print(f"Completed {completed}/{len(llm_df)} cases")

            except Exception as e:
                print(f"Error processing future: {e}")

    return result_df

def fast_batch_processing(final_llm_df: pd.DataFrame,
                         api_key: str,
                         batch_size: int = 200,
                         max_workers: int = 5) -> pd.DataFrame:
    """
    Fast batch processing with parallel execution.
    """

    # Load existing results
    completed_ids = get_completed_case_ids()
    remaining_df = final_llm_df[~final_llm_df['llm_caseID'].isin(completed_ids)].copy()

    print(f"Processing {len(remaining_df)} remaining cases...")
    print(f"Using {max_workers} parallel workers per batch")

    all_results = []
    num_batches = (len(remaining_df) + batch_size - 1) // batch_size

    # Reset index for clean batch numbering
    remaining_df = remaining_df.reset_index(drop=True)

    for i in range(0, len(remaining_df), batch_size):
        batch_num = i // batch_size + 1
        batch_data = remaining_df.iloc[i:i + batch_size].copy()

        # Check if this specific batch was already processed
        save_file = os.path.join(save_directory, f'llm_results_parallel_batch_{batch_num}.pkl')
        if os.path.exists(save_file):
            print(f"Batch {batch_num} already exists, loading from {save_file}")
            try:
                batch_results = pd.read_pickle(save_file)
                all_results.append(batch_results)
                continue
            except Exception as e:
                print(f"Error loading {save_file}, will reprocess: {e}")

        print(f"\nProcessing batch {batch_num}/{num_batches} ({len(batch_data)} cases)...")
        case_ids = batch_data['llm_caseID'].tolist()
        print(f"Case ID range: {min(case_ids)}-{max(case_ids)}")

        # Process batch in parallel
        start_time = time.time()
        batch_results = parallel_process_llm_cases(batch_data, api_key, max_workers)
        end_time = time.time()

        if not batch_results.empty:
            batch_time = end_time - start_time
            cases_per_minute = len(batch_data) / (batch_time / 60)

            print(f"✓ Batch {batch_num} completed in {batch_time:.1f}s ({cases_per_minute:.1f} cases/min)")

            # Save intermediate results immediately
            try:
                batch_results.to_pickle(save_file)
                print(f"  Saved to {save_file}")
                all_results.append(batch_results)

                # Also save a backup with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(save_directory, f'backup_batch_{batch_num}_{timestamp}.pkl')
                batch_results.to_pickle(backup_file)

            except Exception as e:
                print(f"  Error saving batch {batch_num}: {e}")
                all_results.append(batch_results)  # Still keep in memory
        else:
            print(f"✗ Batch {batch_num} returned no results")

        # Clean up
        del batch_data
        if 'batch_results' in locals():
            del batch_results
        gc.collect()

        # Show progress
        total_processed = sum(len(result) for result in all_results)
        print(f"Progress: {total_processed} cases completed so far")

    # Combine results
    if all_results:
        print(f"\n=== Combining {len(all_results)} batches ===")
        final_results = pd.concat(all_results, ignore_index=True)

        # Save final combined results
        final_save_file = os.path.join(save_directory, 'final_parallel_llm_results.pkl')
        final_results.to_pickle(final_save_file)
        print(f"Final results saved to {final_save_file}")

        return final_results
    else:
        print("No results to combine")
        return pd.DataFrame()