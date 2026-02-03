import pandas as pd
import json
import logging
import time
import gc
import os
from typing import Dict, Any, Optional, Set
from tqdm import tqdm

from llm_query.securellm_adapter import query_llm, SecureLLMClient


def query_openai(prompt: str, client=None) -> str:
    """
    Query the LLM for surgical decision based on input prompt.

    This function now uses SecureLLM instead of direct OpenAI calls.
    The client parameter is kept for backward compatibility but is ignored.

    Args:
        prompt: The prompt to send to the LLM.
        client: Deprecated. Kept for backward compatibility.

    Returns:
        The LLM response content or None on error.
    """
    return query_llm(
        prompt=prompt,
        system_message="You are an expert otolaryngologist. Provide a surgical recommendation in the requested JSON format.",
        temperature=0.2
    )

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

def _load_processed_case_ids(output_file: Optional[str]) -> Set[str]:
    """Load already processed case IDs from existing output file."""
    if not output_file or not os.path.exists(output_file):
        return set()

    try:
        existing_df = pd.read_csv(output_file)
        if 'llm_caseID' in existing_df.columns:
            # Only count cases that have a decision (successfully processed)
            processed = existing_df[existing_df['decision'].notna()]['llm_caseID'].astype(str).tolist()
            return set(processed)
    except Exception as e:
        logging.warning(f"Could not read existing output file: {e}")

    return set()


def _flush_results_to_csv(
    results: list,
    output_file: str,
    write_header: bool
) -> None:
    """Flush batch results to CSV file and free memory."""
    if not results:
        return

    batch_df = pd.DataFrame(results)
    batch_df.to_csv(
        output_file,
        mode='a' if not write_header else 'w',
        header=write_header,
        index=False
    )

    # Clear the list and force garbage collection
    results.clear()
    gc.collect()


def process_llm_cases(
    llm_df: pd.DataFrame,
    api_key: str = None,
    delay_seconds: float = 0.2,
    output_file: Optional[str] = None,
    flush_interval: int = 10,
    resume: bool = True
) -> pd.DataFrame:
    """
    Process a clean LLM DataFrame through SecureLLM API with incremental saving.

    Args:
        llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
        api_key: Deprecated. Kept for backward compatibility. SecureLLM uses VAULT_SECRET_KEY.
        delay_seconds: Delay between API calls to avoid rate limiting
        output_file: Path to output CSV file for incremental saving
        flush_interval: Number of cases to process before flushing to disk
        resume: If True, skip cases already in output_file

    Returns:
        DataFrame with additional columns: 'decision', 'confidence', 'reasoning', 'api_response'
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize SecureLLM client
    client = SecureLLMClient()
    logging.info("SecureLLM client initialized successfully")

    # Load already processed case IDs if resuming
    processed_ids: Set[str] = set()
    write_header = True

    if resume and output_file:
        processed_ids = _load_processed_case_ids(output_file)
        if processed_ids:
            logging.info(f"Resuming: {len(processed_ids)} cases already processed, will skip them")
            write_header = False  # Append to existing file

    # Filter out already processed cases
    if processed_ids:
        pending_df = llm_df[~llm_df['llm_caseID'].astype(str).isin(processed_ids)].copy()
        logging.info(f"Remaining cases to process: {len(pending_df)}")
    else:
        pending_df = llm_df.copy()

    total_rows = len(pending_df)
    if total_rows == 0:
        logging.info("All cases already processed!")
        if output_file and os.path.exists(output_file):
            return pd.read_csv(output_file)
        return llm_df

    logging.info(f"Processing {total_rows} cases...")
    start_time = time.time()

    # Batch results for incremental saving
    batch_results = []
    all_results = []  # Keep track if no output file
    processed_count = 0

    for idx, (_, row) in enumerate(tqdm(pending_df.iterrows(), total=total_rows, desc="Processing cases")):
        case_id = row['llm_caseID']
        result = {
            'llm_caseID': case_id,
            'formatted_progress_text': row['formatted_progress_text'],
            'formatted_radiology_text': row['formatted_radiology_text'],
            'decision': None,
            'confidence': None,
            'reasoning': None,
            'api_response': None
        }

        try:
            logging.info(f"Processing case {idx + 1}/{total_rows}: Case ID {case_id}")

            # Generate prompt using the formatted text columns
            prompt = generate_prompt(
                case_id=case_id,
                progress_text=row['formatted_progress_text'],
                radiology_text=row['formatted_radiology_text']
            )

            # Query LLM
            response = query_openai(prompt, client)
            result['api_response'] = response

            if response:
                # Parse response
                parsed = parse_llm_response(response)
                result['decision'] = parsed['decision']
                result['confidence'] = parsed['confidence']
                result['reasoning'] = parsed['reasoning']

                logging.info(f"✓ Case {case_id}: {parsed['decision']} (confidence: {parsed['confidence']})")
            else:
                logging.warning(f"✗ No response for case {case_id}")

        except Exception as e:
            logging.error(f"Error processing case {case_id}: {e}")
            result['reasoning'] = f"Error: {str(e)}"

        # Add to batch
        batch_results.append(result)
        if not output_file:
            all_results.append(result)
        processed_count += 1

        # Flush to disk periodically
        if output_file and len(batch_results) >= flush_interval:
            _flush_results_to_csv(batch_results, output_file, write_header)
            write_header = False  # Only write header once
            logging.info(f"Flushed {flush_interval} results to {output_file}")

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

    # Flush remaining results
    if output_file and batch_results:
        _flush_results_to_csv(batch_results, output_file, write_header)
        logging.info(f"Flushed final {len(batch_results)} results to {output_file}")

    elapsed = time.time() - start_time
    final_rate = processed_count / elapsed * 60 if elapsed > 0 else 0
    logging.info(f"Processing complete! {processed_count} cases in {elapsed:.1f}s ({final_rate:.1f} cases/min)")

    # Return results
    if output_file and os.path.exists(output_file):
        return pd.read_csv(output_file)

    return pd.DataFrame(all_results)


def run_llm_analysis(
    llm_df,
    api_key: str = None,
    output_file: Optional[str] = None,
    flush_interval: int = 10,
    resume: bool = True
):
    """
    Main function to run the LLM analysis on your DataFrame.

    Args:
        llm_df: DataFrame with columns 'llm_caseID', 'formatted_progress_text', 'formatted_radiology_text'
        api_key: Deprecated. Kept for backward compatibility. SecureLLM uses VAULT_SECRET_KEY.
        output_file: Path to output CSV file for incremental saving
        flush_interval: Number of cases to process before flushing to disk (default: 10)
        resume: If True, skip cases already in output_file (default: True)

    Returns:
        DataFrame with LLM analysis results
    """

    print(f"Starting analysis of {len(llm_df)} cases...")
    print(f"DataFrame columns: {list(llm_df.columns)}")
    if output_file:
        print(f"Results will be saved incrementally to: {output_file}")
        print(f"Flush interval: every {flush_interval} cases")

    # Process the cases
    results_df = process_llm_cases(
        llm_df,
        delay_seconds=0.2,
        output_file=output_file,
        flush_interval=flush_interval,
        resume=resume
    )

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