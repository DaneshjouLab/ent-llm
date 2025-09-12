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