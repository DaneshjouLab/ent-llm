import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import logging
import time
import pandas as pd
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc

def initialize_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialize Vertex AI with project and location."""
    vertexai.init(project=project_id, location=location)
    logging.info(f"Vertex AI initialized for project: {project_id}, location: {location}")

def query_gemini(prompt: str, model: GenerativeModel) -> str:
    """Query Gemini model for surgical decision based on input prompt."""
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=3000,
            )
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return None

def generate_prompt(case_id: str, progress_text: str, radiology_text: str) -> str:
    """Generates a structured prompt for the LLM."""
    # Check if radiology text is available
    has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
    radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."

    prompt = f"""
    === OBJECTIVE ===
    You are an expert otolaryngologist evaluating an ENT case. 
    Decide **only** whether surgery is recommended based on the information provided.

    === INSTRUCTIONS ===
    1. Rely strictly on the case details below (do not invent information).  
    2. Respond with a single **valid JSON object** — no extra text, headings, or explanations outside the JSON.  
    3. Follow the schema exactly. 
    4. For CONFIDENCE, choose **one integer value (1–10)** from the Confidence Scale. Do not output ranges or text.

    === CONFIDENCE SCALE (1–10) ===
    1 = no confidence (likely wrong)  
    3–4 = low (uncertain, weak support)  
    5 = moderate (plausible but partly speculative)  
    6–7 = fairly confident (reasonable but some gaps/hedging)  
    8 = high (well supported, minor uncertainty)  
    9 = very high (strong reasoning, unlikely error)  
    10 = certain (clear, fully supported, no doubt)

    === CASE DETAILS ===
    - Case ID: {case_id}
    - Clinical Summary: {progress_text}
    {radiology_section}

    === OUTPUT SCHEMA ===
    Respond **only** using the JSON structure below. Do not repeat or paraphrase the instructions, and do not include introductory
    or closing comments. Your output must begin and end with a single valid JSON object:

    {{
    "DECISION": "Yes" | "No",            // Whether surgery is recommended
    "CONFIDENCE": 1–10,                  // 1 = no confidence, 10 = certain
    "REASONING": "2–4 sentences explaining the decision"
    }}
    """

    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract decision, confidence, and reasoning from JSON."""
    result = {
        'decision': None,
        'confidence': None,
        'reasoning': 'Failed to parse response'
    }

    if not response:
        return result

    try:
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision = line.replace('DECISION:', '').strip()
                if decision in ['Yes', 'No']:
                    result['decision'] = decision
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.replace('CONFIDENCE:', '').strip()
                    # Handle cases where confidence might have comma or other text
                    confidence_num = confidence_str.split(',')[0].strip()
                    confidence = int(confidence_num)
                    if 1 <= confidence <= 10:
                        result['confidence'] = confidence
                except ValueError:
                    pass
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()

        return result

    except Exception as e:
        logging.error(f"Error parsing structured response: {e}")
        return result

def process_single_case(row_data: tuple, model: GenerativeModel) -> Dict[str, Any]:
    """Process a single case with Gemini model."""
    idx, row = row_data

    try:
        case_id = row['llm_caseID']

        # Generate prompt using the formatted text columns
        prompt = generate_prompt(
            case_id=case_id,
            progress_text=row['formatted_progress_text'],
            radiology_text=row['formatted_radiology_text']
        )

        # Query Gemini
        response = query_gemini(prompt, model)

        result = {
            'index': idx,
            'case_id': case_id,
            'api_response': response,
            'decision': None,
            'confidence': None,
            'reasoning': None
        }

        if response:
            # Parse response
            parsed = parse_llm_response(response)
            result.update({
                'decision': parsed['decision'],
                'confidence': parsed['confidence'],
                'reasoning': parsed['reasoning']
            })
            logging.info(f"Case {case_id}: {parsed['decision']} (confidence: {parsed['confidence']})")
        else:
            logging.warning(f"No response for case {case_id}")
            result['reasoning'] = "No response from API"

        return result

    except Exception as e:
        logging.error(f"Error processing case {row.get('llm_caseID', 'unknown')}: {e}")
        return {
            'index': idx,
            'case_id': row.get('llm_caseID', 'unknown'),
            'api_response': None,
            'decision': None,
            'confidence': None,
            'reasoning': f"Error: {str(e)}"
        }

def parallel_process_llm_cases(batch_df: pd.DataFrame, model: GenerativeModel, max_workers: int = 5, batch_num: int = None) -> pd.DataFrame:
    """Process a batch of cases in parallel with progress tracking."""
    from tqdm import tqdm
    import time

    # Create a copy of the dataframe and reset index for clean processing
    result_df = batch_df.copy().reset_index(drop=True)

    # Initialize new columns
    result_df['decision'] = None
    result_df['confidence'] = None
    result_df['reasoning'] = None
    result_df['api_response'] = None

    # Prepare data for parallel processing using the reset index
    row_data = [(idx, row) for idx, row in result_df.iterrows()]

    total_cases = len(row_data)
    completed_count = 0
    start_time = time.time()

    # Create progress bar
    batch_desc = f"Batch {batch_num}" if batch_num else "Processing"
    pbar = tqdm(total=total_cases, desc=batch_desc, unit="cases")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(process_single_case, data, model): data
            for data in row_data
        }

        # Collect results with progress tracking
        for future in as_completed(future_to_data):
            try:
                result = future.result()
                idx = result['index']

                result_df.at[idx, 'decision'] = result['decision']
                result_df.at[idx, 'confidence'] = result['confidence']
                result_df.at[idx, 'reasoning'] = result['reasoning']
                result_df.at[idx, 'api_response'] = result['api_response']

                completed_count += 1
                pbar.update(1)

                # Progress updates every 100 cases for large batches
                if completed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed * 60  # cases per minute
                    remaining = total_cases - completed_count
                    eta_minutes = remaining / (rate / 60) if rate > 0 else 0
                    pbar.set_postfix({
                        'rate': f'{rate:.1f}/min',
                        'ETA': f'{eta_minutes:.1f}min'
                    })

            except Exception as e:
                logging.error(f"Error collecting result: {e}")
                completed_count += 1
                pbar.update(1)

    pbar.close()

    # Final rate calculation
    elapsed = time.time() - start_time
    final_rate = total_cases / elapsed * 60 if elapsed > 0 else 0
    print(f"Batch completed: {total_cases} cases in {elapsed:.1f}s ({final_rate:.1f} cases/min)")

    return result_df

def get_completed_case_ids(save_directory: str) -> set:
    """Get IDs of already completed cases from existing pickle files."""
    completed_ids = set()

    if not os.path.exists(save_directory):
        return completed_ids

    for filename in os.listdir(save_directory):
        if filename.endswith('.pkl') and 'gcpgemini' in filename:
            try:
                filepath = os.path.join(save_directory, filename)
                df = pd.read_pickle(filepath)
                if 'llm_caseID' in df.columns:
                    completed_ids.update(df['llm_caseID'].tolist())
            except Exception as e:
                logging.warning(f"Could not load {filename}: {e}")

    return completed_ids

def load_existing_results(save_directory: str) -> pd.DataFrame:
    """Load all existing results from pickle files."""
    all_results = []

    if not os.path.exists(save_directory):
        return pd.DataFrame()

    for filename in os.listdir(save_directory):
        if filename.endswith('.pkl') and 'gcpgemini' in filename:
            try:
                filepath = os.path.join(save_directory, filename)
                df = pd.read_pickle(filepath)
                all_results.append(df)
            except Exception as e:
                logging.warning(f"Could not load {filename}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def run_llm_analysis(llm_df: pd.DataFrame, project_id: str, location: str = "us-central1",
                    model_name: str = "gemini-2.5-flash") -> pd.DataFrame:
    """
    Main function to run LLM analysis on ENT cases using Vertex AI Gemini.

    Args:
        llm_df: DataFrame containing ENT case data
        project_id: GCP project ID
        location: GCP location for Vertex AI
        model_name: Gemini model name

    Returns:
        DataFrame with analysis results
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print(f"Starting analysis of {len(llm_df)} cases using {model_name}...")

    # Validate required columns
    required_columns = ['llm_caseID', 'formatted_progress_text', 'formatted_radiology_text']
    missing_columns = [col for col in required_columns if col not in llm_df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Initialize Vertex AI
    initialize_vertex_ai(project_id, location)

    # Initialize Gemini model
    model = GenerativeModel(model_name)

    # Process the cases with parallel processing
    results_df = parallel_process_llm_cases(llm_df, model, max_workers=5)

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