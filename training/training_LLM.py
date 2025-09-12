# Formatting for LLM Training Input
import pandas as pd
import logging
from typing import Dict, Any, Union, List, Tuple, Optional


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

def training_create_llm_dataframe(processed_df: pd.DataFrame, num_training_rows: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create two LLM dataframes: one with surgery information (training) and one without (rest of data).

    Args:
        processed_df: DataFrame with 'radiology_reports', 'last_progress_note_censored', and 'had_surgery' columns
        num_training_rows: Number of rows for training data with surgery info (default: None means all rows)

    Returns:
        tuple: (training_df, other_df) where:
        - training_df: DataFrame with llm_caseID, formatted_radiology_text, formatted_progress_text, had_surgery
        - test_df: DataFrame with llm_caseID, formatted_radiology_text, formatted_progress_text (no had_surgery)
    """
    # Initialize columns for formatted text
    formatted_radiology = []
    formatted_progress = []
    surgery_info = []

    # Determine how many training rows to process
    if num_training_rows is None:
        num_training_rows = len(processed_df)
    else:
        num_training_rows = min(num_training_rows, len(processed_df))

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

        # Extract surgery information for training rows
        if idx < num_training_rows:
            had_surgery = True if row.get('had_surgery', True) else False
            surgery_info.append(had_surgery)

    # Create training DataFrame (with surgery info)
    training_df = pd.DataFrame({
        'llm_caseID': range(1, num_training_rows + 1),
        'formatted_radiology_text': formatted_radiology[:num_training_rows],
        'formatted_progress_text': formatted_progress[:num_training_rows],
        'had_surgery': surgery_info
    })

    # Create test DataFrame (without surgery info) 
    test_df = pd.DataFrame({
        'llm_caseID': range(num_training_rows + 1, len(processed_df) + 1),
        'formatted_radiology_text': formatted_radiology[num_training_rows:],
        'formatted_progress_text': formatted_progress[num_training_rows:]
    })

    return training_df, test_df

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

def generate_training_examples(sample_cases: pd.DataFrame) -> str:
    """Generate training examples from sample cases."""
    training_examples = ""
    
    for _, row in sample_cases.iterrows():
        case_id = row['llm_caseID']
        progress_text = row['formatted_progress_text']
        radiology_text = row['formatted_radiology_text']
        had_surgery = "Yes" if row['had_surgery'] else "No"
        
        # Check if radiology text is available
        has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
        radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."
        
        training_examples += f"""
        --- Example Case Details ---
        - Example Case ID: {case_id}:
        - Clinical Summary from ENT Notes: {progress_text}
        {radiology_section}
        - Patient had surgery: {had_surgery}
        
        """
    
    return training_examples.strip()

def generate_prompt(case_id: str, progress_text: str, radiology_text: str, sample_cases: pd.DataFrame) -> str:
    """Generates a structured prompt for the LLM."""

    # Generate training examples
    training_examples = generate_training_examples(sample_cases)
    
    # Check if radiology text is available for current case
    has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
    radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."

    prompt = f"""
    You are an expert otolaryngologist evaluating an ENT case.
    Here are a few examples of cases and whether the patient had surgery:

    {training_examples}

    Based ONLY on these examples and the information provided below, make a recommendation on surgery.

    --- Case Details ---
    - Case ID: {case_id}
    - Clinical Summary from ENT Notes: {progress_text}
    {radiology_section}

    ---

    Provide your response as a JSON object with three keys:
    1. "decision": Your recommendation, either "Yes" or "No".
    2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
    3. "reasoning": A brief, 2-4 sentence explanation for your decision.

    Return ONLY the JSON object, no additional text."""

    return prompt

def process_cases_batch(cases_df: pd.DataFrame, sample_cases: pd.DataFrame, client) -> List[Dict]:
    """Process a batch of cases and return results."""
    results = []
    
    for _, row in cases_df.iterrows():
        case_id = str(row['llm_caseID'])
        progress_text = row['formatted_progress_text']
        radiology_text = row['formatted_radiology_text']
        
        # Generate prompt
        prompt = generate_prompt(case_id, progress_text, radiology_text, sample_cases)
        
        # Query OpenAI
        response = query_openai(prompt, client)
        
        result = {
            'case_id': case_id,
            'prompt': prompt,
            'response': response,
            'progress_text': progress_text,
            'radiology_text': radiology_text
        }
        
        results.append(result)

    return results

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


import pandas as pd
import logging
import json
import time
import openai
from typing import Dict, Any, Union, List, Tuple, Optional


class ConversationalLLMAnalyzer:
    """
    LLM analyzer that maintains conversation context to avoid repeating training examples.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.training_loaded = False

    def _make_api_call(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Make API call with error handling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return None

    def load_training_examples(self, training_df: pd.DataFrame) -> bool:
        """Load training examples into conversation context."""

        # System message
        system_message = {
            "role": "system",
            "content": (
                "You are an expert otolaryngologist. I will first provide you with training examples "
                "of ENT cases and their surgical outcomes. After reviewing all examples, you will "
                "evaluate new cases and provide surgical recommendations in JSON format."
            )
        }

        # Generate training examples text
        training_text = "Here are training examples of ENT cases and their surgical outcomes:\n\n"

        for _, row in training_df.iterrows():
            case_id = row['llm_caseID']
            progress_text = row['formatted_progress_text']
            radiology_text = row['formatted_radiology_text']
            had_surgery = "Yes" if row['had_surgery'] else "No"

            # Check if radiology text is available
            has_radiology = (radiology_text and
                           radiology_text.strip() and
                           radiology_text != "No radiology reports available.")

            radiology_section = f"\nRadiology: {radiology_text}" if has_radiology else "\nRadiology: Not available"

            training_text += f"""TRAINING CASE {case_id}:
            Clinical Notes: {progress_text}{radiology_section}
            Surgical Outcome: {had_surgery}

            ---

            """

            training_text += """
            Please confirm you have reviewed these training examples. I will then ask you to evaluate new cases.
            For each new case, provide your response as JSON with:
            1. "decision": Your recommendation, either "Yes" or "No".
            2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
            3. "reasoning": A brief, 2-4 sentence explanation for your decision.

            """

        # Initialize conversation
        self.conversation_history = [
            system_message,
            {"role": "user", "content": training_text}
        ]

        # Get confirmation from model
        response = self._make_api_call(self.conversation_history, max_tokens=200)

        if response:
            self.conversation_history.append({"role": "assistant", "content": response})
            self.training_loaded = True
            logging.info(f"Training examples loaded successfully. Model response: {response[:100]}...")
            return True
        else:
            logging.error("Failed to load training examples")
            return False

    def evaluate_case(self, case_id: str, progress_text: str, radiology_text: str) -> Dict[str, Any]:
        """Evaluate a single case using the loaded training context."""

        if not self.training_loaded:
            raise RuntimeError("Training examples must be loaded first")

        # Check if radiology text is available
        has_radiology = (radiology_text and
                        radiology_text.strip() and
                        radiology_text != "No radiology reports available.")

        radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available"

        case_prompt = f"""
        Based on training data and the information provided below, make a recommendation on surgery.

        --- Case Details ---
        - Case ID: {case_id}
        - Clinical Summary from ENT Notes: {progress_text}
        {radiology_section}

        ---

        Provide your response as a JSON object with three keys:
        1. "decision": Your recommendation, either "Yes" or "No".
        2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
        3. "reasoning": A brief, 2-4 sentence explanation for your decision.

        Return ONLY the JSON object, no additional text."""

        # Add case to conversation
        messages = self.conversation_history + [{"role": "user", "content": case_prompt}]

        # Get response
        response = self._make_api_call(messages, max_tokens=300)

        if response:
            # Parse the response
            parsed = self.parse_llm_response(response)

            # Add to conversation history (but keep it manageable)
            self.conversation_history.append({"role": "user", "content": case_prompt})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Trim conversation history if it gets too long (keep system + training + last few exchanges)
            if len(self.conversation_history) > 50:  # Adjust based on your needs
                # Keep system message, training examples, and recent exchanges
                self.conversation_history = (
                    self.conversation_history[:3] +  # System + training + confirmation
                    self.conversation_history[-20:]   # Recent exchanges
                )

            return {
                'decision': parsed['decision'],
                'confidence': parsed['confidence'],
                'reasoning': parsed['reasoning'],
                'raw_response': response
            }
        else:
            return {
                'decision': None,
                'confidence': None,
                'reasoning': 'API call failed',
                'raw_response': None
            }

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract decision, confidence, and reasoning."""
        default_response = {
            'decision': None,
            'confidence': None,
            'reasoning': 'Failed to parse response'
        }

        if not response:
            return default_response

        try:
            # Clean up response
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


def process_llm_cases_conversational(test_df: pd.DataFrame, training_df: pd.DataFrame,
                                   api_key: str, delay_seconds: float = 1.0) -> pd.DataFrame:
    """
    Process cases using conversational context approach.

    Args:
        test_df: DataFrame with test cases
        training_df: DataFrame with training examples
        api_key: OpenAI API key
        delay_seconds: Delay between API calls

    Returns:
        DataFrame with analysis results
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize analyzer
    analyzer = ConversationalLLMAnalyzer(api_key)

    # Load training examples
    logging.info(f"Loading {len(training_df)} training examples...")
    if not analyzer.load_training_examples(training_df):
        raise RuntimeError("Failed to load training examples")

    # Create result dataframe
    result_df = test_df.copy()
    result_df['decision'] = None
    result_df['confidence'] = None
    result_df['reasoning'] = None
    result_df['api_response'] = None

    total_rows = len(result_df)
    logging.info(f"Processing {total_rows} test cases...")

    # Process each case
    for idx, row in result_df.iterrows():
        try:
            case_id = str(row['llm_caseID'])
            logging.info(f"Processing case {idx + 1}/{total_rows}: Case ID {case_id}")

            # Evaluate case
            result = analyzer.evaluate_case(
                case_id=case_id,
                progress_text=row['formatted_progress_text'],
                radiology_text=row['formatted_radiology_text']
            )

            # Store results
            result_df.at[idx, 'decision'] = result['decision']
            result_df.at[idx, 'confidence'] = result['confidence']
            result_df.at[idx, 'reasoning'] = result['reasoning']
            result_df.at[idx, 'api_response'] = result['raw_response']

            if result['decision']:
                logging.info(f"✓ Case {case_id}: {result['decision']} (confidence: {result['confidence']})")
            else:
                logging.warning(f"✗ No valid response for case {case_id}")

            # Rate limiting
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        except Exception as e:
            logging.error(f"Error processing case {case_id}: {e}")
            result_df.at[idx, 'reasoning'] = f"Error: {str(e)}"

    logging.info("Processing complete!")
    return result_df


def run_llm_analysis_training(test_df: pd.DataFrame, training_df: pd.DataFrame, api_key: str):
    """
    Main function to run conversational LLM analysis.

    Args:
        test_df: DataFrame with test cases
        training_df: DataFrame with training examples
        api_key: OpenAI API key

    Returns:
        DataFrame with LLM analysis results
    """

    print(f"Starting conversational analysis of {len(test_df)} test cases...")
    print(f"Using {len(training_df)} training examples (loaded once)...")
    print(f"Test DataFrame columns: {list(test_df.columns)}")
    print(f"Training DataFrame columns: {list(training_df.columns)}")

    # Validate required columns
    required_training_cols = ['llm_caseID', 'formatted_progress_text', 'formatted_radiology_text', 'had_surgery']
    missing_cols = [col for col in required_training_cols if col not in training_df.columns]
    if missing_cols:
        raise ValueError(f"Training DataFrame missing required columns: {missing_cols}")

    # Process cases
    results_df = process_llm_cases_conversational(test_df, training_df, api_key, delay_seconds=1.0)

    # Summary
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
    print(f"Success rate: {successful_cases/total_cases*100:.1f}%")

    return results_df

results = run_llm_analysis_training(test_df, training_df, api_key)