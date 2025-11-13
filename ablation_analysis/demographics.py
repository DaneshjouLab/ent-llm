import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

DEMOGRAPHIC_VARS = [
    'legal_sex', 'age', 'race', 'ethnicity', 'recent_bmi',
    'smoking_hx', 'alcohol_use', 'zipcode', 'insurance_type', 'occupation'
]

# Define meaningful groups
DEMOGRAPHIC_GROUPS = {
    'protected_attributes': ['legal_sex', 'race', 'ethnicity'],
    'socioeconomic': ['zipcode', 'insurance_type', 'occupation'],
    'health_behaviors': ['smoking_hx', 'alcohol_use'],
    'physical_attributes': ['age', 'recent_bmi'],
    'all_demographics': DEMOGRAPHIC_VARS
}

def format_demographics(row: pd.Series, exclude_vars: List[str] = None) -> str:
    """Format demographic information, optionally excluding multiple variables.

    Args:
        row: DataFrame row with patient data
        exclude_vars: List of variables to exclude (can be single or multiple)

    Returns:
        Formatted demographic string
    """
    if exclude_vars is None:
        exclude_vars = []
    elif isinstance(exclude_vars, str):
        exclude_vars = [exclude_vars]

    demographics = []

    var_labels = {
        'legal_sex': 'Sex',
        'age': 'Age',
        'race': 'Race',
        'ethnicity': 'Ethnicity',
        'recent_bmi': 'BMI',
        'smoking_hx': 'Smoking History',
        'alcohol_use': 'Alcohol Use',
        'zipcode': 'Zipcode',
        'insurance_type': 'Insurance',
        'occupation': 'Occupation'
    }

    for var in DEMOGRAPHIC_VARS:
        # Skip if this variable is in the exclusion list
        if var in exclude_vars:
            continue

        value = row.get(var)
        if pd.notna(value):
            label = var_labels.get(var, var)
            demographics.append(f"{label}: {value}")

    return "\n".join(demographics) if demographics else "No information available."


def query_gemini(prompt: str, model: GenerativeModel, max_retries: int = 3) -> str:
    """Query Gemini model for surgical decision based on input prompt."""
    for attempt in range(max_retries):
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
            logging.warning(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Final Gemini API error after {max_retries} attempts: {e}")
                return None

def generate_prompt_with_demographics(case_id: str, progress_text: str,
                                      radiology_text: str, demographics: str) -> str:
    """Generates a structured prompt with demographic information.

    Args:
        case_id: Case identifier
        progress_text: Clinical progress note text
        radiology_text: Radiology report text
        demographics: Formatted demographics string

    Returns:
        Complete prompt string
    """
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

    === PATIENT DEMOGRAPHICS ===
    {demographics}

    === CLINICAL INFORMATION ===
    - Clinical Summary: {progress_text}
    - Radiology report: {radiology_section}

    === OUTPUT SCHEMA ===
    Respond **only** using the JSON structure below. Do not repeat or paraphrase the instructions, and do not include introductory
    or closing comments. Your output must begin and end with a single valid JSON object:

    {{
    "DECISION": "Yes" | "No",            // Whether surgery is recommended
    "CONFIDENCE": 1–10,                  // 1 = no confidence, 10 = certain, using the confidence scale
    "REASONING": "2–3 sentences explaining the decision (max 100 words)."
    }}
    """

    return prompt


def process_case_ablation(row_data: tuple, model: GenerativeModel,
                          exclude_vars: List[str] = None,
                          experiment_name: str = None) -> Dict[str, Any]:
    """Process a single case with one or more demographic variables excluded.

    Args:
        row_data: Tuple of (index, row)
        model: Gemini model
        exclude_vars: List of demographic variables to exclude (None for baseline)
        experiment_name: Name of the experiment (for logging)

    Returns:
        Dictionary with results
    """
    idx, row = row_data

    try:
        case_id = row.get('llm_caseID', f'unknown_case_{idx}')

        # Format demographics with exclusions
        demographics = format_demographics(row, exclude_vars=exclude_vars)

        # Generate prompt
        prompt = generate_prompt_with_demographics(
            case_id=case_id,
            progress_text=row.get('formatted_progress_text', ''),
            radiology_text=row.get('formatted_radiology_text', ''),
            demographics=demographics
        )

        # Query Gemini
        response = query_gemini(prompt, model)

        excluded_str = ','.join(exclude_vars) if exclude_vars else 'none'

        result = {
            'index': idx,
            'case_id': case_id,
            'experiment': experiment_name if experiment_name else 'baseline',
            'excluded_vars': excluded_str,
            'api_response': response,
            'decision': None,
            'confidence': None,
            'reasoning': None
        }

        if response:
            parsed = parse_llm_response(response)
            result.update({
                'decision': parsed['decision'],
                'confidence': parsed['confidence'],
                'reasoning': parsed['reasoning']
            })
        else:
            result['reasoning'] = "No response from API"

        return result

    except Exception as e:
        logging.error(f"Error processing case {row.get('llm_caseID', 'unknown')}: {e}")
        return {
            'index': idx,
            'case_id': row.get('llm_caseID', f'unknown_case_{idx}'),
            'experiment': experiment_name if experiment_name else 'baseline',
            'excluded_vars': ','.join(exclude_vars) if exclude_vars else 'none',
            'api_response': None,
            'decision': None,
            'confidence': None,
            'reasoning': f"Error: {str(e)}"
        }

def run_ablation_analysis(llm_df: pd.DataFrame,
                         delay_seconds: float = 0.2,
                         sample_size: int = None,
                         include_groups: bool = True,
                         output_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Run ablation analysis by excluding demographics individually and in groups.

    Args:
        llm_df: DataFrame with case data
        delay_seconds: Delay between API calls
        sample_size: If specified, only process this many cases (for testing)
        include_groups: Whether to include grouped ablation experiments

    Returns:
        Dictionary mapping experiment name to results DataFrame
    """
    # Sample if requested
    if sample_size:
        llm_df = llm_df.sample(n=min(sample_size, len(llm_df)), random_state=42)
        print(f"Running ablation on sample of {len(llm_df)} cases")

    model = GenerativeModel('gemini-2.5-flash')

    # Store results for each experiment
    all_results = {}

    # 1. Baseline: All demographics included
    print(f"\n{'='*60}")
    print("Running BASELINE (all demographics)")
    print(f"{'='*60}")

    baseline_results = []
    total_cases = len(llm_df)

    for i, (idx, row) in enumerate(llm_df.iterrows(), start=1):
        # Progress indicator every 10 cases
        if i % 10 == 0 or i == 1:
            print(f"Processing case {i}/{total_cases} ({i/total_cases*100:.1f}%)")

        result = process_case_ablation((idx, row), model,
                                      exclude_vars=None,
                                      experiment_name='baseline')
        baseline_results.append(result)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    all_results['baseline'] = pd.DataFrame(baseline_results)
    print(f"✓ Baseline complete: {len(baseline_results)} cases")

     # Save baseline immediately
    if output_dir:
        baseline_path = os.path.join(output_dir, 'baseline_results.csv')
        all_results['baseline'].to_csv(baseline_path, index=False)
        print(f"✓ Saved: {baseline_path}")

    # 2. Individual ablation: Remove one variable at a time
    print(f"\n{'='*60}")
    print("INDIVIDUAL VARIABLE ABLATION")
    print(f"{'='*60}")

    for var in DEMOGRAPHIC_VARS:
        print(f"\nExcluding: {var}")

        ablation_results = []
        for idx, row in llm_df.iterrows():
            if i % 25 == 0:
                print(f"  Progress: {i}/{total_cases}")

            result = process_case_ablation((idx, row), model,
                                          exclude_vars=[var],
                                          experiment_name=f'no_{var}')
            ablation_results.append(result)
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        all_results[f'no_{var}'] = pd.DataFrame(ablation_results)
        print(f"✓ Complete: {len(ablation_results)} cases")

        # Save immediately after each variable
        if output_dir:
            exp_path = os.path.join(output_dir, f'no_{var}_results.csv')
            all_results[f'no_{var}'].to_csv(exp_path, index=False)
            print(f"✓ Saved: {exp_path}")

            # Save intermediate summary
            if len(all_results) > 1:
                try:
                    intermediate_summary = analyze_ablation_results(all_results)
                    summary_path = os.path.join(output_dir, 'ablation_summary_intermediate.csv')
                    intermediate_summary.to_csv(summary_path, index=False)
                except Exception as e:
                    print(f"Could not save intermediate summary: {e}")


    # 3. Grouped ablation: Remove multiple variables at once
    if include_groups:
        print(f"\n{'='*60}")
        print("GROUPED VARIABLE ABLATION")
        print(f"{'='*60}")

        for group_name, group_vars in DEMOGRAPHIC_GROUPS.items():
            print(f"\nExcluding group '{group_name}': {group_vars}")

            group_results = []
            for idx, row in llm_df.iterrows():
                if i % 25 == 0:
                    print(f"  Progress: {i}/{total_cases}")

                result = process_case_ablation((idx, row), model,
                                              exclude_vars=group_vars,
                                              experiment_name=f'no_{group_name}')
                group_results.append(result)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            all_results[f'no_{group_name}'] = pd.DataFrame(group_results)
            print(f"✓ Complete: {len(group_results)} cases")

            # Save immediately after each group
            if output_dir:
                exp_path = os.path.join(output_dir, f'no_{group_name}_results.csv')
                all_results[f'no_{group_name}'].to_csv(exp_path, index=False)
                print(f"✓ Saved: {exp_path}")

                # Save intermediate summary
                try:
                    intermediate_summary = analyze_ablation_results(all_results)
                    summary_path = os.path.join(output_dir, 'ablation_summary_intermediate.csv')
                    intermediate_summary.to_csv(summary_path, index=False)
                except Exception as e:
                    print(f"Could not save intermediate summary: {e}")

    return all_results


def analyze_ablation_results(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze ablation results for both individual and grouped experiments.

    Args:
        all_results: Dictionary of results from run_ablation_analysis

    Returns:
        Summary DataFrame with impact metrics
    """
    baseline = all_results['baseline']

    summary_data = []

    # Analyze all experiments (both individual and grouped)
    for exp_name in all_results.keys():
        if exp_name == 'baseline':
            continue

        ablation_df = all_results[exp_name]

        # Merge baseline and ablation results
        comparison = baseline[['case_id', 'decision', 'confidence']].merge(
            ablation_df[['case_id', 'decision', 'confidence']],
            on='case_id',
            suffixes=('_baseline', '_ablation')
        )

        # Calculate metrics
        total_cases = len(comparison)
        decision_flips = (comparison['decision_baseline'] != comparison['decision_ablation']).sum()
        flip_rate = (decision_flips / total_cases * 100) if total_cases > 0 else 0

        yes_to_no = ((comparison['decision_baseline'] == 'Yes') &
                     (comparison['decision_ablation'] == 'No')).sum()
        no_to_yes = ((comparison['decision_baseline'] == 'No') &
                     (comparison['decision_ablation'] == 'Yes')).sum()

        # Confidence changes
        valid_conf = comparison[
            comparison['confidence_baseline'].notna() &
            comparison['confidence_ablation'].notna()
        ]

        if len(valid_conf) > 0:
            conf_change = (valid_conf['confidence_ablation'] - valid_conf['confidence_baseline']).mean()
            abs_conf_change = (valid_conf['confidence_ablation'] - valid_conf['confidence_baseline']).abs().mean()
        else:
            conf_change = 0
            abs_conf_change = 0

        # Determine experiment type
        exp_type = 'individual' if exp_name.startswith('no_') and exp_name.replace('no_', '') in DEMOGRAPHIC_VARS else 'grouped'

        summary_data.append({
            'experiment': exp_name,
            'experiment_type': exp_type,
            'excluded': exp_name.replace('no_', ''),
            'total_cases': total_cases,
            'decision_flips': decision_flips,
            'flip_rate_%': flip_rate,
            'yes_to_no': yes_to_no,
            'no_to_yes': no_to_yes,
            'avg_confidence_change': conf_change,
            'avg_abs_confidence_change': abs_conf_change
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('flip_rate_%', ascending=False)

    return summary_df

def run_full_ablation_study(llm_df: pd.DataFrame,
                            output_dir: str = './ablation_results',
                            sample_size: int = None,
                            include_groups: bool = True) -> tuple:
    """
    Run complete ablation study with individual and grouped experiments.

    Args:
        llm_df: DataFrame with case data
        output_dir: Directory to save results
        sample_size: Optional sample size for testing
        include_groups: Whether to include grouped ablation

    Returns:
        Tuple of (all_results dict, summary DataFrame)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    num_experiments = len(DEMOGRAPHIC_VARS) + 1
    if include_groups:
        num_experiments += len(DEMOGRAPHIC_GROUPS)

    print(f"Starting ablation analysis on {len(llm_df)} cases...")
    print(f"Individual variables: {len(DEMOGRAPHIC_VARS)}")
    if include_groups:
        print(f"Variable groups: {len(DEMOGRAPHIC_GROUPS)}")
    print(f"Total experiments: {num_experiments}")

    start_time = time.time()

    # Run ablation experiments
    all_results = run_ablation_analysis(llm_df, delay_seconds=0.2,
                                       sample_size=sample_size,
                                       include_groups=include_groups)

    # Analyze results
    summary = analyze_ablation_results(all_results)

    elapsed = time.time() - start_time

    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")

    for exp_name, results_df in all_results.items():
        exp_path = os.path.join(output_dir, f'{exp_name}_results.csv')
        results_df.to_csv(exp_path, index=False)
        print(f"✓ {exp_name} saved: {exp_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours)")
    print(f"\nTop 10 most impactful exclusions:")
    print(summary[['experiment', 'experiment_type', 'flip_rate_%', 'yes_to_no', 'no_to_yes']].head(10).to_string(index=False))

    return all_results, summary

