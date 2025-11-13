import pandas as pd
import numpy as np
import time
from vertexai.generative_models import GenerativeModel
from typing import Dict, Tuple
import os

def run_test_retest_baseline(llm_df: pd.DataFrame,
                             n_cases: int = 100,
                             delay_seconds: float = 0.5,
                             output_dir: str = './test_retest',
                             use_ablation_sample: bool = False,
                             ablation_baseline_path: str = None) -> Tuple[pd.DataFrame, float]:
    """
    Measure baseline API noise by querying same cases twice with identical prompts.
    Uses your existing functions: format_demographics, generate_prompt_with_demographics, 
    query_gemini, parse_llm_response.
    
    Args:
        llm_df: Your full DataFrame with case data
        n_cases: Number of cases to test (100 is good, 50 minimum)
        delay_seconds: Delay between API calls
        output_dir: Where to save results
        use_ablation_sample: If True, uses the exact cases from your ablation study
        ablation_baseline_path: Path to baseline_results.csv from ablation (if using same sample)
    
    Returns:
        Tuple of (detailed results DataFrame, baseline flip rate %)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model = GenerativeModel('gemini-2.5-flash')
    
    print(f"\n{'='*70}")
    print("TEST-RETEST RELIABILITY CHECK")
    print(f"{'='*70}")
    
    # Determine which sample to use
    if use_ablation_sample and ablation_baseline_path:
        print(f"Using EXACT SAMPLE from ablation study: {ablation_baseline_path}")
        baseline_results = pd.read_csv(ablation_baseline_path)
        case_ids = baseline_results['case_id'].unique()
        sample_df = llm_df[llm_df['llm_caseID'].isin(case_ids)].copy()
        print(f"Matched {len(sample_df)} cases from ablation study")
    else:
        print(f"Testing {n_cases} randomly sampled cases with IDENTICAL prompts")
        sample_df = llm_df.sample(n=min(n_cases, len(llm_df)), random_state=42)
    
    print("(Each case queried twice to measure API randomness)\n")
    
    results = []
    
    for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/{len(sample_df)} ({i/len(sample_df)*100:.1f}%)")
        
        case_id = row.get('llm_caseID', f'case_{idx}')
        
        # Format demographics - ALL included (this is baseline with full info)
        demographics = format_demographics(row, exclude_vars=None)
        
        # Generate the exact same prompt for both queries
        prompt = generate_prompt_with_demographics(
            case_id=case_id,
            progress_text=row.get('formatted_progress_text', ''),
            radiology_text=row.get('formatted_radiology_text', ''),
            demographics=demographics
        )
        
        # Query #1
        response1 = query_gemini(prompt, model)
        time.sleep(delay_seconds)
        
        # Query #2 (IDENTICAL prompt)
        response2 = query_gemini(prompt, model)
        time.sleep(delay_seconds)
        
        # Parse both responses
        parsed1 = parse_llm_response(response1) if response1 else {
            'decision': None, 'confidence': None, 'reasoning': None
        }
        parsed2 = parse_llm_response(response2) if response2 else {
            'decision': None, 'confidence': None, 'reasoning': None
        }
        
        # Check for consistency
        both_valid = (parsed1['decision'] is not None and 
                     parsed2['decision'] is not None)
        
        decision_match = parsed1['decision'] == parsed2['decision'] if both_valid else None
        decision_flip = not decision_match if both_valid else False
        
        # Confidence difference
        conf_diff = None
        if parsed1['confidence'] is not None and parsed2['confidence'] is not None:
            conf_diff = abs(parsed1['confidence'] - parsed2['confidence'])
        
        results.append({
            'case_id': case_id,
            'decision_test1': parsed1['decision'],
            'decision_test2': parsed2['decision'],
            'confidence_test1': parsed1['confidence'],
            'confidence_test2': parsed2['confidence'],
            'decision_match': decision_match,
            'decision_flip': decision_flip,
            'confidence_diff': conf_diff,
            'both_valid': both_valid
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate baseline metrics
    valid_cases = results_df[results_df['both_valid'] == True]
    n_flips = valid_cases['decision_flip'].sum()
    baseline_flip_rate = (n_flips / len(valid_cases) * 100) if len(valid_cases) > 0 else 0
    
    # Confidence stats
    conf_diffs = results_df['confidence_diff'].dropna()
    avg_conf_diff = conf_diffs.mean() if len(conf_diffs) > 0 else 0
    
    # Print results
    print(f"\n{'='*70}")
    print("BASELINE API NOISE RESULTS")
    print(f"{'='*70}")
    print(f"Valid comparisons: {len(valid_cases)}/{len(results_df)}")
    print(f"Decision flips (API noise): {n_flips}")
    print(f"Baseline flip rate: {baseline_flip_rate:.2f}%")
    if len(conf_diffs) > 0:
        print(f"Avg confidence difference: {avg_conf_diff:.2f} points (std: {conf_diffs.std():.2f})")
    
    # Save results
    results_path = os.path.join(output_dir, 'test_retest_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results: {results_path}")
    
    # Save summary
    summary = pd.DataFrame([{
        'n_cases_tested': len(valid_cases),
        'n_decision_flips': int(n_flips),
        'baseline_flip_rate_%': baseline_flip_rate,
        'avg_confidence_diff': avg_conf_diff,
        'std_confidence_diff': conf_diffs.std() if len(conf_diffs) > 0 else None,
        'interpretation': 'high_noise' if baseline_flip_rate >= 5.0 else 
                         'moderate_noise' if baseline_flip_rate >= 3.0 else 'low_noise'
    }])
    
    summary_path = os.path.join(output_dir, 'baseline_noise_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary: {summary_path}")
    
    return results_df, baseline_flip_rate


def compare_to_ablation_results(ablation_summary_path: str,
                                baseline_flip_rate: float):
    """
    Quick comparison of ablation results to baseline noise.
    
    Args:
        ablation_summary_path: Path to your ablation_summary.csv
        baseline_flip_rate: Result from run_test_retest_baseline()
    """
    summary_df = pd.read_csv(ablation_summary_path)
    individual_df = summary_df[summary_df['experiment_type'] == 'individual']
    
    print(f"\n{'='*70}")
    print("ABLATION vs BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"Baseline API noise: {baseline_flip_rate:.2f}%")
    print(f"Average ablation flip rate: {individual_df['flip_rate_%'].mean():.2f}%")
    print(f"Ablation flip rate range: {individual_df['flip_rate_%'].min():.2f}% - {individual_df['flip_rate_%'].max():.2f}%")
    
    # Calculate how many variables exceed baseline by meaningful margin
    threshold_20pct = baseline_flip_rate * 1.2  # 20% above baseline
    threshold_50pct = baseline_flip_rate * 1.5  # 50% above baseline
    
    above_20 = (individual_df['flip_rate_%'] > threshold_20pct).sum()
    above_50 = (individual_df['flip_rate_%'] > threshold_50pct).sum()
    
    print(f"\nVariables exceeding baseline by >20%: {above_20}/{len(individual_df)}")
    print(f"Variables exceeding baseline by >50%: {above_50}/{len(individual_df)}")