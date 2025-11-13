import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import os


def stratified_ablation_analysis(baseline_results: pd.DataFrame,
                                 ablation_results: pd.DataFrame,
                                 experiment_name: str) -> Dict:
    """
    Analyze ablation results stratified by baseline decision.
    Tests for asymmetric effects (e.g., removing race affects Yes cases differently than No cases).
    
    Args:
        baseline_results: DataFrame with baseline decisions
        ablation_results: DataFrame with ablation experiment decisions
        experiment_name: Name of the experiment (e.g., 'no_race')
    
    Returns:
        Dictionary with stratified metrics
    """
    # Merge baseline and ablation on case_id
    merged = baseline_results[['case_id', 'decision', 'confidence']].merge(
        ablation_results[['case_id', 'decision', 'confidence']],
        on='case_id',
        suffixes=('_baseline', '_ablation')
    )
    
    # Remove cases with missing decisions
    merged = merged[
        merged['decision_baseline'].notna() & 
        merged['decision_ablation'].notna()
    ].copy()
    
    # Split by baseline decision
    yes_baseline = merged[merged['decision_baseline'] == 'Yes'].copy()
    no_baseline = merged[merged['decision_baseline'] == 'No'].copy()
    
    # Calculate flip rates for each stratum
    yes_to_no = ((yes_baseline['decision_baseline'] == 'Yes') & 
                 (yes_baseline['decision_ablation'] == 'No')).sum()
    yes_to_no_rate = (yes_to_no / len(yes_baseline) * 100) if len(yes_baseline) > 0 else 0
    
    no_to_yes = ((no_baseline['decision_baseline'] == 'No') & 
                 (no_baseline['decision_ablation'] == 'Yes')).sum()
    no_to_yes_rate = (no_to_yes / len(no_baseline) * 100) if len(no_baseline) > 0 else 0
    
    # Calculate asymmetry
    asymmetry = abs(yes_to_no_rate - no_to_yes_rate)
    
    # Statistical test for asymmetry (chi-square test)
    # H0: Flip rates are equal in both directions
    contingency_table = np.array([
        [yes_to_no, len(yes_baseline) - yes_to_no],
        [no_to_yes, len(no_baseline) - no_to_yes]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Determine direction of bias
    if yes_to_no_rate > no_to_yes_rate:
        bias_direction = "AWAY from surgery"
        dominant_flip = "Yes→No"
    elif no_to_yes_rate > yes_to_no_rate:
        bias_direction = "TOWARD surgery"
        dominant_flip = "No→Yes"
    else:
        bias_direction = "No bias (symmetric)"
        dominant_flip = "Balanced"
    
    return {
        'experiment': experiment_name,
        'n_baseline_yes': len(yes_baseline),
        'n_baseline_no': len(no_baseline),
        'yes_to_no_flips': yes_to_no,
        'yes_to_no_rate_%': yes_to_no_rate,
        'no_to_yes_flips': no_to_yes,
        'no_to_yes_rate_%': no_to_yes_rate,
        'asymmetry_%': asymmetry,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'bias_direction': bias_direction,
        'dominant_flip': dominant_flip,
        'significant_asymmetry': p_value < 0.05
    }


def analyze_confidence_shifts(baseline_results: pd.DataFrame,
                              ablation_results: pd.DataFrame,
                              experiment_name: str) -> Dict:
    """
    Analyze confidence changes as primary outcome (instead of binary decision flips).
    Even if decisions don't flip, systematic confidence changes indicate influence.
    
    Args:
        baseline_results: DataFrame with baseline decisions and confidence
        ablation_results: DataFrame with ablation experiment decisions and confidence
        experiment_name: Name of the experiment (e.g., 'no_race')
    
    Returns:
        Dictionary with confidence shift metrics
    """
    # Merge baseline and ablation
    merged = baseline_results[['case_id', 'decision', 'confidence']].merge(
        ablation_results[['case_id', 'decision', 'confidence']],
        on='case_id',
        suffixes=('_baseline', '_ablation')
    )
    
    # Filter to cases with valid confidence scores
    valid_conf = merged[
        merged['confidence_baseline'].notna() & 
        merged['confidence_ablation'].notna()
    ].copy()
    
    if len(valid_conf) == 0:
        return {
            'experiment': experiment_name,
            'n_cases': 0,
            'mean_conf_change': None,
            'error': 'No valid confidence scores'
        }
    
    # Calculate confidence changes
    valid_conf['conf_delta'] = valid_conf['confidence_ablation'] - valid_conf['confidence_baseline']
    valid_conf['abs_conf_delta'] = valid_conf['conf_delta'].abs()
    
    # Overall statistics
    mean_change = valid_conf['conf_delta'].mean()
    std_change = valid_conf['conf_delta'].std()
    mean_abs_change = valid_conf['abs_conf_delta'].mean()
    
    # Effect size (Cohen's d)
    effect_size = mean_change / std_change if std_change > 0 else 0
    
    # Test if mean change differs from zero
    t_stat, p_value = stats.ttest_1samp(valid_conf['conf_delta'], 0)
    
    # Categorize changes
    large_increases = (valid_conf['conf_delta'] >= 2).sum()  # Confidence increased by 2+ points
    large_decreases = (valid_conf['conf_delta'] <= -2).sum()  # Confidence decreased by 2+ points
    minimal_change = (valid_conf['abs_conf_delta'] < 1).sum()  # Changed by less than 1 point
    
    # Stratify by baseline decision
    yes_cases = valid_conf[valid_conf['decision_baseline'] == 'Yes']
    no_cases = valid_conf[valid_conf['decision_baseline'] == 'No']
    
    yes_mean_change = yes_cases['conf_delta'].mean() if len(yes_cases) > 0 else None
    no_mean_change = no_cases['conf_delta'].mean() if len(no_cases) > 0 else None
    
    # Determine interpretation
    if abs(effect_size) < 0.2:
        magnitude = "negligible"
    elif abs(effect_size) < 0.5:
        magnitude = "small"
    elif abs(effect_size) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    if mean_change > 0:
        direction = "INCREASED confidence (pushes toward surgery)"
    elif mean_change < 0:
        direction = "DECREASED confidence (pushes away from surgery)"
    else:
        direction = "No systematic change"
    
    return {
        'experiment': experiment_name,
        'n_cases': len(valid_conf),
        'mean_conf_change': mean_change,
        'std_conf_change': std_change,
        'mean_abs_conf_change': mean_abs_change,
        'effect_size_cohens_d': effect_size,
        'effect_magnitude': magnitude,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'direction': direction,
        'large_increases_n': large_increases,
        'large_increases_%': (large_increases / len(valid_conf) * 100),
        'large_decreases_n': large_decreases,
        'large_decreases_%': (large_decreases / len(valid_conf) * 100),
        'minimal_change_n': minimal_change,
        'minimal_change_%': (minimal_change / len(valid_conf) * 100),
        'yes_baseline_mean_change': yes_mean_change,
        'no_baseline_mean_change': no_mean_change
    }


def run_comprehensive_ablation_analysis(all_results: Dict[str, pd.DataFrame],
                                       output_dir: str = './ablation_results') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run both stratified and confidence-based analyses on ablation results.
    
    Args:
        all_results: Dictionary from run_ablation_analysis() containing all experiment results
        output_dir: Directory to save results
    
    Returns:
        Tuple of (stratified_summary_df, confidence_summary_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    baseline = all_results['baseline']
    
    stratified_results = []
    confidence_results = []
    
    print(f"\n{'='*70}")
    print("RUNNING COMPREHENSIVE ABLATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Baseline cases: {len(baseline)}")
    print(f"Experiments to analyze: {len(all_results) - 1}")
    
    for exp_name in all_results.keys():
        if exp_name == 'baseline':
            continue
        
        print(f"\nAnalyzing: {exp_name}")
        ablation_df = all_results[exp_name]
        
        # Run stratified analysis
        stratified = stratified_ablation_analysis(baseline, ablation_df, exp_name)
        stratified_results.append(stratified)
        
        # Run confidence shift analysis
        confidence = analyze_confidence_shifts(baseline, ablation_df, exp_name)
        confidence_results.append(confidence)
    
    # Create summary DataFrames
    stratified_df = pd.DataFrame(stratified_results)
    confidence_df = pd.DataFrame(confidence_results)
    
    # Sort by asymmetry and effect size
    stratified_df = stratified_df.sort_values('asymmetry_%', ascending=False)
    confidence_df = confidence_df.sort_values('effect_size_cohens_d', 
                                             key=lambda x: x.abs(), 
                                             ascending=False)
    
    # Save results
    stratified_path = os.path.join(output_dir, 'stratified_analysis.csv')
    confidence_path = os.path.join(output_dir, 'confidence_shift_analysis.csv')
    
    stratified_df.to_csv(stratified_path, index=False)
    confidence_df.to_csv(confidence_path, index=False)
    
    print(f"\n{'='*70}")
    print("STRATIFIED ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"✓ Saved: {stratified_path}")
    
    # Print top asymmetric effects
    sig_asymmetric = stratified_df[stratified_df['significant_asymmetry'] == True]
    if len(sig_asymmetric) > 0:
        print(f"\n  {len(sig_asymmetric)} variables show SIGNIFICANT ASYMMETRIC effects:")
        for _, row in sig_asymmetric.head(5).iterrows():
            print(f"\n  {row['experiment']}:")
            print(f"    Yes→No: {row['yes_to_no_rate_%']:.2f}% | No→Yes: {row['no_to_yes_rate_%']:.2f}%")
            print(f"    Asymmetry: {row['asymmetry_%']:.2f}% (p={row['p_value']:.4f})")
            print(f"    Bias: {row['bias_direction']}")
    else:
        print("\n✓ No significant asymmetric effects detected")
    
    print(f"\n{'='*70}")
    print("CONFIDENCE SHIFT ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"✓ Saved: {confidence_path}")
    
    # Print significant confidence shifts
    sig_confidence = confidence_df[
        (confidence_df['significant'] == True) & 
        (confidence_df['effect_size_cohens_d'].abs() >= 0.2)
    ]
    
    if len(sig_confidence) > 0:
        print(f"\n {len(sig_confidence)} variables show SIGNIFICANT confidence shifts:")
        for _, row in sig_confidence.head(5).iterrows():
            print(f"\n  {row['experiment']}:")
            print(f"    Mean change: {row['mean_conf_change']:.2f} points (p={row['p_value']:.4f})")
            print(f"    Effect size: {row['effect_size_cohens_d']:.3f} ({row['effect_magnitude']})")
            print(f"    Direction: {row['direction']}")
    else:
        print("\n  No significant confidence shifts detected")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    avg_asymmetry = stratified_df['asymmetry_%'].mean()
    max_asymmetry = stratified_df['asymmetry_%'].max()
    
    avg_effect_size = confidence_df['effect_size_cohens_d'].abs().mean()
    max_effect_size = confidence_df['effect_size_cohens_d'].abs().max()
    
    print(f"Average asymmetry: {avg_asymmetry:.2f}%")
    print(f"Max asymmetry: {max_asymmetry:.2f}%")
    print(f"Average confidence effect size: {avg_effect_size:.3f}")
    print(f"Max confidence effect size: {max_effect_size:.3f}")
    
    return stratified_df, confidence_df


# Example Usage
if __name__ == "__main__":
    
    # Define experiments
    experiment_files = {
        'baseline': 'baseline_results.csv',
        'no_zipcode': 'no_zipcode_results.csv',
        'no_protected_attributes': 'no_protected_attributes_results.csv',
        'no_age': 'no_age_results.csv',
        'no_smoking_hx': 'no_smoking_hx_results.csv',
        'no_socioeconomic': 'no_socioeconomic_results.csv',
        'no_health_behaviors': 'no_health_behaviors_results.csv',
        'no_all_demographics': 'no_all_demographics_results.csv',
        'no_legal_sex': 'no_legal_sex_results.csv',
        'no_occupation': 'no_occupation_results.csv',
        'no_alcohol_use': 'no_alcohol_use_results.csv',
        'no_insurance_type': 'no_insurance_type_results.csv',
        'no_race': 'no_race_results.csv',
        'no_ethnicity': 'no_ethnicity_results.csv',
        'no_recent_bmi': 'no_recent_bmi_results.csv',
        'no_physical_attributes': 'no_physical_attributes_results.csv'
    }
    
    # Load all your ablation results
    results_dir = './ablation_results_stratified'  
    all_results = {}
    
    print("Loading ablation results...")
    for exp_name, filename in experiment_files.items():
        filepath = os.path.join(results_dir, filename)
        try:
            all_results[exp_name] = pd.read_csv(filepath)
            print(f"✓ Loaded {exp_name}: {len(all_results[exp_name])} cases")
        except FileNotFoundError:
            print(f" File not found: {filepath}")
        except Exception as e:
            print(f" Error loading {exp_name}: {e}")
    
    if 'baseline' not in all_results:
        print("\ ERROR: baseline_results.csv is missing.")
    elif len(all_results) < 2:
        print("\n ERROR: Need at least baseline + 1 experiment to compare")
    else:
        print(f"\n Successfully loaded {len(all_results)} experiments")
        print(f"  Baseline has {len(all_results['baseline'])} cases")
        
        # Run comprehensive analysis
        print("\nRunning comprehensive analysis...")
        stratified_df, confidence_df = run_comprehensive_ablation_analysis(
            all_results=all_results,
            output_dir=results_dir
        )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"✓ Stratified analysis: {results_dir}/stratified_analysis.csv")
        print(f"✓ Confidence analysis: {results_dir}/confidence_shift_analysis.csv")


