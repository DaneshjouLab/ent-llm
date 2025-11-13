import pandas as pd
import numpy as np
from scipy import stats
import os


def analyze_sex_bias_direction(baseline_results: pd.DataFrame,
                               no_sex_results: pd.DataFrame,
                               original_data: pd.DataFrame) -> dict:
    """
    Determine which sex is favored for surgical recommendations.
    
    Args:
        baseline_results: Baseline ablation results (with all demographics)
        no_sex_results: Ablation results with sex removed
        original_data: Original patient data with 'legal_sex' column
    
    Returns:
        Dictionary with sex bias analysis
    """
    
    print(f"\n{'='*70}")
    print("SEX BIAS DIRECTION ANALYSIS")
    print(f"{'='*70}")
    
    # Merge all datasets
    merged = baseline_results[['case_id', 'decision']].merge(
        no_sex_results[['case_id', 'decision']],
        on='case_id',
        suffixes=('_baseline', '_no_sex')
    ).merge(
        original_data[['llm_caseID', 'legal_sex']],
        left_on='case_id',
        right_on='llm_caseID',
        how='left'
    )
    
    # Remove cases with missing data
    merged = merged[
        merged['decision_baseline'].notna() & 
        merged['decision_no_sex'].notna() &
        merged['legal_sex'].notna()
    ].copy()
    
    print(f"Total cases analyzed: {len(merged)}")
    
    # Identify flips
    merged['yes_to_no_flip'] = ((merged['decision_baseline'] == 'Yes') & 
                                (merged['decision_no_sex'] == 'No'))
    merged['no_to_yes_flip'] = ((merged['decision_baseline'] == 'No') & 
                                (merged['decision_no_sex'] == 'Yes'))
    merged['any_flip'] = merged['yes_to_no_flip'] | merged['no_to_yes_flip']
    
    # Count by sex
    sex_counts = merged.groupby('legal_sex').agg({
        'case_id': 'count',
        'yes_to_no_flip': 'sum',
        'no_to_yes_flip': 'sum',
        'any_flip': 'sum'
    }).rename(columns={'case_id': 'total_cases'})
    
    # Calculate rates
    sex_counts['yes_to_no_rate_%'] = (sex_counts['yes_to_no_flip'] / sex_counts['total_cases'] * 100)
    sex_counts['no_to_yes_rate_%'] = (sex_counts['no_to_yes_flip'] / sex_counts['total_cases'] * 100)
    sex_counts['flip_rate_%'] = (sex_counts['any_flip'] / sex_counts['total_cases'] * 100)
    sex_counts['asymmetry_%'] = sex_counts['yes_to_no_rate_%'] - sex_counts['no_to_yes_rate_%']
    
    print(f"\n{'Sex':<10} {'N Cases':<10} {'Yes→No':<10} {'No→Yes':<10} {'Asymmetry':<12}")
    print("-" * 70)
    for sex in sex_counts.index:
        print(f"{sex:<10} {int(sex_counts.loc[sex, 'total_cases']):<10} "
              f"{sex_counts.loc[sex, 'yes_to_no_rate_%']:>6.2f}%   "
              f"{sex_counts.loc[sex, 'no_to_yes_rate_%']:>6.2f}%   "
              f"{sex_counts.loc[sex, 'asymmetry_%']:>8.2f}%")
    
    # Baseline decision rates by sex (to understand starting point)
    baseline_yes_by_sex = merged[merged['decision_baseline'] == 'Yes'].groupby('legal_sex').size()
    baseline_no_by_sex = merged[merged['decision_baseline'] == 'No'].groupby('legal_sex').size()
    
    print(f"\n{'='*70}")
    print("BASELINE SURGERY RECOMMENDATION RATES (WITH sex included)")
    print(f"{'='*70}")
    
    for sex in sex_counts.index:
        yes_count = baseline_yes_by_sex.get(sex, 0)
        no_count = baseline_no_by_sex.get(sex, 0)
        total = yes_count + no_count
        yes_rate = (yes_count / total * 100) if total > 0 else 0
        print(f"{sex}: {yes_count}/{total} = {yes_rate:.1f}% recommended surgery")
    
    # Statistical test: Are flip rates different between sexes?
    if len(sex_counts) == 2:
        sexes = list(sex_counts.index)
        sex1, sex2 = sexes[0], sexes[1]
        
        # Chi-square test for Yes→No flips
        yes_to_no_contingency = np.array([
            [sex_counts.loc[sex1, 'yes_to_no_flip'], 
             sex_counts.loc[sex1, 'total_cases'] - sex_counts.loc[sex1, 'yes_to_no_flip']],
            [sex_counts.loc[sex2, 'yes_to_no_flip'], 
             sex_counts.loc[sex2, 'total_cases'] - sex_counts.loc[sex2, 'yes_to_no_flip']]
        ])
        
        chi2_yes_no, p_yes_no = stats.chi2_contingency(yes_to_no_contingency)[:2]
        
        # Chi-square test for asymmetry difference
        asymmetry_contingency = np.array([
            [sex_counts.loc[sex1, 'yes_to_no_flip'], sex_counts.loc[sex1, 'no_to_yes_flip']],
            [sex_counts.loc[sex2, 'yes_to_no_flip'], sex_counts.loc[sex2, 'no_to_yes_flip']]
        ])
        
        chi2_asym, p_asym = stats.chi2_contingency(asymmetry_contingency)[:2]
        
        print(f"\n{'='*70}")
        print("STATISTICAL TESTS")
        print(f"{'='*70}")
        print(f"Yes→No flip rate difference: χ²={chi2_yes_no:.3f}, p={p_yes_no:.4f}")
        print(f"Asymmetry pattern difference: χ²={chi2_asym:.3f}, p={p_asym:.4f}")
        
        if p_asym < 0.05:
            print("\n Sex groups show SIGNIFICANTLY DIFFERENT flip patterns")
        else:
            print("\n Sex groups show similar flip patterns (p>0.05)")
    
    # Determine which sex is favored
    print(f"\n{'='*70}")
    print("INTERPRETATION: WHICH SEX IS FAVORED?")
    print(f"{'='*70}")
    
    # Logic: If removing sex causes MORE Yes→No flips for group X,
    # that means sex information was HELPING group X get surgery
    
    for sex in sex_counts.index:
        yes_to_no = sex_counts.loc[sex, 'yes_to_no_rate_%']
        no_to_yes = sex_counts.loc[sex, 'no_to_yes_rate_%']
        asymmetry = sex_counts.loc[sex, 'asymmetry_%']
        
        print(f"\n{sex} patients:")
        print(f"  When sex is REMOVED:")
        print(f"    • {yes_to_no:.2f}% lose surgery recommendation (Yes→No)")
        print(f"    • {no_to_yes:.2f}% gain surgery recommendation (No→Yes)")
        print(f"    • Net asymmetry: {asymmetry:.2f}%")
        
        if asymmetry > 2.0:
            print(f"  → Being {sex} INCREASES surgery likelihood")
            print(f"  → Model favors {sex} patients for surgical intervention")
        elif asymmetry < -2.0:
            print(f"  → Being {sex} DECREASES surgery likelihood")
            print(f"  → Model disfavors {sex} patients for surgical intervention")
        else:
            print(f"  → Minimal bias (asymmetry < 2%)")
    
    # Overall interpretation
    print(f"\n{'='*70}")
    print("CLINICAL SIGNIFICANCE")
    print(f"{'='*70}")
    
    max_asym_sex = sex_counts['asymmetry_%'].abs().idxmax()
    max_asym_value = sex_counts.loc[max_asym_sex, 'asymmetry_%']
    
    if abs(max_asym_value) > 3.0:
        if max_asym_value > 0:
            print(f" STRONG BIAS: {max_asym_sex} patients are significantly MORE LIKELY")
            print(f"   to be recommended surgery when sex information is included.")
            print(f"   Asymmetry: {max_asym_value:.2f}%")
        else:
            print(f" STRONG BIAS: {max_asym_sex} patients are significantly LESS LIKELY")
            print(f"   to be recommended surgery when sex information is included.")
            print(f"   Asymmetry: {max_asym_value:.2f}%")
    elif abs(max_asym_value) > 1.5:
        print(f"  MODERATE BIAS detected for {max_asym_sex} patients")
        print(f"   Asymmetry: {max_asym_value:.2f}%")
    else:
        print(f" Minimal sex-specific bias detected")
    
    return {
        'sex_counts': sex_counts,
        'merged_data': merged
    }


def analyze_flipped_cases(baseline_results: pd.DataFrame,
                         no_sex_results: pd.DataFrame,
                         original_data: pd.DataFrame,
                         output_dir: str = './ablation_results_stratified'):
    """
    Detailed analysis of specific cases that flipped when sex was removed.
    
    Args:
        baseline_results: Baseline ablation results
        no_sex_results: Results with sex removed
        original_data: Original patient data with all demographics
        output_dir: Where to save detailed case analysis
    """
    
    # Merge datasets
    merged = baseline_results[['case_id', 'decision', 'confidence', 'reasoning']].merge(
        no_sex_results[['case_id', 'decision', 'confidence', 'reasoning']],
        on='case_id',
        suffixes=('_baseline', '_no_sex')
    ).merge(
        original_data[['llm_caseID', 'legal_sex', 'age', 'race', 'ethnicity', 
                      'recent_bmi', 'insurance_type']],
        left_on='case_id',
        right_on='llm_caseID',
        how='left'
    )
    
    # Identify flips
    merged['flip_type'] = 'no_flip'
    merged.loc[
        (merged['decision_baseline'] == 'Yes') & (merged['decision_no_sex'] == 'No'),
        'flip_type'
    ] = 'yes_to_no'
    merged.loc[
        (merged['decision_baseline'] == 'No') & (merged['decision_no_sex'] == 'Yes'),
        'flip_type'
    ] = 'no_to_yes'
    
    # Get flipped cases
    flipped = merged[merged['flip_type'] != 'no_flip'].copy()
    
    print(f"\n{'='*70}")
    print("FLIPPED CASES ANALYSIS")
    print(f"{'='*70}")
    print(f"Total flipped cases: {len(flipped)}")
    
    # Breakdown by sex and flip direction
    flip_summary = flipped.groupby(['legal_sex', 'flip_type']).size().reset_index(name='count')
    print(f"\nFlip breakdown by sex:")
    print(flip_summary.to_string(index=False))
    
    # Save detailed flipped cases
    os.makedirs(output_dir, exist_ok=True)
    
    flipped_path = os.path.join(output_dir, 'flipped_cases_detailed.csv')
    flipped.to_csv(flipped_path, index=False)
    print(f"\n✓ Detailed flipped cases saved: {flipped_path}")
    
    # Summary by sex
    summary_by_sex = flipped.groupby('legal_sex').agg({
        'case_id': 'count',
        'confidence_baseline': 'mean',
        'confidence_no_sex': 'mean'
    }).rename(columns={'case_id': 'n_flips'})
    
    summary_by_sex['conf_change'] = (summary_by_sex['confidence_no_sex'] - 
                                     summary_by_sex['confidence_baseline'])
    
    print(f"\nConfidence changes in flipped cases:")
    print(summary_by_sex.to_string())
    
    return flipped


def extract_male_yes_to_no_flips(baseline_results: pd.DataFrame,
                                 no_sex_results: pd.DataFrame,
                                 original_data: pd.DataFrame,
                                 output_dir: str = './ablation_results_stratified') -> pd.DataFrame:
    """
    Extract and analyze the specific male cases that flipped from Yes→No when sex was removed.
    
    Args:
        baseline_results: Baseline ablation results
        no_sex_results: Results with sex removed
        original_data: Original patient data
        output_dir: Where to save results
    
    Returns:
        DataFrame with the 14 male Yes→No cases
    """
    
    print(f"\n{'='*70}")
    print("EXTRACTING MALE YES→NO FLIP CASES")
    print(f"{'='*70}")
    
    # Merge all data
    merged = baseline_results[['case_id', 'decision', 'confidence', 'reasoning']].merge(
        no_sex_results[['case_id', 'decision', 'confidence', 'reasoning']],
        on='case_id',
        suffixes=('_baseline', '_no_sex')
    ).merge(
        original_data,
        left_on='case_id',
        right_on='llm_caseID',
        how='left'
    )
    
    # Filter to male Yes→No flips only
    male_yes_to_no = merged[
        (merged['legal_sex'] == 'Male') &
        (merged['decision_baseline'] == 'Yes') &
        (merged['decision_no_sex'] == 'No')
    ].copy()
    
    print(f"Found {len(male_yes_to_no)} male cases that flipped Yes→No")
    
    if len(male_yes_to_no) == 0:
        print("  No male Yes→No flips found!")
        return pd.DataFrame()
    
    # Calculate confidence change
    male_yes_to_no['confidence_change'] = (male_yes_to_no['confidence_no_sex'] - 
                                           male_yes_to_no['confidence_baseline'])
    
    # Sort by confidence change (most concerning flips first)
    male_yes_to_no = male_yes_to_no.sort_values('confidence_change', ascending=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Average baseline confidence: {male_yes_to_no['confidence_baseline'].mean():.2f}")
    print(f"Average no-sex confidence: {male_yes_to_no['confidence_no_sex'].mean():.2f}")
    print(f"Average confidence change: {male_yes_to_no['confidence_change'].mean():.2f}")
    print(f"Confidence increased in: {(male_yes_to_no['confidence_change'] > 0).sum()} cases")
    print(f"Confidence decreased in: {(male_yes_to_no['confidence_change'] < 0).sum()} cases")
    print(f"Confidence unchanged in: {(male_yes_to_no['confidence_change'] == 0).sum()} cases")
    
    # Demographics summary
    print(f"\n{'='*70}")
    print("DEMOGRAPHIC CHARACTERISTICS")
    print(f"{'='*70}")
    
    # Get available demographic columns
    demo_cols = ['age', 'race', 'ethnicity', 'recent_bmi', 'smoking_hx', 
                 'alcohol_use', 'insurance_type', 'zipcode', 'occupation']
    available_demos = [col for col in demo_cols if col in male_yes_to_no.columns]
    
    for col in available_demos:
        if male_yes_to_no[col].notna().sum() > 0:
            if male_yes_to_no[col].dtype == 'object':
                print(f"\n{col}:")
                print(male_yes_to_no[col].value_counts().to_string())
            else:
                print(f"\n{col}: mean={male_yes_to_no[col].mean():.1f}, "
                      f"median={male_yes_to_no[col].median():.1f}, "
                      f"range=[{male_yes_to_no[col].min():.1f}-{male_yes_to_no[col].max():.1f}]")
    
    # Print case-by-case details
    print(f"\n{'='*70}")
    print("CASE-BY-CASE ANALYSIS")
    print(f"{'='*70}")
    
    for i, (idx, row) in enumerate(male_yes_to_no.iterrows(), 1):
        print(f"\n--- CASE {i}: {row['case_id']} ---")
        print(f"Baseline: YES (confidence {row['confidence_baseline']})")
        print(f"No Sex:   NO  (confidence {row['confidence_no_sex']})")
        print(f"Confidence change: {row['confidence_change']:+.1f}")
        
        # Print key demographics
        if 'age' in row and pd.notna(row['age']):
            print(f"Age: {row['age']}")
        if 'race' in row and pd.notna(row['race']):
            print(f"Race: {row['race']}")
        if 'recent_bmi' in row and pd.notna(row['recent_bmi']):
            print(f"BMI: {row['recent_bmi']}")
        
        print(f"\nBASELINE REASONING (with sex):")
        print(f"{row['reasoning_baseline'][:300]}..." if len(str(row['reasoning_baseline'])) > 300 
              else row['reasoning_baseline'])
        
        print(f"\nNO-SEX REASONING (without sex):")
        print(f"{row['reasoning_no_sex'][:300]}..." if len(str(row['reasoning_no_sex'])) > 300 
              else row['reasoning_no_sex'])
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'male_yes_to_no_flips_detailed.csv')
    
    # Select relevant columns for CSV
    output_cols = ['case_id', 'decision_baseline', 'decision_no_sex', 
                   'confidence_baseline', 'confidence_no_sex', 'confidence_change',
                   'reasoning_baseline', 'reasoning_no_sex', 'legal_sex']
    
    # Add demographic columns if available
    output_cols.extend([col for col in available_demos if col in male_yes_to_no.columns])
    
    male_yes_to_no[output_cols].to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Detailed results saved: {output_path}")
    print(f"{'='*70}")
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'male_yes_to_no_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MALE YES→NO FLIP CASES SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total cases: {len(male_yes_to_no)}\n\n")
        f.write(f"Average baseline confidence: {male_yes_to_no['confidence_baseline'].mean():.2f}\n")
        f.write(f"Average no-sex confidence: {male_yes_to_no['confidence_no_sex'].mean():.2f}\n")
        f.write(f"Average confidence change: {male_yes_to_no['confidence_change'].mean():.2f}\n\n")
        f.write(f"Confidence increased: {(male_yes_to_no['confidence_change'] > 0).sum()} cases\n")
        f.write(f"Confidence decreased: {(male_yes_to_no['confidence_change'] < 0).sum()} cases\n")
        f.write(f"Confidence unchanged: {(male_yes_to_no['confidence_change'] == 0).sum()} cases\n\n")
        f.write("="*70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*70 + "\n\n")
        
        avg_change = male_yes_to_no['confidence_change'].mean()
        if avg_change > 0.3:
            f.write(" CRITICAL FINDING: Model is MORE CONFIDENT when recommending\n")
            f.write("   AGAINST surgery after sex is removed. This suggests sex information\n")
            f.write("   was inappropriately biasing toward surgery for these male patients.\n\n")
        elif avg_change > 0:
            f.write("  Model shows slight confidence increase when sex removed.\n")
            f.write("   Sex information may have introduced some inappropriate bias.\n\n")
        else:
            f.write("Model confidence decreased when sex removed, suggesting sex\n")
            f.write("information was providing clinically relevant context.\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("- Review these cases manually with clinical experts\n")
        f.write("- Determine if male sex should be a factor in surgical decisions\n")
        f.write("- Consider demographic-blind validation on external dataset\n")
    
    print(f"✓ Summary report saved: {summary_path}")
    
    return male_yes_to_no


# Main execution function
def run_sex_bias_analysis(baseline_path: str,
                         no_sex_path: str,
                         llm_df_filtered: pd.DataFrame = None,
                         original_data_path: str = None,
                         output_dir: str = './ablation_results_stratified'):
    """
    Complete sex bias analysis pipeline.
    
    Args:
        baseline_path: Path to baseline_results.csv
        no_sex_path: Path to no_legal_sex_results.csv
        llm_df_filtered: DataFrame with llm_caseID and legal_sex (preferred)
        original_data_path: Path to CSV with patient data (alternative to llm_df_filtered)
        output_dir: Where to save results
    """
    
    print("Loading data...")
    baseline = pd.read_csv(baseline_path)
    no_sex = pd.read_csv(no_sex_path)
    
    # Handle original data - either from DataFrame or CSV
    if llm_df_filtered is not None:
        print("Using provided llm_df_filtered DataFrame")
        original = llm_df_filtered.copy()
        
        # Save it for future reference
        os.makedirs(output_dir, exist_ok=True)
        temp_csv_path = os.path.join(output_dir, 'patient_data_temp.csv')
        original.to_csv(temp_csv_path, index=False)
        print(f"✓ Saved patient data to: {temp_csv_path}")
        
    elif original_data_path is not None:
        print(f"Loading from CSV: {original_data_path}")
        original = pd.read_csv(original_data_path)
    else:
        print("\n ERROR: Must provide either llm_df_filtered or original_data_path!")
        return
    
    print(f"✓ Baseline: {len(baseline)} cases")
    print(f"✓ No sex: {len(no_sex)} cases")
    print(f"✓ Original data: {len(original)} patients")
    
    # Check if legal_sex column exists
    if 'legal_sex' not in original.columns:
        print("\n ERROR: 'legal_sex' column not found in original data!")
        print(f"   Available columns: {original.columns.tolist()}")
        return
    
    # Check if llm_caseID exists
    if 'llm_caseID' not in original.columns:
        print("\n ERROR: 'llm_caseID' column not found in original data!")
        print(f"   Available columns: {original.columns.tolist()}")
        return
    
    # Main analysis
    results = analyze_sex_bias_direction(baseline, no_sex, original)
    
    # Detailed case analysis
    print("\n" + "="*70)
    flipped_cases = analyze_flipped_cases(baseline, no_sex, original, output_dir)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'sex_bias_summary.csv')
    results['sex_counts'].to_csv(summary_path)
    print(f"\n✓ Sex bias summary saved: {summary_path}")
    
    return results, flipped_cases

# Example usage
if __name__ == "__main__":
    """
    Run this to determine which sex is favored for surgery.
    """
    
    baseline_path = './baseline_results.csv'
    no_sex_path = './no_legal_sex_results.csv'

    print("SEX BIAS DIRECTION ANALYSIS")
    print("="*70)
    print("This will determine which sex (Male/Female) is favored")
    print("for surgical recommendations when sex information is included.\n")
    
    if not os.path.exists(baseline_path):
        print(f" ERROR: {baseline_path} not found!")
    elif not os.path.exists(no_sex_path):
        print(f" ERROR: {no_sex_path} not found!")
    else:
      results, flipped = run_sex_bias_analysis(
      baseline_path='./baseline_results.csv',
      no_sex_path='./no_legal_sex_results.csv',
      llm_df_filtered=llm_df_filtered 
)