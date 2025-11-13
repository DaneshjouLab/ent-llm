def analyze_baseline_noise_symmetry(test_retest_results_path: str) -> dict:
    """
    Analyze whether baseline API noise is symmetric or has directional bias.
    
    Args:
        test_retest_results_path: Path to test_retest_results.csv
    
    Returns:
        Dictionary with symmetry analysis results
    """
    # Load test-retest results
    results = pd.read_csv(test_retest_results_path)
    
    # Filter to valid cases only
    valid = results[results['both_valid'] == True].copy()
    
    print(f"\n{'='*70}")
    print("BASELINE NOISE SYMMETRY ANALYSIS")
    print(f"{'='*70}")
    print(f"Valid test-retest pairs: {len(valid)}")
    
    # Count flips by direction
    yes_to_no = ((valid['decision_test1'] == 'Yes') & 
                 (valid['decision_test2'] == 'No')).sum()
    no_to_yes = ((valid['decision_test1'] == 'No') & 
                 (valid['decision_test2'] == 'Yes')).sum()
    
    total_flips = yes_to_no + no_to_yes
    
    # Calculate rates
    baseline_yes = (valid['decision_test1'] == 'Yes').sum()
    baseline_no = (valid['decision_test1'] == 'No').sum()
    
    yes_to_no_rate = (yes_to_no / baseline_yes * 100) if baseline_yes > 0 else 0
    no_to_yes_rate = (no_to_yes / baseline_no * 100) if baseline_no > 0 else 0
    
    asymmetry = abs(yes_to_no_rate - no_to_yes_rate)
    
    print(f"\nBaseline Decision Distribution:")
    print(f"  Test 1 'Yes' decisions: {baseline_yes}")
    print(f"  Test 1 'No' decisions: {baseline_no}")
    
    print(f"\nFlip Counts:")
    print(f"  Yes→No flips: {yes_to_no}")
    print(f"  No→Yes flips: {no_to_yes}")
    print(f"  Total flips: {total_flips}")
    
    print(f"\nFlip Rates:")
    print(f"  Yes→No rate: {yes_to_no_rate:.2f}% (of {baseline_yes} Yes cases)")
    print(f"  No→Yes rate: {no_to_yes_rate:.2f}% (of {baseline_no} No cases)")
    print(f"  Asymmetry: {asymmetry:.2f}%")
    
    # Statistical test for symmetry
    # H0: Flip rates are equal (symmetric noise)
    # H1: Flip rates differ (asymmetric noise)
    
    contingency_table = np.array([
        [yes_to_no, baseline_yes - yes_to_no],
        [no_to_yes, baseline_no - no_to_yes]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nSymmetry Test:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    is_symmetric = p_value > 0.05
    
    if is_symmetric:
        print(f" BASELINE NOISE IS SYMMETRIC (p={p_value:.4f} > 0.05)")
    else:
        print(f"BASELINE NOISE IS ASYMMETRIC (p={p_value:.4f} < 0.05)")
        print(f"   API has inherent directional bias: {yes_to_no_rate:.2f}% vs {no_to_yes_rate:.2f}%")
        
    # Calculate net demographic effect
    
    net_legal_sex = 6.19 - asymmetry
    net_all_demo = 5.96 - asymmetry
    net_protected = 5.93 - asymmetry
    
    print(f"  legal_sex: {net_legal_sex:.2f}% (6.19% - {asymmetry:.2f}%)")
    print(f"  all_demographics: {net_all_demo:.2f}% (5.96% - {asymmetry:.2f}%)")
    print(f"  protected_attributes: {net_protected:.2f}% (5.93% - {asymmetry:.2f}%)")
    
    return {
        'baseline_yes_count': baseline_yes,
        'baseline_no_count': baseline_no,
        'yes_to_no_flips': yes_to_no,
        'no_to_yes_flips': no_to_yes,
        'yes_to_no_rate_%': yes_to_no_rate,
        'no_to_yes_rate_%': no_to_yes_rate,
        'asymmetry_%': asymmetry,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'is_symmetric': is_symmetric,
        'net_legal_sex_effect_%': net_legal_sex,
        'net_all_demographics_effect_%': net_all_demo,
        'net_protected_attributes_effect_%': net_protected
    }


def compare_baseline_to_ablation_asymmetry(test_retest_path: str,
                                           stratified_analysis_path: str):
    """
    Direct comparison of baseline noise asymmetry vs ablation asymmetries.
    
    Args:
        test_retest_path: Path to test_retest_results.csv
        stratified_analysis_path: Path to stratified_analysis.csv
    """
    # Analyze baseline
    baseline_stats = analyze_baseline_noise_symmetry(test_retest_path)
    
    # Load ablation stratified results
    stratified = pd.read_csv(stratified_analysis_path)
    
    print(f"\n{'='*70}")
    print("BASELINE vs ABLATION ASYMMETRY COMPARISON")
    print(f"{'='*70}")
    
    baseline_asym = baseline_stats['asymmetry_%']
    
    print(f"\nBaseline API noise asymmetry: {baseline_asym:.2f}%")
    print(f"(This is the 'floor' - any ablation asymmetry must exceed this)\n")
    
    # Sort by asymmetry
    stratified = stratified.sort_values('asymmetry_%', ascending=False)
    
    print(f"{'Variable':<25} {'Asymmetry':<12} {'Exceeds Baseline':<20} {'Status'}")
    print("-" * 80)
    
    for _, row in stratified.head(10).iterrows():
        var_name = row['experiment'].replace('no_', '')
        asym = row['asymmetry_%']
        exceeds = asym - baseline_asym
        
        if exceeds > 2.0:
            status = "REAL EFFECT"
        elif exceeds > 1.0:
            status = "WEAK"
        else:
            status = "NOISE"
        
        print(f"{var_name:<25} {asym:>7.2f}%     +{exceeds:>6.2f}%            {status}")
    
    # Save comparison
    comparison = stratified.copy()
    comparison['baseline_asymmetry_%'] = baseline_asym
    comparison['net_effect_%'] = comparison['asymmetry_%'] - baseline_asym
    comparison['exceeds_baseline'] = comparison['net_effect_%'] > 2.0
    
    output_path = os.path.dirname(test_retest_path)
    comparison_file = os.path.join(output_path, 'baseline_vs_ablation_comparison.csv')
    comparison.to_csv(comparison_file, index=False)
    
    print(f"\n✓ Saved detailed comparison: {comparison_file}")


# Example Usage

if __name__ == "__main__":
    """
    Run this to check if your baseline noise was actually symmetric.
    """
    test_retest_path = './test_retest_ablation_sample/test_retest_results.csv'
    stratified_path = './ablation_results_stratified/stratified_analysis.csv'
    
    # Check if files exist
    if not os.path.exists(test_retest_path):
        print(f" ERROR: {test_retest_path} not found!")
        print("   Run your test-retest analysis first.")
    elif not os.path.exists(stratified_path):
        print(f" ERROR: {stratified_path} not found!")
        print("   Run the stratified analysis first.")
    else:
        # Run the comparison
        compare_baseline_to_ablation_asymmetry(test_retest_path, stratified_path)