def stratified_sample_for_ablation(df: pd.DataFrame,
                                   sample_size: int,
                                   stratify_vars: List[str] = None,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Create a stratified sample that maintains demographic distributions.

    Args:
        df: Full DataFrame
        sample_size: Target sample size
        stratify_vars: Variables to stratify on (default: key demographics)
        random_state: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    if stratify_vars is None:
        # Stratify on protected attributes and key demographics
        stratify_vars = ['legal_sex', 'race']

    # Remove any stratify vars that don't exist or have too many NAs
    stratify_vars = [v for v in stratify_vars if v in df.columns
                     and df[v].notna().sum() > sample_size * 0.1]

    if not stratify_vars:
        print("Warning: No valid stratification variables, using random sample")
        return df.sample(n=min(sample_size, len(df)), random_state=random_state)

    # Create a composite stratification key
    df_copy = df.copy()
    df_copy['_strata'] = df_copy[stratify_vars].astype(str).agg('_'.join, axis=1)

    # Calculate proportional sample sizes for each stratum
    strata_counts = df_copy['_strata'].value_counts()
    strata_proportions = strata_counts / len(df_copy)

    # Ensure minimum samples per stratum (at least 5 if possible)
    min_per_stratum = 5
    strata_samples = (strata_proportions * sample_size).round().astype(int)
    strata_samples = strata_samples.clip(lower=min(min_per_stratum, sample_size // len(strata_samples)))

    # Adjust if total exceeds sample_size
    while strata_samples.sum() > sample_size:
        # Reduce from largest strata
        largest = strata_samples.idxmax()
        strata_samples[largest] -= 1

    # Sample from each stratum
    sampled_dfs = []
    for stratum, n_samples in strata_samples.items():
        stratum_df = df_copy[df_copy['_strata'] == stratum]
        if len(stratum_df) >= n_samples:
            sampled_dfs.append(stratum_df.sample(n=n_samples, random_state=random_state))
        else:
            # Take all if stratum is smaller than target
            sampled_dfs.append(stratum_df)

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=['_strata'])

    return result


def check_demographic_balance(df_full: pd.DataFrame,
                              df_sample: pd.DataFrame,
                              demographic_vars: List[str]) -> pd.DataFrame:
    """
    Compare demographic distributions between full dataset and sample.

    Args:
        df_full: Full dataset
        df_sample: Sampled dataset
        demographic_vars: Variables to compare

    Returns:
        DataFrame with comparison statistics
    """
    comparisons = []

    for var in demographic_vars:
        if var not in df_full.columns:
            continue

        # Get value counts and proportions
        full_counts = df_full[var].value_counts(normalize=True)
        sample_counts = df_sample[var].value_counts(normalize=True)

        # Combine and compare
        for value in full_counts.index:
            full_prop = full_counts.get(value, 0)
            sample_prop = sample_counts.get(value, 0)

            comparisons.append({
                'variable': var,
                'value': value,
                'full_proportion': full_prop,
                'sample_proportion': sample_prop,
                'difference': abs(full_prop - sample_prop),
                'full_count': (df_full[var] == value).sum(),
                'sample_count': (df_sample[var] == value).sum()
            })

    comparison_df = pd.DataFrame(comparisons)
    comparison_df = comparison_df.sort_values('difference', ascending=False)

    return comparison_df

def run_ablation_with_stratified_sampling(llm_df: pd.DataFrame,
                                         output_dir: str = './ablation_results',
                                         sample_size: int = 500,
                                         stratify_vars: List[str] = None,
                                         include_groups: bool = True) -> tuple:
    """
    Run ablation study with stratified sampling to maintain demographic balance.

    Args:
        llm_df: Full DataFrame with case data
        output_dir: Directory to save results
        sample_size: Sample size for ablation
        stratify_vars: Variables to stratify on (None = use defaults)
        include_groups: Whether to include grouped ablation

    Returns:
        Tuple of (all_results dict, summary DataFrame, balance_check DataFrame)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Original dataset size: {len(llm_df)}")
    print(f"Requested sample size: {sample_size}")

    # Create stratified sample
    print("\nCreating stratified sample...")
    sampled_df = stratified_sample_for_ablation(
        llm_df,
        sample_size=sample_size,
        stratify_vars=stratify_vars
    )

    print(f"Actual sample size: {len(sampled_df)}")

    # Check demographic balance
    print("\nChecking demographic balance...")
    balance_check = check_demographic_balance(
        llm_df,
        sampled_df,
        DEMOGRAPHIC_VARS
    )

    # Print top differences
    print("\nTop 10 demographic distribution differences:")
    print(balance_check.head(10)[['variable', 'value', 'full_proportion',
                                   'sample_proportion', 'difference']].to_string(index=False))

    # Save balance check
    balance_path = os.path.join(output_dir, 'sampling_balance_check.csv')
    balance_check.to_csv(balance_path, index=False)
    print(f"\n✓ Balance check saved: {balance_path}")

    # Run ablation on stratified sample
    print("\n" + "="*60)
    print("Running ablation analysis on stratified sample...")
    print("="*60)

    all_results = run_ablation_analysis(
        sampled_df,
        delay_seconds=0.2,
        sample_size=None,  # Don't resample - already sampled
        include_groups=include_groups,
        output_dir=output_dir
    )

    # Analyze results
    summary = analyze_ablation_results(all_results)

    # Save final summary
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")

    return all_results, summary, balance_check


# Example usage
ablation_results, summary, balance = run_ablation_with_stratified_sampling(
    llm_df_filtered,
    output_dir='./ablation_results_stratified',
    sample_size=500,
    stratify_vars=['legal_sex', 'race'],
    include_groups=True
)