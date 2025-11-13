import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def statistical_analysis_ablation(summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Perform comprehensive statistical analysis on ablation results.
    
    Args:
        summary_df: Summary DataFrame from analyze_ablation_results()
    
    Returns:
        Dictionary of analysis DataFrames
    """
    
    # Separate individual vs grouped experiments
    individual_df = summary_df[summary_df['experiment_type'] == 'individual'].copy()
    grouped_df = summary_df[summary_df['experiment_type'] == 'grouped'].copy()
    
    results = {}
    
    # ============================================
    # 1. DESCRIPTIVE STATISTICS
    # ============================================
    desc_stats = pd.DataFrame({
        'metric': ['flip_rate_%', 'yes_to_no', 'no_to_yes', 
                   'avg_confidence_change', 'avg_abs_confidence_change'],
        'mean_individual': [
            individual_df['flip_rate_%'].mean(),
            individual_df['yes_to_no'].mean(),
            individual_df['no_to_yes'].mean(),
            individual_df['avg_confidence_change'].mean(),
            individual_df['avg_abs_confidence_change'].mean()
        ],
        'std_individual': [
            individual_df['flip_rate_%'].std(),
            individual_df['yes_to_no'].std(),
            individual_df['no_to_yes'].std(),
            individual_df['avg_confidence_change'].std(),
            individual_df['avg_abs_confidence_change'].std()
        ],
        'median_individual': [
            individual_df['flip_rate_%'].median(),
            individual_df['yes_to_no'].median(),
            individual_df['no_to_yes'].median(),
            individual_df['avg_confidence_change'].median(),
            individual_df['avg_abs_confidence_change'].median()
        ],
        'mean_grouped': [
            grouped_df['flip_rate_%'].mean(),
            grouped_df['yes_to_no'].mean(),
            grouped_df['no_to_yes'].mean(),
            grouped_df['avg_confidence_change'].mean(),
            grouped_df['avg_abs_confidence_change'].mean()
        ],
        'std_grouped': [
            grouped_df['flip_rate_%'].std(),
            grouped_df['yes_to_no'].std(),
            grouped_df['no_to_yes'].std(),
            grouped_df['avg_confidence_change'].std(),
            grouped_df['avg_abs_confidence_change'].std()
        ]
    })
    results['descriptive_stats'] = desc_stats
    
    # ============================================
    # 2. STATISTICAL SIGNIFICANCE TESTS
    # ============================================
    
    # Test if individual variables differ from baseline (zero effect)
    significance_tests = []
    
    for _, row in individual_df.iterrows():
        var_name = row['excluded']
        flip_rate = row['flip_rate_%']
        
        n_flips = row['decision_flips']
        n_total = row['total_cases']
        
        # CORRECTED: Test against random/negligible baseline
        # Null hypothesis: flip rate = 1% (essentially zero effect, accounting for API noise)
        # Alternative: flip rate is significantly greater than 1%
        from scipy.stats import binomtest
        
        # Test 1: Against 1% baseline (very conservative - any signal)
        p_vs_1pct = binomtest(n_flips, n_total, p=0.01, alternative='greater').pvalue
        
        # Test 2: Against 2% baseline (more realistic noise threshold)
        p_vs_2pct = binomtest(n_flips, n_total, p=0.02, alternative='greater').pvalue
        
        # Test 3: Two-sided test against 3% (is it different from low baseline?)
        p_vs_3pct_twosided = binomtest(n_flips, n_total, p=0.03, alternative='two-sided').pvalue
        
        # Use the most relevant test (against 1% baseline)
        primary_p_value = p_vs_1pct
        
        significance_tests.append({
            'variable': var_name,
            'flip_rate_%': flip_rate,
            'n_flips': n_flips,
            'n_total': n_total,
            'p_value': primary_p_value,
            'p_vs_1pct': p_vs_1pct,
            'p_vs_2pct': p_vs_2pct,
            'p_vs_3pct_twosided': p_vs_3pct_twosided,
            'significant_at_0.05': primary_p_value < 0.05,
            'significant_at_0.01': primary_p_value < 0.01,
            'significant_at_0.001': primary_p_value < 0.001
        })
    
    sig_df = pd.DataFrame(significance_tests)
    sig_df = sig_df.sort_values('p_value')
    results['significance_tests'] = sig_df
    
    # ============================================
    # 3. EFFECT SIZE CALCULATIONS
    # ============================================
    
    # Cohen's h for effect size (proportion differences)
    effect_sizes = []
    
    for _, row in individual_df.iterrows():
        flip_rate = row['flip_rate_%'] / 100
        # Compare against 1% baseline (near-zero effect)
        baseline_rate = 0.01
        
        # Cohen's h for proportions
        h = 2 * (np.arcsin(np.sqrt(flip_rate)) - np.arcsin(np.sqrt(baseline_rate)))
        
        # Effect size interpretation (standard thresholds)
        if abs(h) < 0.2:
            interpretation = 'negligible'
        elif abs(h) < 0.5:
            interpretation = 'small'
        elif abs(h) < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        # Also calculate odds ratio for interpretability
        # Odds of flip when variable excluded vs baseline
        p_excluded = flip_rate
        p_baseline = baseline_rate
        odds_ratio = (p_excluded / (1 - p_excluded)) / (p_baseline / (1 - p_baseline))
        
        effect_sizes.append({
            'variable': row['excluded'],
            'flip_rate_%': row['flip_rate_%'],
            'cohens_h': h,
            'effect_size': interpretation,
            'odds_ratio': odds_ratio,
            'avg_confidence_change': row['avg_confidence_change']
        })
    
    effect_df = pd.DataFrame(effect_sizes)
    effect_df = effect_df.sort_values('cohens_h', ascending=False, key=abs)
    results['effect_sizes'] = effect_df
    
    # ============================================
    # 4. RANKING AND CATEGORIZATION
    # ============================================
    
    # Rank variables by impact
    individual_df['impact_rank'] = individual_df['flip_rate_%'].rank(ascending=False)
    
    # Categorize impact level
    def categorize_impact(flip_rate):
        if flip_rate >= 30:
            return 'high'
        elif flip_rate >= 20:
            return 'moderate'
        elif flip_rate >= 10:
            return 'low'
        else:
            return 'minimal'
    
    individual_df['impact_level'] = individual_df['flip_rate_%'].apply(categorize_impact)
    results['ranked_variables'] = individual_df[['excluded', 'flip_rate_%', 'impact_rank', 'impact_level']].copy()
    
    # ============================================
    # 5. DIRECTIONAL BIAS ANALYSIS
    # ============================================
    
    bias_analysis = []
    
    for _, row in individual_df.iterrows():
        yes_to_no = row['yes_to_no']
        no_to_yes = row['no_to_yes']
        total_flips = yes_to_no + no_to_yes
        
        if total_flips > 0:
            bias_ratio = yes_to_no / total_flips
            
            if bias_ratio > 0.6:
                direction = 'toward_no_surgery'
            elif bias_ratio < 0.4:
                direction = 'toward_yes_surgery'
            else:
                direction = 'balanced'
        else:
            bias_ratio = 0.5
            direction = 'no_flips'
        
        bias_analysis.append({
            'variable': row['excluded'],
            'yes_to_no': yes_to_no,
            'no_to_yes': no_to_yes,
            'bias_ratio': bias_ratio,
            'directional_bias': direction
        })
    
    bias_df = pd.DataFrame(bias_analysis)
    results['directional_bias'] = bias_df
    
    # ============================================
    # 6. GROUPED VS INDIVIDUAL COMPARISON
    # ============================================
    
    if len(grouped_df) > 0:
        # Compare grouped experiments to their constituent individual variables
        group_comparison = []
        
        for _, group_row in grouped_df.iterrows():
            group_name = group_row['excluded']
            group_flip_rate = group_row['flip_rate_%']
            
            # Get corresponding individual variables (approximate)
            if 'protected' in group_name:
                constituent_vars = ['legal_sex', 'race', 'ethnicity']
            elif 'socioeconomic' in group_name:
                constituent_vars = ['zipcode', 'insurance_type', 'occupation']
            elif 'health' in group_name:
                constituent_vars = ['smoking_hx', 'alcohol_use']
            elif 'physical' in group_name:
                constituent_vars = ['age', 'recent_bmi']
            else:
                constituent_vars = []
            
            if constituent_vars:
                individual_flip_rates = individual_df[
                    individual_df['excluded'].isin(constituent_vars)
                ]['flip_rate_%']
                
                avg_individual = individual_flip_rates.mean()
                max_individual = individual_flip_rates.max()
                
                group_comparison.append({
                    'group': group_name,
                    'group_flip_rate': group_flip_rate,
                    'avg_individual_flip_rate': avg_individual,
                    'max_individual_flip_rate': max_individual,
                    'synergy_effect': group_flip_rate - avg_individual,
                    'is_superadditive': group_flip_rate > avg_individual
                })
        
        if group_comparison:
            results['group_vs_individual'] = pd.DataFrame(group_comparison)
    
    return results


def create_ablation_visualizations(summary_df: pd.DataFrame, 
                                   analysis_results: Dict[str, pd.DataFrame],
                                   output_dir: str = './ablation_plots'):
    """
    Create comprehensive visualizations for ablation analysis.
    
    Args:
        summary_df: Summary DataFrame from analyze_ablation_results()
        analysis_results: Results from statistical_analysis_ablation()
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    individual_df = summary_df[summary_df['experiment_type'] == 'individual'].copy()
    grouped_df = summary_df[summary_df['experiment_type'] == 'grouped'].copy()
    
    # ============================================
    # PLOT 1: Flip Rate by Variable (Ranked)
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    individual_sorted = individual_df.sort_values('flip_rate_%', ascending=True)
    
    colors = ['#d62728' if x >= 30 else '#ff7f0e' if x >= 20 else '#2ca02c' if x >= 10 else '#1f77b4' 
              for x in individual_sorted['flip_rate_%']]
    
    ax.barh(individual_sorted['excluded'], individual_sorted['flip_rate_%'], color=colors, alpha=0.8)
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax.set_xlabel('Decision Flip Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Excluded Variable', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Excluding Each Demographic Variable\n(Higher = More Influential)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_flip_rates_ranked.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_flip_rates_ranked.png")
    plt.close()
    
    # ============================================
    # PLOT 2: Directional Bias (Yes→No vs No→Yes)
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    individual_sorted = individual_df.sort_values('flip_rate_%', ascending=False)
    
    x = np.arange(len(individual_sorted))
    width = 0.35
    
    ax.bar(x - width/2, individual_sorted['yes_to_no'], width, label='Yes → No', 
           color='#d62728', alpha=0.8)
    ax.bar(x + width/2, individual_sorted['no_to_yes'], width, label='No → Yes', 
           color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Excluded Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Decision Flips', fontsize=12, fontweight='bold')
    ax.set_title('Directional Bias of Decision Changes by Variable', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(individual_sorted['excluded'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_directional_bias.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_directional_bias.png")
    plt.close()
    
    # ============================================
    # PLOT 3: Confidence Changes
    # ============================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average confidence change
    individual_sorted = individual_df.sort_values('avg_confidence_change', ascending=True)
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in individual_sorted['avg_confidence_change']]
    
    ax1.barh(individual_sorted['excluded'], individual_sorted['avg_confidence_change'], 
             color=colors, alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Average Confidence Change', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Excluded Variable', fontsize=11, fontweight='bold')
    ax1.set_title('Average Confidence Change\n(Negative = Less Confident)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Absolute confidence change
    individual_sorted = individual_df.sort_values('avg_abs_confidence_change', ascending=False)
    ax2.barh(individual_sorted['excluded'], individual_sorted['avg_abs_confidence_change'], 
             color='#9467bd', alpha=0.8)
    ax2.set_xlabel('Average Absolute Confidence Change', fontsize=11, fontweight='bold')
    ax2.set_title('Magnitude of Confidence Impact\n(Higher = More Uncertainty)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_confidence_changes.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_confidence_changes.png")
    plt.close()
    
    # ============================================
    # PLOT 4: Individual vs Grouped Comparison
    # ============================================
    if len(grouped_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Combine and sort
        combined = pd.concat([
            individual_df[['excluded', 'flip_rate_%', 'experiment_type']],
            grouped_df[['excluded', 'flip_rate_%', 'experiment_type']]
        ])
        combined = combined.sort_values('flip_rate_%', ascending=True)
        
        colors = ['#1f77b4' if t == 'individual' else '#ff7f0e' for t in combined['experiment_type']]
        
        ax.barh(combined['excluded'], combined['flip_rate_%'], color=colors, alpha=0.8)
        ax.axvline(x=25, color='red', linestyle='--', alpha=0.5, label='25% threshold')
        ax.set_xlabel('Decision Flip Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Excluded Variable/Group', fontsize=12, fontweight='bold')
        ax.set_title('Individual vs Grouped Variable Impact', fontsize=14, fontweight='bold', pad=20)
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.8, label='Individual Variable'),
            Patch(facecolor='#ff7f0e', alpha=0.8, label='Grouped Variables')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_individual_vs_grouped.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/04_individual_vs_grouped.png")
        plt.close()
    
    # ============================================
    # PLOT 5: Statistical Significance Heatmap
    # ============================================
    if 'significance_tests' in analysis_results:
        sig_df = analysis_results['significance_tests']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create significance matrix
        sig_matrix = sig_df[['variable', 'flip_rate_%', 'p_value']].copy()
        sig_matrix['neg_log_p'] = -np.log10(sig_matrix['p_value'] + 1e-10)  # Avoid log(0)
        sig_matrix = sig_matrix.sort_values('flip_rate_%', ascending=False)
        
        # Create color map based on significance
        colors_sig = []
        for p in sig_matrix['p_value']:
            if p < 0.001:
                colors_sig.append('#8b0000')  # Dark red: highly significant
            elif p < 0.01:
                colors_sig.append('#d62728')  # Red: significant at 0.01
            elif p < 0.05:
                colors_sig.append('#ff7f0e')  # Orange: significant at 0.05
            else:
                colors_sig.append('#808080')  # Gray: not significant
        
        ax.barh(sig_matrix['variable'], sig_matrix['flip_rate_%'], color=colors_sig, alpha=0.8)
        ax.set_xlabel('Decision Flip Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Significance of Variable Impact\n(Color = p-value)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#8b0000', alpha=0.8, label='p < 0.001 (***)'),
            Patch(facecolor='#d62728', alpha=0.8, label='p < 0.01 (**)'),
            Patch(facecolor='#ff7f0e', alpha=0.8, label='p < 0.05 (*)'),
            Patch(facecolor='#808080', alpha=0.8, label='p ≥ 0.05 (ns)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_statistical_significance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/05_statistical_significance.png")
        plt.close()
    
    # ============================================
    # PLOT 6: Scatter: Flip Rate vs Confidence Change
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(individual_df['flip_rate_%'], 
                        individual_df['avg_abs_confidence_change'],
                        s=individual_df['decision_flips'] * 2,  # Size by number of flips
                        c=individual_df['flip_rate_%'],
                        cmap='RdYlGn_r',
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=1)
    
    # Add labels for each point
    for _, row in individual_df.iterrows():
        ax.annotate(row['excluded'], 
                   (row['flip_rate_%'], row['avg_abs_confidence_change']),
                   fontsize=9,
                   alpha=0.8,
                   xytext=(5, 5),
                   textcoords='offset points')
    
    ax.set_xlabel('Decision Flip Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Absolute Confidence Change', fontsize=12, fontweight='bold')
    ax.set_title('Impact on Decisions vs Confidence\n(Size = Number of Flips)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Flip Rate (%)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_fliprate_vs_confidence.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/06_fliprate_vs_confidence.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")


def generate_ablation_report(summary_df: pd.DataFrame, 
                            analysis_results: Dict[str, pd.DataFrame],
                            output_path: str = './ablation_report.txt'):
    """
    Generate a text report summarizing the ablation analysis.
    """
    
    individual_df = summary_df[summary_df['experiment_type'] == 'individual']
    grouped_df = summary_df[summary_df['experiment_type'] == 'grouped']
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ABLATION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overview
        f.write("OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total cases analyzed: {individual_df['total_cases'].iloc[0]}\n")
        f.write(f"Individual variables tested: {len(individual_df)}\n")
        f.write(f"Grouped experiments: {len(grouped_df)}\n\n")
        
        # Top impactful variables
        f.write("TOP 5 MOST IMPACTFUL VARIABLES\n")
        f.write("-"*70 + "\n")
        top5 = individual_df.nlargest(5, 'flip_rate_%')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            f.write(f"{i}. {row['excluded']}: {row['flip_rate_%']:.2f}% flip rate ")
            f.write(f"({row['decision_flips']} flips: {row['yes_to_no']} Y→N, {row['no_to_yes']} N→Y)\n")
        f.write("\n")
        
        # Statistical significance
        if 'significance_tests' in analysis_results:
            sig_df = analysis_results['significance_tests']
            sig_vars = sig_df[sig_df['significant_at_0.05']]
            
            f.write("STATISTICALLY SIGNIFICANT VARIABLES (p < 0.05)\n")
            f.write("-"*70 + "\n")
            f.write("Testing H0: flip rate ≤ 1% (negligible effect) vs H1: flip rate > 1%\n\n")
            if len(sig_vars) > 0:
                for _, row in sig_vars.iterrows():
                    stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
                    f.write(f"• {row['variable']}: {row['flip_rate_%']:.2f}% flip rate, ")
                    f.write(f"p = {row['p_value']:.4f} {stars}\n")
            else:
                f.write("No variables reached statistical significance at p < 0.05\n")
                f.write(f"\nNote: All flip rates (~{individual_df['flip_rate_%'].mean():.1f}%) are ")
                f.write("highly consistent, suggesting either:\n")
                f.write("  1. All demographics have similar modest influence\n")
                f.write("  2. Sample size (n={}) may be insufficient to detect differences\n".format(
                    individual_df['total_cases'].iloc[0]))
                f.write("  3. The model is relatively robust to demographic exclusion\n")
            f.write("\n")
        
        # Effect sizes
        if 'effect_sizes' in analysis_results:
            effect_df = analysis_results['effect_sizes']
            large_effects = effect_df[effect_df['effect_size'].isin(['medium', 'large'])]
            
            f.write("VARIABLES WITH MEDIUM/LARGE EFFECT SIZES\n")
            f.write("-"*70 + "\n")
            if len(large_effects) > 0:
                for _, row in large_effects.iterrows():
                    f.write(f"• {row['variable']}: Cohen's h = {row['cohens_h']:.3f} ({row['effect_size']})\n")
            else:
                f.write("No variables showed medium or large effect sizes\n")
            f.write("\n")
        
        # Directional bias
        if 'directional_bias' in analysis_results:
            bias_df = analysis_results['directional_bias']
            biased_vars = bias_df[bias_df['directional_bias'] != 'balanced']
            
            f.write("DIRECTIONAL BIAS ANALYSIS\n")
            f.write("-"*70 + "\n")
            toward_no = biased_vars[biased_vars['directional_bias'] == 'toward_no_surgery']
            toward_yes = biased_vars[biased_vars['directional_bias'] == 'toward_yes_surgery']
            
            if len(toward_no) > 0:
                f.write("Variables biasing TOWARD no surgery:\n")
                for _, row in toward_no.iterrows():
                    f.write(f"  • {row['variable']}: {row['yes_to_no']} Y→N vs {row['no_to_yes']} N→Y\n")
            
            if len(toward_yes) > 0:
                f.write("\nVariables biasing TOWARD yes surgery:\n")
                for _, row in toward_yes.iterrows():
                    f.write(f"  • {row['variable']}: {row['yes_to_no']} Y→N vs {row['no_to_yes']} N→Y\n")
            f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        desc = analysis_results['descriptive_stats']
        f.write(f"Average flip rate (individual): {desc[desc['metric']=='flip_rate_%']['mean_individual'].values[0]:.2f}%\n")
        f.write(f"Median flip rate (individual): {desc[desc['metric']=='flip_rate_%']['median_individual'].values[0]:.2f}%\n")
        f.write(f"Std dev flip rate (individual): {desc[desc['metric']=='flip_rate_%']['std_individual'].values[0]:.2f}%\n")
        
        if len(grouped_df) > 0:
            f.write(f"\nAverage flip rate (grouped): {desc[desc['metric']=='flip_rate_%']['mean_grouped'].values[0]:.2f}%\n")
            f.write(f"Std dev flip rate (grouped): {desc[desc['metric']=='flip_rate_%']['std_grouped'].values[0]:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Report saved to: {output_path}")


# ============================================
# MAIN EXECUTION FUNCTION
# ============================================

def run_complete_ablation_analysis(summary_csv_path: str, 
                                   output_dir: str = './ablation_analysis'):
    """
    Run complete statistical analysis and visualization from summary CSV.
    
    Args:
        summary_csv_path: Path to ablation_summary.csv
        output_dir: Directory for outputs
    
    Usage:
        run_complete_ablation_analysis('./ablation_results/ablation_summary.csv')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading ablation summary...")
    summary_df = pd.read_csv(summary_csv_path)
    print(f"✓ Loaded {len(summary_df)} experiments\n")
    
    # Statistical analysis
    print("Running statistical analysis...")
    analysis_results = statistical_analysis_ablation(summary_df)
    print("✓ Statistical analysis complete\n")
    
    # Save analysis results
    print("Saving analysis results...")
    for name, df in analysis_results.items():
        output_path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(output_path, index=False)
        print(f"  ✓ {name}.csv")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    plots_dir = os.path.join(output_dir, 'plots')
    create_ablation_visualizations(summary_df, analysis_results, plots_dir)
    print()
    
    # Generate report
    print("Generating text report...")
    report_path = os.path.join(output_dir, 'ablation_report.txt')
    generate_ablation_report(summary_df, analysis_results, report_path)
    print()
    
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"All results saved to: {output_dir}/")
    print(f"  • Statistical tests: {output_dir}/*.csv")
    print(f"  • Visualizations: {plots_dir}/*.png")
    print(f"  • Text report: {report_path}")
    
    return analysis_results

# Example Usage
stat_analysis_results = run_complete_ablation_analysis(
    summary_csv_path='./ablation_results_stratified/ablation_summary.csv',
    output_dir='./ablation_analysis'
)