#!/usr/bin/env python3
"""
CLI entrypoint for ENT-LLM demographic ablation analysis.

Runs ablation experiments that selectively exclude demographic variables
from LLM prompts to measure their influence on surgical recommendations.
"""

import argparse
import logging
import os
import shutil
import sys

import pandas as pd

from llm_query.ablation_analysis import (
    DEMOGRAPHIC_GROUPS,
    DEMOGRAPHIC_VARS,
    analyze_ablation_results,
    filter_long_cases,
    run_ablation_experiment,
    stratified_sample,
)
from llm_query.securellm_adapter import ModelConfig

# Available LLM models (same as cli.py)
AVAILABLE_MODELS = [
    "apim:llama-3.3-70b",
    "apim:claude-3.7",
    "apim:gpt-4.1",
    "apim:gemini-2.5-pro-preview-05-06",
]


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def list_experiments() -> None:
    """List all ablation experiments and exit."""
    print("Ablation experiments:\n")
    print("Baseline:")
    print("  baseline — all demographics included\n")
    print("Individual ablations (exclude one variable):")
    for var in DEMOGRAPHIC_VARS:
        print(f"  no_{var}")
    print("\nGrouped ablations (exclude a group):")
    for group_name, group_vars in DEMOGRAPHIC_GROUPS.items():
        print(f"  no_{group_name} — excludes: {', '.join(group_vars)}")

    total = 1 + len(DEMOGRAPHIC_VARS) + len(DEMOGRAPHIC_GROUPS)
    print(f"\nTotal experiments: {total}")


def validate_input(df: pd.DataFrame) -> None:
    """Validate required columns exist in input DataFrame."""
    clinical_cols = ["llm_caseID", "formatted_progress_text", "formatted_radiology_text"]
    missing_clinical = [c for c in clinical_cols if c not in df.columns]
    if missing_clinical:
        raise ValueError(f"Input CSV missing required clinical columns: {missing_clinical}")

    demographic_present = [v for v in DEMOGRAPHIC_VARS if v in df.columns]
    if not demographic_present:
        raise ValueError(
            f"Input CSV has no demographic columns. Expected at least some of: {DEMOGRAPHIC_VARS}"
        )

    missing_demo = [v for v in DEMOGRAPHIC_VARS if v not in df.columns]
    if missing_demo:
        logging.getLogger(__name__).warning(
            f"Input CSV missing some demographic columns (will be skipped): {missing_demo}"
        )


def build_experiment_list(experiments_mode: str):
    """Build list of (experiment_name, exclude_vars) tuples.

    Args:
        experiments_mode: One of 'all', 'individual', 'grouped', 'baseline-only'.

    Returns:
        List of (experiment_name, exclude_vars) tuples.
    """
    experiments = []

    if experiments_mode in ("all", "baseline-only"):
        experiments.append(("baseline", None))

    if experiments_mode in ("all", "individual"):
        for var in DEMOGRAPHIC_VARS:
            experiments.append((f"no_{var}", [var]))

    if experiments_mode in ("all", "grouped"):
        for group_name, group_vars in DEMOGRAPHIC_GROUPS.items():
            experiments.append((f"no_{group_name}", list(group_vars)))

    return experiments


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print formatted summary table to stdout."""
    if summary_df.empty:
        print("No ablation results to summarize.")
        return

    print(f"\n{'='*80}")
    print("ABLATION ANALYSIS SUMMARY")
    print(f"{'='*80}")

    cols = [
        "experiment",
        "experiment_type",
        "flip_rate_%",
        "yes_to_no",
        "no_to_yes",
        "avg_confidence_change",
    ]
    if "accuracy_change_%" in summary_df.columns:
        cols.append("accuracy_change_%")

    display_cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))
    print(f"{'='*80}")


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="ENT-LLM Ablation Analysis CLI - demographic ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python cli_ablation.py --list-experiments

  # Run on small sample
  python cli_ablation.py -m apim:gpt-4.1 -i data.csv -o ./ablation_test -n 10

  # Run with pre-computed baseline
  python cli_ablation.py -m apim:gpt-4.1 -i data.csv -o ./ablation_results \\
      -b ./ablation_results/baseline_results.csv

  # Run only individual ablations
  python cli_ablation.py -m apim:gpt-4.1 -i data.csv -e individual

  # Run only baseline
  python cli_ablation.py -m apim:gpt-4.1 -i data.csv -e baseline-only
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=AVAILABLE_MODELS,
        default="apim:gpt-4.1",
        help="LLM model to use (default: apim:gpt-4.1)",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input CSV file (clinical text + demographics)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./ablation_results",
        help="Output directory for result CSVs (default: ./ablation_results)",
    )

    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        help="Path to pre-computed baseline CSV (skip baseline run)",
    )

    parser.add_argument(
        "--experiments",
        "-e",
        type=str,
        choices=["all", "individual", "grouped", "baseline-only"],
        default="all",
        help="Which experiments to run (default: all)",
    )

    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        help="Optional sample size (stratified sampling)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Filter out cases exceeding this estimated token count (e.g. 5000)",
    )

    parser.add_argument(
        "--ground-truth",
        "-g",
        type=str,
        default="had_surgery",
        help="Column name for ground truth (default: had_surgery)",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds (default: 0.2)",
    )

    parser.add_argument(
        "--flush-interval",
        "-f",
        type=int,
        default=10,
        help="Flush interval for incremental saving (default: 10)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output",
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all experiments and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle --list-experiments
    if args.list_experiments:
        list_experiments()
        return 0

    # --input is required for all other modes
    if not args.input:
        parser.error("--input / -i is required (use --list-experiments to see experiments)")

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Set model
    ModelConfig.DEFAULT_LLM_MODEL = args.model
    logger.info(f"ENT-LLM Ablation Analysis CLI")
    logger.info(f"Model: {args.model}")

    # Load input
    logger.info(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)
    validate_input(df)
    logger.info(f"Loaded {len(df)} cases")

    # Optional token-length filter
    if args.max_tokens:
        before = len(df)
        df = filter_long_cases(df, max_tokens=args.max_tokens)
        logger.info(f"Token filter ({args.max_tokens}): {before} -> {len(df)} cases")

    # Stratified sample
    if args.sample_size:
        logger.info(f"Creating stratified sample of {args.sample_size} cases")
        df = stratified_sample(df, args.sample_size)
        logger.info(f"Sample size after stratification: {len(df)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build experiment list
    experiments = build_experiment_list(args.experiments)
    logger.info(f"Planned experiments: {len(experiments)}")

    # Handle pre-computed baseline
    if args.baseline:
        if not os.path.exists(args.baseline):
            logger.error(f"Baseline file not found: {args.baseline}")
            return 1
        baseline_dest = os.path.join(args.output_dir, "baseline_results.csv")
        if os.path.abspath(args.baseline) != os.path.abspath(baseline_dest):
            shutil.copy2(args.baseline, baseline_dest)
            logger.info(f"Copied baseline to {baseline_dest}")
        # Remove baseline from experiments list since it's pre-computed
        experiments = [(name, evars) for name, evars in experiments if name != "baseline"]
        logger.info(f"Using pre-computed baseline, {len(experiments)} experiments remaining")

    # Run experiments
    try:
        for exp_name, exclude_vars in experiments:
            output_file = os.path.join(args.output_dir, f"{exp_name}_results.csv")
            logger.info(f"Starting experiment: {exp_name}")
            run_ablation_experiment(
                df=df,
                exclude_vars=exclude_vars,
                experiment_name=exp_name,
                output_file=output_file,
                model_name=args.model,
                delay_seconds=args.delay,
                flush_interval=args.flush_interval,
                resume=not args.no_resume,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved incrementally.")
        return 130

    # Analyze results
    logger.info("Analyzing results...")
    ground_truth_df = None
    if args.ground_truth in df.columns:
        ground_truth_df = df

    summary_df = analyze_ablation_results(
        results_dir=args.output_dir,
        ground_truth_df=ground_truth_df,
        ground_truth_col=args.ground_truth,
    )

    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    print_summary_table(summary_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
