#!/usr/bin/env python3
# This source file is part of the ARPA-H CARE LLM project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
CLI entrypoint for ENT-LLM analysis.

This module provides a command-line interface to run the LLM analysis
with different model backends.
"""

import argparse
import logging
import sys
from typing import Optional

import pandas as pd

from llm_query.securellm_adapter import ModelConfig, query_llm, SecureLLMClient
from llm_query.LLM_analysis import (
    generate_prompt,
    parse_llm_response,
    process_llm_cases,
    run_llm_analysis,
)

# Available LLM models
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


def run_single_query(model: str, prompt: str) -> Optional[str]:
    """
    Run a single LLM query with the specified model.

    Args:
        model: The model identifier to use.
        prompt: The prompt to send to the LLM.

    Returns:
        The LLM response or None on error.
    """
    client = SecureLLMClient(model_name=model)
    return client.query(prompt)


def run_analysis_with_model(
    model: str,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    delay_seconds: float = 0.2,
    flush_interval: int = 10,
    no_resume: bool = False,
    start_row: int = 0,
    end_row: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full LLM analysis pipeline with a specified model.

    Args:
        model: The model identifier to use.
        input_file: Path to input CSV file with case data.
        output_file: Path to save results CSV (saved incrementally).
        delay_seconds: Delay between API calls.
        flush_interval: Number of cases to process before flushing to disk.
        no_resume: If True, start fresh instead of resuming from existing output.
        start_row: Start processing from this row index (0-based).

    Returns:
        DataFrame with analysis results.
    """
    logger = logging.getLogger(__name__)

    # Update the default model configuration
    ModelConfig.DEFAULT_LLM_MODEL = model
    logger.info(f"Using model: {model}")

    if input_file:
        logger.info(f"Loading data from: {input_file}")
        llm_df = pd.read_csv(input_file)

        # Validate required columns
        required_cols = ["llm_caseID", "formatted_progress_text", "formatted_radiology_text"]
        missing_cols = [col for col in required_cols if col not in llm_df.columns]
        if missing_cols:
            raise ValueError(f"Input file missing required columns: {missing_cols}")

        # Apply row range filter
        total_rows = len(llm_df)
        if start_row > 0 or end_row is not None:
            if start_row >= total_rows:
                raise ValueError(f"start_row ({start_row}) is >= total rows ({total_rows})")
            if end_row is not None and end_row <= start_row:
                raise ValueError(f"end_row ({end_row}) must be greater than start_row ({start_row})")
            actual_end = min(end_row, total_rows) if end_row is not None else total_rows
            logger.info(f"Processing rows {start_row} to {actual_end - 1} (of {total_rows} total)")
            llm_df = llm_df.iloc[start_row:actual_end].reset_index(drop=True)

        # Run analysis with incremental saving
        results_df = run_llm_analysis(
            llm_df,
            output_file=output_file,
            flush_interval=flush_interval,
            resume=not no_resume
        )

        if output_file:
            logger.info(f"Results saved to: {output_file}")

        return results_df
    else:
        logger.warning("No input file provided. Running in demo mode.")
        # Demo mode: create a simple test case
        demo_data = {
            "llm_caseID": ["DEMO_001"],
            "formatted_progress_text": [
                "Patient presents with chronic rhinosinusitis refractory to medical management "
                "including multiple courses of antibiotics and intranasal corticosteroids. "
                "Symptoms include persistent nasal congestion, facial pressure, and purulent discharge "
                "for over 12 weeks. Previous treatments have failed to provide lasting relief."
            ],
            "formatted_radiology_text": [
                "CT Sinuses: Mucosal thickening in bilateral maxillary sinuses with partial "
                "opacification. Ostiomeatal complex obstruction bilaterally. No bony erosion."
            ],
        }
        demo_df = pd.DataFrame(demo_data)

        logger.info("Processing demo case...")
        results_df = run_llm_analysis(demo_df)

        print("\n=== Demo Results ===")
        for _, row in results_df.iterrows():
            print(f"Case ID: {row['llm_caseID']}")
            print(f"Decision: {row['decision']}")
            print(f"Confidence: {row['confidence']}")
            print(f"Reasoning: {row['reasoning']}")

        return results_df


def interactive_query(model: str) -> None:
    """
    Run an interactive query session with the specified model.

    Args:
        model: The model identifier to use.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting interactive session with model: {model}")
    print(f"\nInteractive query mode with {model}")
    print("Type 'quit' or 'exit' to end the session.\n")

    client = SecureLLMClient(model_name=model)

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not prompt:
                continue

            response = client.query(
                prompt,
                system_message="You are an expert otolaryngologist. Answer questions about ENT cases.",
            )
            if response:
                print(f"\nAssistant: {response}\n")
            else:
                print("\n[No response received]\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n[Error: {e}]\n")


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="ENT-LLM Analysis CLI - Run clinical case analysis with various LLM backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with default model (demo mode)
  python -m cli --model apim:gpt-4.1

  # Run analysis on a CSV file (saves incrementally every 10 cases)
  python -m cli --model apim:claude-3.7 --input cases.csv --output results.csv

  # Run with custom flush interval (save every 5 cases)
  python -m cli --model apim:gpt-4.1 -i cases.csv -o results.csv --flush-interval 5

  # Start from a specific row (e.g., skip first 100 rows)
  python -m cli --model apim:gpt-4.1 -i cases.csv -o results.csv --start-row 100

  # Process a range of rows (for parallel execution)
  python -m cli -m apim:gpt-4.1 -i cases.csv -o results_0_500.csv -s 0 -e 500
  python -m cli -m apim:gpt-4.1 -i cases.csv -o results_500_1000.csv -s 500 -e 1000

  # Start fresh (don't resume from existing output)
  python -m cli --model apim:gpt-4.1 -i cases.csv -o results.csv --no-resume

  # Interactive query mode
  python -m cli --model apim:llama-3.3-70b --interactive

  # List available models
  python -m cli --list-models
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=AVAILABLE_MODELS,
        default="apim:gpt-4.1",
        help="LLM model to use for analysis (default: apim:gpt-4.1)",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input CSV file with case data (columns: llm_caseID, formatted_progress_text, formatted_radiology_text)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file for results",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.2,
        help="Delay in seconds between API calls (default: 0.2)",
    )

    parser.add_argument(
        "--flush-interval",
        "-f",
        type=int,
        default=10,
        help="Number of cases to process before flushing to disk (default: 10)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output file",
    )

    parser.add_argument(
        "--start-row",
        "-s",
        type=int,
        default=0,
        help="Start processing from this row index (0-based, default: 0)",
    )

    parser.add_argument(
        "--end-row",
        "-e",
        type=int,
        default=None,
        help="Stop processing at this row index (exclusive, 0-based). If omitted, processes to the end.",
    )

    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        help="Run in interactive query mode",
    )

    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available models and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("Available models:")
        for model in AVAILABLE_MODELS:
            print(f"  - {model}")
        return 0

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"ENT-LLM Analysis CLI")
    logger.info(f"Selected model: {args.model}")

    try:
        if args.interactive:
            interactive_query(args.model)
        else:
            run_analysis_with_model(
                model=args.model,
                input_file=args.input,
                output_file=args.output,
                delay_seconds=args.delay,
                flush_interval=args.flush_interval,
                no_resume=args.no_resume,
                start_row=args.start_row,
                end_row=args.end_row,
            )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
