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
) -> pd.DataFrame:
    """
    Run the full LLM analysis pipeline with a specified model.

    Args:
        model: The model identifier to use.
        input_file: Path to input CSV file with case data.
        output_file: Path to save results CSV.
        delay_seconds: Delay between API calls.

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

        results_df = run_llm_analysis(llm_df)

        if output_file:
            results_df.to_csv(output_file, index=False)
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

  # Run analysis on a CSV file
  python -m cli --model apim:claude-3.7 --input cases.csv --output results.csv

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
