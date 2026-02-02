#!/usr/bin/env python3
# This source file is part of the ARPA-H CARE LLM project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
CLI entrypoint for BigQuery data extraction.

This module provides a command-line interface to extract and preprocess
ENT clinical data from BigQuery for LLM analysis.
"""

import argparse
import gc
import logging
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from google.cloud import bigquery

from data_extraction.config import (
    PROJECT_ID,
    DATASET_IDS,
    DATA_TABLES,
    CLINICAL_NOTE_TYPES,
    CLINICAL_NOTE_TITLES,
    RADIOLOGY_REPORT_TYPE,
    RADIOLOGY_REPORT_TITLE,
    SURGERY_CPT_CODES,
    DIAGNOSTIC_ENDOSCOPY_CPT_CODES,
)
from data_extraction.raw_data_parsing import (
    extract_ent_notes,
    extract_radiology_reports,
    procedures_df,
    build_patient_df,
)
from data_extraction.note_extraction import (
    add_last_progress_note,
    recursive_censor_notes,
)
from llm_query.llm_input import create_llm_dataframe


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of patient data from BigQuery."""

    def __init__(
        self,
        project_id: str,
        dataset_ids: List[str],
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_ids = dataset_ids
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.patient_identifier = "patient_id"
        self.logger = logging.getLogger(__name__)

    def get_total_patient_count(self) -> int:
        """Get total number of patients with clinical notes."""
        notes_union = "\nUNION ALL\n".join(
            f"SELECT {self.patient_identifier} FROM `{self.project_id}.{ds}.clinical_note`"
            for ds in self.dataset_ids
        )

        count_query = f"""
        WITH all_notes AS (
            SELECT DISTINCT {self.patient_identifier} FROM ({notes_union})
        )
        SELECT COUNT(*) as total_patients
        FROM all_notes
        """

        result = self.client.query(count_query).to_dataframe()
        return int(result["total_patients"].iloc[0])

    def get_patient_batches(self, limit: Optional[int] = None) -> Iterator[List[str]]:
        """Generator that yields batches of patient IDs."""
        notes_union = "\nUNION ALL\n".join(
            f"SELECT {self.patient_identifier} FROM `{self.project_id}.{ds}.clinical_note`"
            for ds in self.dataset_ids
        )

        all_patients_query = f"""
        WITH all_notes AS (
            SELECT DISTINCT {self.patient_identifier} FROM ({notes_union})
        )
        SELECT {self.patient_identifier}
        FROM all_notes
        ORDER BY {self.patient_identifier}
        """

        offset = 0
        total_yielded = 0

        while True:
            # Adjust batch size if we're near the limit
            current_batch_size = self.batch_size
            if limit is not None:
                remaining = limit - total_yielded
                if remaining <= 0:
                    break
                current_batch_size = min(self.batch_size, remaining)

            batch_query = f"""
            {all_patients_query}
            LIMIT {current_batch_size} OFFSET {offset}
            """

            batch_df = self.client.query(batch_query).to_dataframe()

            if batch_df.empty:
                break

            patient_ids = batch_df[self.patient_identifier].tolist()
            yield patient_ids

            total_yielded += len(patient_ids)
            offset += self.batch_size

            del batch_df
            gc.collect()

    def extract_batch_data(
        self, patient_ids: List[str], table_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Extract data for a batch of patients."""
        batch_data = {}
        id_list_str = ", ".join(f"'{pid}'" for pid in patient_ids)

        self.logger.info(f"Extracting data for {len(patient_ids)} patients...")

        for table in table_names:
            self.logger.debug(f"Loading table: {table}")

            for attempt in range(self.max_retries):
                try:
                    union_query = "\nUNION ALL\n".join(
                        f"SELECT * FROM `{self.project_id}.{ds}.{table}`"
                        for ds in self.dataset_ids
                    )

                    full_query = f"""
                    SELECT * FROM ({union_query})
                    WHERE {self.patient_identifier} IN ({id_list_str})
                    """

                    job_config = bigquery.QueryJobConfig(
                        use_query_cache=True, use_legacy_sql=False
                    )

                    df = self.client.query(full_query, job_config=job_config).to_dataframe()
                    batch_data[table] = df
                    self.logger.debug(f"  {table}: {df.shape[0]} rows loaded")
                    break

                except Exception as e:
                    self.logger.warning(f"  Attempt {attempt + 1} failed for '{table}': {e}")
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"  Failed to load '{table}' after {self.max_retries} attempts")
                        batch_data[table] = pd.DataFrame()
                    else:
                        time.sleep(2**attempt)

        return batch_data


def process_batch(
    batch_data: Dict[str, pd.DataFrame],
    patient_ids: List[str],
    global_case_id_counter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Process a single batch of patient data."""
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Processing batch of {len(patient_ids)} patients...")

        # Extract ENT notes
        if "clinical_note" in batch_data and not batch_data["clinical_note"].empty:
            ent_notes = extract_ent_notes(
                batch_data["clinical_note"],
                CLINICAL_NOTE_TYPES,
                CLINICAL_NOTE_TITLES,
            )
            logger.debug(f"  Found {len(ent_notes)} ENT notes")
        else:
            ent_notes = pd.DataFrame()

        # Extract radiology reports
        if "radiology_report" in batch_data and not batch_data["radiology_report"].empty:
            rad_reports = extract_radiology_reports(
                batch_data["radiology_report"],
                RADIOLOGY_REPORT_TYPE,
                RADIOLOGY_REPORT_TITLE,
            )
            logger.debug(f"  Found {len(rad_reports)} radiology reports")
        else:
            rad_reports = pd.DataFrame()

        # Process procedures
        if "procedures" in batch_data and not batch_data["procedures"].empty:
            procedures = procedures_df(
                batch_data["procedures"],
                SURGERY_CPT_CODES,
                DIAGNOSTIC_ENDOSCOPY_CPT_CODES,
            )
            logger.debug(f"  Found {len(procedures)} procedure records")
        else:
            procedures = pd.DataFrame()

        # Check if we have any data
        if ent_notes.empty and rad_reports.empty and procedures.empty:
            logger.debug("No relevant data found in this batch")
            return pd.DataFrame(), pd.DataFrame(), global_case_id_counter

        # Build patient dataframe
        patient_df = build_patient_df(ent_notes, rad_reports, procedures)

        if patient_df.empty:
            return pd.DataFrame(), pd.DataFrame(), global_case_id_counter

        logger.debug(f"  Patient dataframe: {len(patient_df)} patients")

        # Add progress notes
        patient_df_with_progress = add_last_progress_note(patient_df)

        # Censor notes
        processed_df, skipped_ids = recursive_censor_notes(patient_df_with_progress)
        logger.debug(f"  After censoring: {len(processed_df)} patients, {len(skipped_ids)} skipped")

        # Create sequential case IDs
        if not processed_df.empty:
            case_ids = range(global_case_id_counter, global_case_id_counter + len(processed_df))
            processed_df["llm_caseID"] = list(case_ids)
            new_counter = global_case_id_counter + len(processed_df)
        else:
            new_counter = global_case_id_counter

        # Format for LLM input
        llm_df = create_llm_dataframe(processed_df) if not processed_df.empty else pd.DataFrame()

        # Add radiology flag
        if not processed_df.empty:
            processed_df["has_radiology"] = [
                arr.size > 0 if hasattr(arr, "size") else len(arr) > 0
                for arr in processed_df["radiology_reports"]
            ]

        return llm_df, processed_df, new_counter

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), global_case_id_counter


def run_extraction(
    output_file: str,
    batch_size: int = 100,
    limit: Optional[int] = None,
    save_processed: bool = False,
    processed_output: Optional[str] = None,
    checkpoint_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full data extraction pipeline.

    Args:
        output_file: Path to save the LLM-ready CSV.
        batch_size: Number of patients per batch.
        limit: Maximum number of patients to process (None for all).
        save_processed: Whether to also save the processed dataframe.
        processed_output: Path for processed dataframe CSV.
        checkpoint_interval: Save checkpoint every N batches.
        checkpoint_dir: Directory for checkpoint files.

    Returns:
        Tuple of (llm_df, processed_df).
    """
    logger = logging.getLogger(__name__)

    logger.info("Initializing BigQuery batch processor...")
    processor = BatchProcessor(PROJECT_ID, DATASET_IDS, batch_size=batch_size)

    # Get total patient count
    try:
        total_patients = processor.get_total_patient_count()
        if limit:
            total_patients = min(total_patients, limit)
        logger.info(f"Total patients to process: {total_patients}")
    except Exception as e:
        logger.error(f"Error getting patient count: {e}")
        return pd.DataFrame(), pd.DataFrame()

    all_llm_data = []
    all_processed_data = []
    global_case_id_counter = 1
    batch_num = 0
    start_time = time.time()

    try:
        for patient_batch in processor.get_patient_batches(limit=limit):
            batch_num += 1
            batch_start = time.time()

            logger.info(f"{'=' * 50}")
            logger.info(f"BATCH {batch_num}")
            logger.info(f"{'=' * 50}")

            # Extract batch data
            batch_data = processor.extract_batch_data(patient_batch, DATA_TABLES)

            # Process the batch
            llm_df, processed_df, global_case_id_counter = process_batch(
                batch_data, patient_batch, global_case_id_counter
            )

            # Collect results
            if not llm_df.empty:
                all_llm_data.append(llm_df)
            if not processed_df.empty:
                all_processed_data.append(processed_df)

            # Clean up memory
            del batch_data
            gc.collect()

            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - start_time
            logger.info(
                f"Batch {batch_num} completed in {batch_elapsed:.1f}s. "
                f"Total cases: {global_case_id_counter - 1}. "
                f"Total time: {total_elapsed:.1f}s"
            )

            # Save checkpoint
            if checkpoint_dir and batch_num % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_batch_{batch_num}.pkl"
                if all_llm_data:
                    temp_df = pd.concat(all_llm_data, ignore_index=True)
                    temp_df.to_pickle(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving partial results...")
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        import traceback
        traceback.print_exc()

    # Combine all results
    if all_llm_data:
        final_llm_df = pd.concat(all_llm_data, ignore_index=True)
        final_processed_df = pd.concat(all_processed_data, ignore_index=True) if all_processed_data else pd.DataFrame()

        # Save LLM-ready CSV
        final_llm_df.to_csv(output_file, index=False)
        logger.info(f"LLM data saved to: {output_file}")
        logger.info(f"Total cases: {len(final_llm_df)}")

        # Optionally save processed dataframe
        if save_processed and processed_output:
            # Convert complex columns to JSON strings for CSV compatibility
            save_df = final_processed_df.copy()
            for col in save_df.columns:
                if save_df[col].dtype == object:
                    try:
                        save_df[col] = save_df[col].apply(
                            lambda x: str(x) if isinstance(x, (list, dict)) else x
                        )
                    except Exception:
                        pass
            save_df.to_csv(processed_output, index=False)
            logger.info(f"Processed data saved to: {processed_output}")

        total_time = time.time() - start_time
        logger.info(f"Extraction complete in {total_time:.1f}s ({total_time/60:.1f} min)")

        return final_llm_df, final_processed_df
    else:
        logger.warning("No data extracted")
        return pd.DataFrame(), pd.DataFrame()


def main() -> int:
    """Main CLI entrypoint for data extraction."""
    parser = argparse.ArgumentParser(
        description="ENT-LLM Data Extraction CLI - Extract clinical data from BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all data to a CSV file
  python cli_extract.py --output cases.csv

  # Extract with a patient limit (for testing)
  python cli_extract.py --output cases.csv --limit 100

  # Extract with custom batch size
  python cli_extract.py --output cases.csv --batch-size 200

  # Extract and save both LLM and processed data
  python cli_extract.py --output cases.csv --save-processed --processed-output processed.csv

  # Extract with checkpoints
  python cli_extract.py --output cases.csv --checkpoint-dir ./checkpoints

  # Show patient count only
  python cli_extract.py --count-only
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="llm_cases.csv",
        help="Output CSV file for LLM-ready data (default: llm_cases.csv)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Number of patients per batch (default: 100)",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of patients to process (default: all)",
    )

    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Also save the full processed dataframe",
    )

    parser.add_argument(
        "--processed-output",
        type=str,
        default="processed_data.csv",
        help="Output file for processed data (default: processed_data.csv)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoint files",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N batches (default: 10)",
    )

    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only show total patient count and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("ENT-LLM Data Extraction CLI")

    try:
        if args.count_only:
            # Just show patient count
            logger.info("Counting patients in BigQuery...")
            processor = BatchProcessor(PROJECT_ID, DATASET_IDS, batch_size=100)
            total = processor.get_total_patient_count()
            print(f"Total patients with clinical notes: {total}")
            return 0

        # Create checkpoint directory if specified
        if args.checkpoint_dir:
            import os
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Run extraction
        llm_df, processed_df = run_extraction(
            output_file=args.output,
            batch_size=args.batch_size,
            limit=args.limit,
            save_processed=args.save_processed,
            processed_output=args.processed_output,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )

        if llm_df.empty:
            logger.error("No data was extracted")
            return 1

        # Print summary
        print(f"\n{'=' * 50}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total cases extracted: {len(llm_df)}")
        print(f"Output file: {args.output}")
        if args.save_processed:
            print(f"Processed file: {args.processed_output}")
        print(f"\nNext step: Run LLM analysis with:")
        print(f"  python cli.py --model apim:gpt-4.1 --input {args.output} --output results.csv")

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
