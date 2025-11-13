from google.cloud import bigquery
from typing import List, Dict, Iterator, Tuple
import pandas as pd
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class BatchProcessor:
    """Handles batch processing of patient data."""

    def __init__(self, project_id: str, dataset_ids: List[str],
                 batch_size: int = 100, max_retries: int = 3):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_ids = dataset_ids
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.patient_identifier = 'patient_id'

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
        return int(result['total_patients'].iloc[0])

    def get_patient_batches(self, max_patients: int = None) -> Iterator[List[str]]:
        """Generator that yields batches of patient IDs."""
        notes_union = "\nUNION ALL\n".join(
            f"SELECT {self.patient_identifier} FROM `{self.project_id}.{ds}.clinical_note`"
            for ds in self.dataset_ids
        )

        base_query = f"""
        WITH all_notes AS (
            SELECT DISTINCT {self.patient_identifier} FROM ({notes_union})
        )
        SELECT {self.patient_identifier}
        FROM all_notes
        ORDER BY {self.patient_identifier}
        """

        offset = 0
        patients_yielded = 0

        while True:
            current_batch_size = self.batch_size
            if max_patients is not None:
                remaining = max_patients - patients_yielded
                if remaining <= 0:
                    break
                current_batch_size = min(self.batch_size, remaining)

            batch_query = f"""
            {base_query}
            LIMIT {current_batch_size} OFFSET {offset}
            """

            batch_df = self.client.query(batch_query).to_dataframe()

            if batch_df.empty:
                break

            patient_ids = batch_df[self.patient_identifier].tolist()
            yield patient_ids

            patients_yielded += len(patient_ids)
            offset += current_batch_size

            del batch_df
            gc.collect()

    def extract_batch_data(self, patient_ids: List[str],
                          table_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract all data for a batch of patients."""
        batch_data = {}
        id_list_str = ", ".join(f"'{pid}'" for pid in patient_ids)

        for table in table_names:
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
                        use_query_cache=True,
                        use_legacy_sql=False,
                        priority=bigquery.QueryPriority.INTERACTIVE
                    )

                    df = self.client.query(full_query, job_config=job_config).to_dataframe()
                    batch_data[table] = df
                    break

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        batch_data[table] = pd.DataFrame()
                    else:
                        time.sleep(2 ** attempt)

        return batch_data


def process_batch_wrapper(args: Tuple) -> Tuple[int, pd.DataFrame, pd.DataFrame, int]:
    """Wrapper function for parallel processing.

    Returns:
        (batch_idx, llm_df, processed_df, num_cases)
    """
    (batch_idx, batch_data, patient_ids, surgery_cpt_codes, radiology_types,
     radiology_titles, clinical_note_types, clinical_note_titles) = args

    try:
        print(f"Processing batch {batch_idx + 1} with {len(patient_ids)} patients...")

        # Extract ENT notes
        if 'clinical_note' in batch_data and not batch_data['clinical_note'].empty:
            ent_notes = extract_ent_notes(
                batch_data["clinical_note"],
                clinical_note_types,
                clinical_note_titles
            )
        else:
            ent_notes = pd.DataFrame()

        if ent_notes.empty:
            return (batch_idx, pd.DataFrame(), pd.DataFrame(), 0)

        # Build patient dataframe
        patient_df = build_patient_df(
            ent_df=ent_notes,
            radiology_df=batch_data.get('radiology_report', pd.DataFrame()),
            procedures_df=batch_data.get('procedures', pd.DataFrame()),
            demographics_df=batch_data.get('demographics', pd.DataFrame()),
            surgery_cpt_codes=surgery_cpt_codes,
            radiology_types=radiology_types,
            radiology_titles=radiology_titles
        )

        if patient_df.empty:
            return (batch_idx, pd.DataFrame(), pd.DataFrame(), 0)

        # Add and redact notes
        patient_df_with_progress = add_last_progress_note(patient_df)
        processed_df, skipped_ids = recursive_censor_notes(patient_df_with_progress)

        if processed_df.empty:
            return (batch_idx, pd.DataFrame(), pd.DataFrame(), 0)

        # Add has_radiology flag
        processed_df['has_radiology'] = processed_df['radiology_reports'].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else False
        )

        # Add temporary case ID, this will get relabeled in processing
        num_cases = len(processed_df)
        processed_df['llm_caseID'] = range(num_cases)

        # Create LLM dataframe
        llm_df = create_llm_dataframe(processed_df)

        num_cases = len(processed_df)
        print(f"Batch {batch_idx + 1} completed: {num_cases} cases")

        return (batch_idx, llm_df, processed_df, num_cases)

    except Exception as e:
        print(f"Error in batch {batch_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        return (batch_idx, pd.DataFrame(), pd.DataFrame(), 0)


def main_batch_processing_parallel(surgery_cpt_codes: List[str],
                                   radiology_types: List[str],
                                   radiology_titles: List[str],
                                   clinical_note_types: List[str],
                                   clinical_note_titles: List[str],
                                   project_id: str,
                                   dataset_ids: List[str],
                                   data_tables: List[str],
                                   max_patients: int = None,
                                   max_workers: int = 4,
                                   prefetch_batches: int = 8,
                                   checkpoint_dir: str = './checkpoints'):
    """Main function with parallel batch processing and checkpointing.

    Args:
        max_workers: Number of parallel workers (default 4)
        prefetch_batches: Number of batches to fetch ahead (default 8)
        checkpoint_dir: Directory to save checkpoints (default './checkpoints')
    """

    import os

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    processor = BatchProcessor(project_id, dataset_ids, batch_size=100)

    try:
        total_patients = processor.get_total_patient_count()
        print(f"Total patients available: {total_patients}")
        if max_patients:
            print(f"Processing first {max_patients} patients with {max_workers} parallel workers")
        else:
            print(f"Processing all {total_patients} patients with {max_workers} parallel workers")
        print(f"Fetching {prefetch_batches} batches at a time ({prefetch_batches * 100} patients per group)")
    except Exception as e:
        print(f"Error getting patient count: {e}")
        return pd.DataFrame(), pd.DataFrame()

    start_time = time.time()
    all_llm_data = []
    all_processed_data = []
    case_id_counter = 1
    batch_num = 0
    total_batches_processed = 0
    checkpoint_num = 0
    group_times = []

    try:
        # Process in groups
        batch_generator = processor.get_patient_batches(max_patients=max_patients)

        while True:
            group_start_time = time.time()
            # Fetch a group of batches
            print(f"\n{'='*60}")
            print(f"FETCHING BATCH GROUP {checkpoint_num + 1}")
            print(f"{'='*60}")
            fetch_start = time.time()
            batch_queue = []

            for _ in range(prefetch_batches):
                try:
                    patient_batch = next(batch_generator)
                    batch_num += 1
                    print(f"  Fetching batch {batch_num}...")
                    batch_data = processor.extract_batch_data(patient_batch, data_tables)

                    batch_queue.append((
                        batch_num - 1,  # 0-indexed for sorting
                        batch_data,
                        patient_batch,
                        surgery_cpt_codes,
                        radiology_types,
                        radiology_titles,
                        clinical_note_types,
                        clinical_note_titles
                    ))
                except StopIteration:
                    break

            if not batch_queue:
                break

            fetch_time = time.time() - fetch_start
            print(f"\nProcessing {len(batch_queue)} batches in parallel...")

            # Process this group in parallel
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_batch_wrapper, batch): batch[0]
                                 for batch in batch_queue}

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        result = future.result()
                        results.append(result)
                        total_batches_processed += 1
                        print(f"  Batch {result[0] + 1} completed: {result[3]} cases")
                    except Exception as e:
                        print(f"  Batch {batch_idx + 1} failed: {e}")
                        results.append((batch_idx, pd.DataFrame(), pd.DataFrame(), 0))

            # Sort results and assign case IDs
            results.sort(key=lambda x: x[0])

            group_llm_data = []
            group_processed_data = []

            for batch_idx, llm_df, processed_df, num_cases in results:
                if not processed_df.empty:
                    processed_df['llm_caseID'] = range(case_id_counter, case_id_counter + num_cases)
                    case_id_counter += num_cases
                    all_llm_data.append(llm_df)
                    all_processed_data.append(processed_df)
                    group_llm_data.append(llm_df)
                    group_processed_data.append(processed_df)

            # Save checkpoint for this group
            checkpoint_num += 1
            if group_llm_data:
                checkpoint_llm = pd.concat(group_llm_data, ignore_index=True)
                checkpoint_processed = pd.concat(group_processed_data, ignore_index=True)

                llm_checkpoint_path = os.path.join(checkpoint_dir, f'llm_checkpoint_{checkpoint_num}.parquet')
                processed_checkpoint_path = os.path.join(checkpoint_dir, f'processed_checkpoint_{checkpoint_num}.parquet')

                checkpoint_llm.to_parquet(llm_checkpoint_path)
                checkpoint_processed.to_parquet(processed_checkpoint_path)

                print(f"\n✓ Checkpoint {checkpoint_num} saved:")
                print(f"  - {llm_checkpoint_path}")
                print(f"  - {processed_checkpoint_path}")

            # Clean up
            del batch_queue, results, group_llm_data, group_processed_data
            gc.collect()

            # Track group timing
            group_time = time.time() - group_start_time
            group_times.append(group_time)

            # Calculate progress and estimates
            elapsed = time.time() - start_time
            print(f"\nProgress: {total_batches_processed} batches completed, {case_id_counter - 1} total cases")
            print(f"Elapsed time: {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours)")

        total_time = time.time() - start_time

        if all_llm_data:
            final_llm_df = pd.concat(all_llm_data, ignore_index=True)
            final_processed_df = pd.concat(all_processed_data, ignore_index=True)

            # Save final results
            final_llm_path = os.path.join(checkpoint_dir, 'final_llm_data.parquet')
            final_processed_path = os.path.join(checkpoint_dir, 'final_processed_data.parquet')

            final_llm_df.to_parquet(final_llm_path)
            final_processed_df.to_parquet(final_processed_path)

            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Final results: {len(final_llm_df)} cases for LLM processing")
            print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
            print(f"\nFinal files saved:")
            print(f"  - {final_llm_path}")
            print(f"  - {final_processed_path}")
            print(f"\nCheckpoints saved in: {checkpoint_dir}/")

            return final_llm_df, final_processed_df
        else:
            print("No data processed successfully")
            return pd.DataFrame(), pd.DataFrame()

    except Exception as e:
        print(f"Error in parallel batch processing: {e}")
        import traceback
        traceback.print_exc()

        # Save progress so far if there's an error
        if all_llm_data:
            print("\nSaving progress before exit...")
            emergency_llm = pd.concat(all_llm_data, ignore_index=True)
            emergency_processed = pd.concat(all_processed_data, ignore_index=True)

            emergency_llm.to_parquet(os.path.join(checkpoint_dir, 'emergency_llm_data.parquet'))
            emergency_processed.to_parquet(os.path.join(checkpoint_dir, 'emergency_processed_data.parquet'))
            print(f"Emergency checkpoint saved in {checkpoint_dir}/")

        return pd.DataFrame(), pd.DataFrame()
    

# USAGE
# Run batch processing!
llm_df, processed_df = main_batch_processing_parallel(
    surgery_cpt_codes=SURGERY_CPT_CODES,
    radiology_types=RADIOLOGY_REPORT_TYPE,
    radiology_titles=RADIOLOGY_REPORT_TITLE,
    clinical_note_types=CLINICAL_NOTE_TYPES,
    clinical_note_titles=CLINICAL_NOTE_TITLES,
    project_id=PROJECT_ID,
    dataset_ids=DATASET_IDS,
    data_tables=DATA_TABLES,
    max_workers=4,
    checkpoint_dir='./my_checkpoints'
)