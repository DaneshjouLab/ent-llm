from google.cloud import bigquery
from typing import List, Dict, Tuple, Iterator
import pandas as pd
import gc
import time
from multiprocessing import Pool, Manager
from functools import partial
import os

class BatchProcessor:
    """Handles batch processing of patient data with multiprocessing support."""

    def __init__(self, project_id: str, dataset_ids: List[str],
                 batch_size: int = 100, max_retries: int = 3, num_workers: int = 4):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_ids = dataset_ids
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.num_workers = num_workers
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

    def get_patient_batches(self) -> Iterator[List[str]]:
        """Generator that yields batches of patient IDs."""
        notes_union = "\nUNION ALL\n".join(
            f"SELECT {self.patient_identifier} FROM `{self.project_id}.{ds}.clinical_note`"
            for ds in self.dataset_ids
        )

        # Get all patient IDs, ordered for consistent batching (same as extract_sample)
        all_patients_query = f"""
        WITH all_notes AS (
            SELECT DISTINCT {self.patient_identifier} FROM ({notes_union})
        )
        SELECT {self.patient_identifier}
        FROM all_notes
        ORDER BY {self.patient_identifier}
        """

        # Use pagination to avoid loading all patient IDs at once
        offset = 0
        while True:
            batch_query = f"""
            {all_patients_query}
            LIMIT {self.batch_size} OFFSET {offset}
            """

            batch_df = self.client.query(batch_query).to_dataframe()

            if batch_df.empty:
                break

            patient_ids = batch_df[self.patient_identifier].tolist()
            yield patient_ids

            offset += self.batch_size

            # Clean up memory
            del batch_df
            gc.collect()

    def extract_batch_data(self, patient_ids: List[str],
                          table_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract all data for a batch of patients."""
        batch_data = {}

        # Format patient IDs for SQL IN clause (same as extract_sample)
        id_list_str = ", ".join(f"'{pid}'" for pid in patient_ids)

        print(f"Extracting data for {len(patient_ids)} patients...")

        # Extract patient data from each table for patients
        for table in table_names:
            print(f"Loading table: {table}")

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

                    # Use job config to optimize query
                    job_config = bigquery.QueryJobConfig(
                        use_query_cache=True,
                        use_legacy_sql=False
                    )

                    df = self.client.query(full_query, job_config=job_config).to_dataframe()
                    batch_data[table] = df
                    print(f"  {df.shape[0]} rows loaded.")
                    break

                except Exception as e:
                    print(f"  Attempt {attempt + 1} failed for table '{table}': {e}")
                    if attempt == self.max_retries - 1:
                        print(f"  Failed to load '{table}' after {self.max_retries} attempts")
                        batch_data[table] = pd.DataFrame()
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff

        return batch_data


def process_batch(
    batch_args: Tuple[List[str], int, str, List[str], List[str], List[str], 
                      List[str], List[str], List[str], List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    (patient_ids, batch_id, project_id, dataset_ids, data_tables,
     surgery_cpt_codes, radiology_types, radiology_titles,
     clinical_note_types, clinical_note_titles) = batch_args
    
    try:
        print(f"Worker batch {batch_id}: Starting processing of {len(patient_ids)} patients")
        
        # Create new BigQuery client for this worker process
        client = bigquery.Client(project=project_id)
        
        # Extract batch data (same logic as original)
        batch_data = {}
        id_list_str = ", ".join(f"'{pid}'" for pid in patient_ids)
        
        for table in data_tables:
            try:
                union_query = "\nUNION ALL\n".join(
                    f"SELECT * FROM `{project_id}.{ds}.{table}`"
                    for ds in dataset_ids
                )

                full_query = f"""
                SELECT * FROM ({union_query})
                WHERE patient_id IN ({id_list_str})
                """

                job_config = bigquery.QueryJobConfig(
                    use_query_cache=True,
                    use_legacy_sql=False
                )

                df = client.query(full_query, job_config=job_config).to_dataframe()
                batch_data[table] = df
                print(f"Worker batch {batch_id}: Loaded {df.shape[0]} rows from {table}")
                
            except Exception as e:
                print(f"Worker batch {batch_id}: Error loading {table}: {e}")
                batch_data[table] = pd.DataFrame()

        # Process the batch
        if 'clinical_note' in batch_data and not batch_data['clinical_note'].empty:
            ent_notes = extract_ent_notes(
                batch_data["clinical_note"],
                clinical_note_types,
                clinical_note_titles
            )
        else:
            ent_notes = pd.DataFrame()

        if ent_notes.empty:
            print(f"Worker batch {batch_id}: No ENT notes found - skipping")
            return pd.DataFrame(), pd.DataFrame(), 0, batch_id

        # Prepare data tables
        radiology_data = batch_data.get('radiology_report', pd.DataFrame())
        procedures_data = batch_data.get('procedures', pd.DataFrame())
        demographics_data = batch_data.get('demographics', pd.DataFrame())
        lab_data = batch_data.get('labs', pd.DataFrame())

        # Build patient dataframe
        patient_df = build_patient_df(
            ent_df=ent_notes,
            radiology_df=radiology_data,
            procedures_df=procedures_data,
            demographics_df=demographics_data,
            lab_df=lab_data,
            surgery_cpt_codes=surgery_cpt_codes,
            radiology_types=radiology_types,
            radiology_titles=radiology_titles
        )

        if patient_df.empty:
            print(f"Worker batch {batch_id}: No patients after building patient_df")
            return pd.DataFrame(), pd.DataFrame(), 0, batch_id

        # Add & redact progress notes
        patient_df_with_progress = add_last_progress_note(patient_df)
        processed_df, skipped_ids = recursive_censor_notes(patient_df_with_progress)
        
        num_cases = len(processed_df)
        print(f"Worker batch {batch_id}: Processed {num_cases} patients, {len(skipped_ids)} skipped")

        # Format for LLM input
        llm_df = create_llm_dataframe(processed_df) if not processed_df.empty else pd.DataFrame()

        # Add has_radiology flag
        if not processed_df.empty:
            processed_df['has_radiology'] = processed_df['radiology_reports'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            )

        # Clean up batch data
        del batch_data
        gc.collect()

        return llm_df, processed_df, num_cases, batch_id

    except Exception as e:
        print(f"Worker batch {batch_id}: Error processing: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), 0, batch_id


def main_batch_processing(surgery_cpt_codes: List[str],
                         radiology_types: List[str],
                         radiology_titles: List[str],
                         clinical_note_types: List[str],
                         clinical_note_titles: List[str],
                         project_id: str,
                         dataset_ids: List[str],
                         data_tables: List[str],
                         use_multiprocessing: bool = True,
                         num_workers: int = 4,
                         max_batches: int = None):
    """Main function that processes data in batches with optional multiprocessing."""
    
    # Initialize processor
    processor = BatchProcessor(project_id, dataset_ids, batch_size=100, num_workers=num_workers)

    # Get total count for progress tracking
    try:
        total_patients = processor.get_total_patient_count()
        print(f"Total patients to process: {total_patients}")
        if use_multiprocessing:
            print(f"Using {num_workers} worker processes")
    except Exception as e:
        print(f"Error getting patient count: {e}")
        return pd.DataFrame(), pd.DataFrame()

    all_llm_data = []
    all_processed_data = []
    global_case_id_counter = 1
    
    if use_multiprocessing:
        # Multiprocessing version
        batch_args_list = []
        batch_num = 0
        
        # Collect batch arguments (just patient IDs and metadata, not the actual data)
        for patient_batch in processor.get_patient_batches():
            batch_num += 1
            if max_batches and batch_num > max_batches:
                break
                
            batch_args = (
                patient_batch, batch_num, project_id, dataset_ids, data_tables,
                surgery_cpt_codes, radiology_types, radiology_titles,
                clinical_note_types, clinical_note_titles
            )
            batch_args_list.append(batch_args)
        
        print(f"Processing {len(batch_args_list)} batches with {num_workers} workers")
        
        # Process batches using multiprocessing
        with Pool(processes=num_workers) as pool:
            results = pool.map(worker_extract_and_process_batch, batch_args_list)
        
        # Collect results and assign case IDs
        for llm_df, processed_df, num_cases, batch_id in results:
            if num_cases > 0:
                # Assign sequential case IDs
                case_ids = range(global_case_id_counter, global_case_id_counter + num_cases)
                processed_df['llm_caseID'] = list(case_ids)
                global_case_id_counter += num_cases
                
                all_llm_data.append(llm_df)
                all_processed_data.append(processed_df)
                
            print(f"Batch {batch_id}: {num_cases} cases added to final results")
    
    else:
        # Original single-threaded version
        batch_num = 0
        try:
            for patient_batch in processor.get_patient_batches():
                batch_num += 1
                if max_batches and batch_num > max_batches:
                    break
                    
                print(f"\n{'='*60}")
                print(f"BATCH {batch_num}")
                print(f"{'='*60}")

                # Extract batch data
                batch_data = processor.extract_batch_data(patient_batch, data_tables)

                # Process the batch using original function
                llm_df, processed_df, global_case_id_counter = process_batch(
                    batch_data=batch_data,
                    patient_ids=patient_batch,
                    global_case_id_counter=global_case_id_counter,
                    surgery_cpt_codes=surgery_cpt_codes,
                    radiology_types=radiology_types,
                    radiology_titles=radiology_titles,
                    clinical_note_types=clinical_note_types,
                    clinical_note_titles=clinical_note_titles
                )

                # Collect results
                if not llm_df.empty:
                    all_llm_data.append(llm_df)
                if not processed_df.empty:
                    all_processed_data.append(processed_df)

                # Clean up memory
                del batch_data
                gc.collect()

                print(f"Batch {batch_num} completed. Total cases so far: {global_case_id_counter - 1}")

        except Exception as e:
            print(f"Error in single-threaded batch processing: {e}")
            import traceback
            traceback.print_exc()

    # Combine all results
    if all_llm_data:
        final_llm_df = pd.concat(all_llm_data, ignore_index=True)
        final_processed_df = pd.concat(all_processed_data, ignore_index=True)
        print(f"\nFinal results: {len(final_llm_df)} cases for LLM processing")
        return final_llm_df, final_processed_df
    else:
        print("No data processed successfully")
        return pd.DataFrame(), pd.DataFrame()