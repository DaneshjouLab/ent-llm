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

    def get_patient_batches(self) -> Iterator[List[str]]:
        """Generator that yields batches of patient IDs, matching extract_sample logic."""
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
        """Extract data for a batch of patients, exactly matching extract_sample logic."""
        batch_data = {}

        # Format patient IDs for SQL IN clause (same as extract_sample)
        id_list_str = ", ".join(f"'{pid}'" for pid in patient_ids)

        print(f"Extracting data for {len(patient_ids)} patients...")

        # Extract patient data from each table for sampled patients (same as extract_sample)
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

# Extract ENT and Radiology reports in batches
def process_batch(batch_data: Dict[str, pd.DataFrame],
                 patient_ids: List[str],
                 global_case_id_counter: int) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Process a single batch of patient data following the exact original pipeline."""
    try:
        print(f"\n=== Processing Batch of {len(patient_ids)} patients ===")

        # Extract ENT notes for this batch (exact same call as your original)
        if 'clinical_note' in batch_data and not batch_data['clinical_note'].empty:
            print("Extracting ENT notes...")
            ent_notes = extract_ent_notes(
                batch_data["clinical_note"],
                CLINICAL_NOTE_TYPES,
                CLINICAL_NOTE_TITLES
            )
            print(f"  Found {len(ent_notes)} ENT notes")
        else:
            print("No clinical notes data for this batch")
            ent_notes = pd.DataFrame()

        # Extract radiology reports for this batch
        if 'radiology_report' in batch_data and not batch_data['radiology_report'].empty:
            print("Extracting radiology reports...")
            rad_reports = extract_radiology_reports(
                batch_data["radiology_report"],
                RADIOLOGY_REPORT_TYPE,
                RADIOLOGY_REPORT_TITLE
            )
            print(f"  Found {len(rad_reports)} radiology reports")
        else:
            print("No radiology data for this batch")
            rad_reports = pd.DataFrame()

        # Process procedures
        if 'procedures' in batch_data and not batch_data['procedures'].empty:
            print("Processing procedures...")
            procedures = procedures_df(
                batch_data['procedures'],
                SURGERY_CPT_CODES,
                DIAGNOSTIC_ENDOSCOPY_CPT_CODES
            )
            print(f"  Found {len(procedures)} relevant procedures")
        else:
            print("No procedures data for this batch")
            procedures = pd.DataFrame()

        # Check if we have any data to process
        if ent_notes.empty and rad_reports.empty and procedures.empty:
            print("No relevant data found in this batch")
            return pd.DataFrame(), pd.DataFrame(), global_case_id_counter

        # Build patient dataframe for this batch
        print("Building patient dataframe...")
        patient_df = build_patient_df(ent_notes, rad_reports, procedures)

        if patient_df.empty:
            print("No patients to process after building patient_df")
            return pd.DataFrame(), pd.DataFrame(), global_case_id_counter

        print(f"Patient dataframe created: {len(patient_df)} patients")

        # Add progress notes
        print("Adding progress notes...")
        patient_df_with_progress = add_last_progress_note(patient_df)
        print(f"After adding progress notes: {len(patient_df_with_progress)} patients")

        # Censor notes and get skipped IDs
        print("Censoring notes...")
        processed_df, skipped_ids = recursive_censor_notes(patient_df_with_progress)
        print(f"After censoring: {len(processed_df)} patients, {len(skipped_ids)} skipped")

        # Create sequential case IDs continuing from global counter
        if not processed_df.empty:
            case_ids = range(global_case_id_counter,
                           global_case_id_counter + len(processed_df))
            processed_df['llm_caseID'] = list(case_ids)
            new_counter = global_case_id_counter + len(processed_df)
        else:
            new_counter = global_case_id_counter

        # Format for LLM input
        print("Creating LLM dataframe...")
        llm_df = create_llm_dataframe(processed_df) if not processed_df.empty else pd.DataFrame()
        print(f"LLM dataframe created: {len(llm_df)} records")

        print(f"Batch processed: {len(processed_df)} cases ready for LLM, {len(skipped_ids)} skipped")

        # Format Processed_DF
        processed_df['has_radiology'] = [arr.size > 0 for arr in processed_df['radiology_reports']]

        return llm_df, processed_df, new_counter

    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), global_case_id_counter


def main_batch_processing():
    """Main function that processes data in batches"""

    # Initialize processor
    processor = BatchProcessor(PROJECT_ID, DATASET_IDS, batch_size=100)

    # Get total count for progress tracking
    try:
        total_patients = processor.get_total_patient_count()
        print(f"Total patients to process: {total_patients}")
    except Exception as e:
        print(f"Error getting patient count: {e}")
        return pd.DataFrame(), pd.DataFrame()

    all_llm_data = []
    all_processed_data = []
    global_case_id_counter = 1
    batch_num = 0

    try:
        # This loop extracts patients in multiple batches
        for patient_batch in processor.get_patient_batches():
            batch_num += 1
            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}")
            print(f"{'='*60}")

            # Extract batch
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

            print(f"Batch {batch_num} completed. Total cases so far: {global_case_id_counter - 1}")

    except Exception as e:
        print(f"Error in main batch processing: {e}")
        import traceback
        traceback.print_exc()

    # Combine all results
    if all_llm_data:
        final_llm_df = pd.concat(all_llm_data, ignore_index=True)
        final_processed_df = pd.concat(all_processed_data, ignore_index=True)
        print(f"\n Final results: {len(final_llm_df)} cases for LLM processing")
        return final_llm_df, final_processed_df
    else:
        print("No data processed successfully")
        return pd.DataFrame(), pd.DataFrame()