def extract_sample(project_ID, dataset_IDs, table_names, patient_id_col, n_patients):
     """Extract a sample for testing."""
    client = bigquery.Client(project=project_ID)
    sample_data = {}

    # Build a UNION ALL query from all clinical_note tables
    notes_union = "\nUNION ALL\n".join(
        f"SELECT {patient_id_col} FROM `{project_ID}.{ds}.clinical_note`" for ds in dataset_ids
    )

    # Query distinct patient IDs and sample first N sorted
    sample_patients_query = f"""
    WITH all_notes AS (
        -- Get all patient IDs that have at least one clinical note
        SELECT DISTINCT {patient_id_col} FROM ({notes_union})
    )
    SELECT {patient_id_col}
    FROM all_notes
    ORDER BY {patient_id_col}
    LIMIT {n_patients}
    """

    try:
        sample_df = client.query(sample_patients_query).to_dataframe()
        sample_id = sample_df[patient_id_col].tolist()
        print(f"Retrieved {len(sample_ids)} patient IDs.")
    except Exception as e:
        print(f"Error extracting patient IDs: {e}")
        return {}, []

    # Format patient IDs for SQL IN clause
    id_list_str = ", ".join(f"'{pid}'" for pid in sample_id)

    # Extract patient data from each table for sampled patients
    for table in table_names:

        union_query = "\nUNION ALL\n".join(
            f"SELECT * FROM `{project_ID}.{ds}.{table}`" for ds in dataset_ids
        )

        full_query = f"""
        SELECT * FROM ({union_query})
        WHERE {patient_id_col} IN ({id_list_str})
        """

        try:
            df = client.query(full_query).to_dataframe()
            sample_data[table] = df
            print(f"{df.shape[0]} rows loaded.")
        except Exception as e:
            print(f"Failed to load '{table}': {e}")
            sample_data[table] = pd.DataFrame()

    return sample_data, sample_id