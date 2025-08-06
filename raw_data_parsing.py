# Helper functions for processing patient data

# ENT Note and Procedure Extraction
def extract_ent_notes_ref(clinical_notes_df, note_types, note_titles):
    """Extract relevant ENT notes."""

    df = clinical_notes_df.copy()

    # Normalize string fields
    df['author'] = df['author'].astype(str).str.lower()
    df['text'] = df['text'].astype(str).str.lower()
    df['type'] = df['type'].astype(str)
    df['title'] = df['title'].astype(str)

    # Match full word "ent" or "otolaryngology"
    keyword = r'\b(?:ent|otolaryngology)\b'

    ent_filter = (
        df['author'].str.contains(keyword, na=False, regex=True) |
        df['linked_author'].str.contains(keyword, na=False, regex=True)
    )

    key_filter = df['type'].isin(note_types) | df['title'].isin(note_titles)

    ent_df = df[ent_filter & key_filter].copy()

    return ent_df

def extract_radiology_reports(radiology_df, types, titles):
    """Extract relevant radiology reports."""
    df = radiology_df.copy()

    # Normalize string fields
    df['type'] = df['type'].astype(str)
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str).str.lower()

    # Filter by type or title match
    type_filter = df['type'].isin(types)
    title_filter = df['title'].isin(titles)

    filtered_df = df[type_filter & title_filter].copy()
    
    return filtered_df

def procedures_df(procedures_df, surgery_cpt_codes, endoscopy_cpt_codes):
    """Returns a dataframe with surgery/endoscopy flags and their earliest CPT dates."""
    import pandas as pd

    # Ensure proper data types
    procedures_df['code'] = procedures_df['code'].astype(str)
    procedures_df['code_type'] = procedures_df['code_type'].astype(str)

    # Results dictionary to collect per-patient results
    results = {}

    for patient_id in procedures_df['patient_id'].unique():
        patient_procs = procedures_df[procedures_df['patient_id'] == patient_id]
        patient_cpt = patient_procs[patient_procs['code_type'].str.upper() == 'CPT']

        had_surgery = False
        had_endoscopy = False
        surgery_dates = []
        endoscopy_dates = []

        for _, row in patient_cpt.iterrows():
            code = row['code']
            date = row.get('date')
            parsed_date = pd.to_datetime(date) if pd.notna(date) else None

            if code in surgery_cpt_codes:
                had_surgery = True
                if parsed_date:
                    surgery_dates.append(parsed_date)

            if code in endoscopy_cpt_codes:
                had_endoscopy = True
                if parsed_date:
                    endoscopy_dates.append(parsed_date)

        first_surgery_date = min(surgery_dates) if surgery_dates else pd.NaT
        first_endoscopy_date = min(endoscopy_dates) if endoscopy_dates else pd.NaT

        results[patient_id] = {
            'had_surgery': had_surgery,
            'had_endoscopy': had_endoscopy,
            'first_surgery_date': first_surgery_date,
            'first_endoscopy_date': first_endoscopy_date
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'patient_id'}, inplace=True)

    return results_df


def build_patient_df(ent_df, radiology_df, surgery_df):
    """Builds a patient-level DataFrame with ENT notes, radiology reports, and surgery data."""
    import pandas as pd
    # Normalize and clean dates
    ent_df['date'] = pd.to_datetime(ent_df['date'], errors='coerce')
    radiology_df['date'] = pd.to_datetime(radiology_df['date'], errors='coerce')

    # Ensure columns are strings
    for df in [ent_df, radiology_df]:
        df['text'] = df['text'].astype(str)
        df['type'] = df['type'].astype(str)
        df['title'] = df['title'].astype(str)

    # Group ENT notes by patient
    ent_grouped = ent_df.groupby('patient_id').apply(
        lambda x: sorted(
            [
                {
                    'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
                    'type': typ,
                    'title': ttl,
                    'text': t
                }
                for d, t, typ, ttl in zip(x['date'], x['text'], x['type'], x['title'])
            ],
            key=lambda note: note['date'] if note['date'] else ''
        ),
        include_groups=False
    ).reset_index(name='ent_notes')

    # Merge with surgery data first to get surgery dates
    patient_data = pd.merge(ent_grouped, surgery_df, on='patient_id', how='left')
    patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)

    # Group radiology reports by patient, filtering by surgery date
    def filter_radiology_by_surgery(group):
        patient_id = group.name
        # Get surgery date for this patient
        surgery_date = patient_data[patient_data['patient_id'] == patient_id]['first_surgery_date'].iloc[0]

        # Filter radiology reports
        if pd.notna(surgery_date):
            # Only include reports before surgery date
            group = group[group['date'] < surgery_date]
        # If no surgery date (NaT), keep all reports

        return sorted(
            [
                {
                    'date': d.strftime('%Y-%m-%d') if pd.notnull(d) else None,
                    'type': typ,
                    'title': ttl,
                    'text': t
                }
                for d, t, typ, ttl in zip(group['date'], group['text'], group['type'], group['title'])
            ],
            key=lambda note: note['date'] if note['date'] else ''
        )

    rad_grouped = radiology_df.groupby('patient_id').apply(filter_radiology_by_surgery, include_groups=False).reset_index(name='radiology_reports')

    # Merge radiology data
    patient_data = pd.merge(patient_data, rad_grouped, on='patient_id', how='left')

    # Handle missing values
    patient_data['ent_notes'] = patient_data['ent_notes'].apply(lambda x: x if isinstance(x, list) else [])
    patient_data['radiology_reports'] = patient_data['radiology_reports'].apply(lambda x: x if isinstance(x, list) else [])

    return patient_data