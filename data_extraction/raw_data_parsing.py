# Helper functions for processing patient data

# ENT Note and Procedure Extraction
import pandas as pd

def extract_ent_notes(clinical_notes_df, note_types, note_titles):
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

def extract_radiology_reports(radiology_df, ent_patient_ids, types, titles):
    """Extract relevant radiology reports."""
    
    df = radiology_df[radiology_df['patient_id'].isin(ent_patient_ids)].copy()

    # Normalize string fields
    df['type'] = df['type'].astype(str)
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str).str.lower()

    # Filter by type or title match
    type_filter = df['type'].isin(types)
    title_filter = df['title'].isin(titles)

    filtered_df = df[type_filter & title_filter].copy()
    
    return filtered_df

def extract_procedures_df(ent_procedures_df, ent_patient_ids, surgery_cpt_codes):
    """Returns a dataframe with surgery/endoscopy flags and their earliest CPT dates."""
    import pandas as pd

    # Filter to ENT patients 
    procedures_df = ent_procedures_df[ent_procedures_df['patient_id'].isin(ent_patient_ids)].copy()
    
    # Ensure proper data types
    procedures_df['code'] = procedures_df['code'].astype(str)
    procedures_df['code_type'] = procedures_df['code_type'].astype(str)

    # Results dictionary to collect per-patient results
    results = {}

    for patient_id in procedures_df['patient_id'].unique():
        patient_procs = procedures_df[procedures_df['patient_id'] == patient_id]
        patient_cpt = patient_procs[patient_procs['code_type'].str.upper() == 'CPT']

        had_surgery = False
        surgery_dates = []

        for _, row in patient_cpt.iterrows():
            code = row['code']
            date = row.get('date')
            parsed_date = pd.to_datetime(date) if pd.notna(date) else None

            if code in surgery_cpt_codes:
                had_surgery = True
                if parsed_date:
                    surgery_dates.append(parsed_date)

        first_surgery_date = min(surgery_dates) if surgery_dates else pd.NaT

        results[patient_id] = {
            'had_surgery': had_surgery,
            'first_surgery_date': first_surgery_date,
        }

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.rename(columns={'index': 'patient_id'}, inplace=True)

    return results_df

def extract_demographic_data(demographics_df, ent_patient_ids):
    """Extract demographic data for specific patients only."""

    # Filter to patient IDs 
    ent_demographics_df = demographics_df[demographics_df['patient_id'].isin(ent_patient_ids)].copy()
    
    # Filter to relevant demographic columns
    demographic_columns = [
        'patient_id', 'legal_sex', 'race', 'ethnicity', 'date_of_birth', 
        'recent_bmi', 'smoking_hx', 'alcohol_use', 'zipcode', 'insurance_type', 'occupation'
    ]
    
    demo_df = ent_demographics_df[demographic_columns].copy()
    
    # Normalize data types
    if 'date_of_birth' in demo_df.columns:
        demo_df['date_of_birth'] = pd.to_datetime(demo_df['date_of_birth'], errors='coerce')
    
    # Remove duplicates by keeping most recent record
    demo_df = demo_df.drop_duplicates(subset=['patient_id'], keep='last')
    
    
    return demo_df

def extract_lab_data(lab_df, ent_patient_ids, surgery_dates_dict=None):
    """Extract and process lab data for specific patients only, filtered by surgery date if available."""
    import pandas as pd
    
    # Filter to patient IDs
    lab_df = lab_df[lab_df['patient_id'].isin(ent_patient_ids)].copy()

    if lab_df.empty:
        return pd.DataFrame(columns=['patient_id', 'lab_results'])
    
    # Ensure proper data types
    lab_df['patient_id'] = lab_df['patient_id'].astype(str)
    
    # Convert date columns to datetime
    date_columns = ['order_date', 'taken_date', 'result_date']
    for col in date_columns:
        if col in lab_df.columns:
            lab_df[col] = pd.to_datetime(lab_df[col], errors='coerce')
    
    # Filter by surgery date if provided
    if surgery_dates_dict:
        def filter_by_surgery_date(row):
            patient_id = row['patient_id']
            surgery_date = surgery_dates_dict.get(patient_id)
            
            if pd.notna(surgery_date):
                # Use result_date if available, otherwise taken_date, otherwise order_date
                lab_date = row['result_date'] if pd.notna(row['result_date']) else (
                    row['taken_date'] if pd.notna(row['taken_date']) else row['order_date']
                )
                if pd.notna(lab_date):
                    return lab_date < surgery_date
            
            # If no surgery date or no lab date, include the lab result
            return True
        
        lab_df = lab_df[lab_df.apply(filter_by_surgery_date, axis=1)]
    
    # Group lab results by patient
    lab_grouped = lab_df.groupby('patient_id').apply(
        lambda x: sorted(
            [
                {
                    'order_date': row['order_date'].strftime('%Y-%m-%d') if pd.notnull(row['order_date']) else None,
                    'taken_date': row['taken_date'].strftime('%Y-%m-%d') if pd.notnull(row['taken_date']) else None,
                    'result_date': row['result_date'].strftime('%Y-%m-%d') if pd.notnull(row['result_date']) else None,
                    'age': row.get('age'),
                    'lab': row['lab'],
                    'result': row['result'],
                    'value': row.get('value')
                }
                for _, row in x.iterrows()
            ],
            key=lambda lab: lab['result_date'] if lab['result_date'] else ''
        ),
        include_groups=False
    ).reset_index(name='lab_results')

    return lab_grouped
    
def build_patient_df(ent_df, radiology_df, procedures_df, demographics_df, lab_df, surgery_cpt_codes, radiology_types, radiology_titles):
    """Builds a patient-level DataFrame with ENT notes, radiology reports, surgery, demographics, and lab data."""
    import pandas as pd

    # Get unique ENT patient IDs (this is our master list)
    ent_patient_ids = set(ent_df['patient_id'].unique())
    print(f"Found {len(ent_patient_ids)} unique ENT patients")

    # Normalize and clean dates
    ent_df['date'] = pd.to_datetime(ent_df['date'], errors='coerce')
    ent_df['text'] = ent_df['text'].astype(str)
    ent_df['type'] = ent_df['type'].astype(str)
    ent_df['title'] = ent_df['title'].astype(str)

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

    print(f"Check count of ENT patients: {len(ent_grouped)}")

    # Get procedures data for ENT patients only
    surgery_df = extract_procedures_df(procedures_df, ent_patient_ids, surgery_cpt_codes)
    
    # Merge with surgery data first to get surgery dates
    patient_data = pd.merge(ent_grouped, surgery_df, on='patient_id', how='left')
    patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)
    print(f"After surgery merge: {len(patient_data)} patients") 

    # Create surgery dates dictionary for filtering
    surgery_dates_dict = {}
    for _, row in patient_data.iterrows():
        if pd.notna(row['first_surgery_date']):
            surgery_dates_dict[row['patient_id']] = row['first_surgery_date']

    filtered_radiology = extract_radiology_reports(radiology_df, ent_patient_ids, radiology_types, radiology_titles)
    if not filtered_radiology.empty:
        filtered_radiology['date'] = pd.to_datetime(filtered_radiology['date'], errors='coerce')
        filtered_radiology['text'] = filtered_radiology['text'].astype(str)
        filtered_radiology['type'] = filtered_radiology['type'].astype(str)
        filtered_radiology['title'] = filtered_radiology['title'].astype(str)

        # Filter radiology by surgery date
        def filter_radiology_by_surgery(group):
            patient_id = group.name
            patient_match = patient_data[patient_data['patient_id'] == patient_id]
            
            if len(patient_match) > 0:
                surgery_date = patient_match['first_surgery_date'].iloc[0]
                if pd.notna(surgery_date):
                    # Only include reports before surgery date
                    group = group[group['date'] < surgery_date]

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
        
        rad_grouped = filtered_radiology.groupby('patient_id').apply(filter_radiology_by_surgery, include_groups=False).reset_index(name='radiology_reports')
        patient_data = pd.merge(patient_data, rad_grouped, on='patient_id', how='left')
        print(f"After radiology merge: {len(patient_data)} patients")
    
    # Handle missing radiology reports
    patient_data['radiology_reports'] = patient_data['radiology_reports'].apply(lambda x: x if isinstance(x, list) else [])
    
    demo_data = extract_demographic_data(demographics_df, ent_patient_ids)
    patient_data = pd.merge(patient_data, demo_data, on='patient_id', how='left')
    print(f"After demographics merge: {len(patient_data)} patients")

    lab_data = extract_lab_data(lab_df, ent_patient_ids)
    patient_data = pd.merge(patient_data, lab_data, on='patient_id', how='left')
    patient_data['lab_results'] = patient_data['lab_results'].apply(lambda x: x if isinstance(x, list) else [])
    print(f"After lab merge: {len(patient_data)} patients")

    print(f"Final dataset contains {len(patient_data)} ENT patients")
    return patient_data