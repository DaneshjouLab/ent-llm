# Helper functions for processing patient data

# ENT Note and Procedure Extraction
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
        df['text'].str.contains(keyword, na=False, regex=True)
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
    """Returns a dataframe with patients that had surgery or endoscopy (CPT based)."""

    # Check code and code_type columns are strings
    procedures_df['code'] = procedures_df['code'].astype(str)
    procedures_df['code_type'] = procedures_df['code_type'].astype(str)

    # Initialize the results dictionary
    results = {}

    for patient_id in procedures_df['patient_id'].unique():
        patient_procs = procedures_df[procedures_df['patient_id'] == patient_id]

        had_surgery = any(
            proc in surgery_cpt_codes
            for proc in patient_procs[patient_procs['code_type'].str.upper() == 'CPT']['code']
        )
        had_endoscopy = any(
            proc in endoscopy_cpt_codes
            for proc in patient_procs[patient_procs['code_type'].str.upper() == 'CPT']['code']
        )

        results[patient_id] = {'had_surgery': had_surgery, 'had_endoscopy': had_endoscopy}

    # Convert the results dictionary into a DataFrame
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
        )
    ).reset_index(name='ent_notes')

    # Group radiology reports by patient
    rad_grouped = radiology_df.groupby('patient_id').apply(
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
        )
    ).reset_index(name='radiology_reports')

    # Merge patient note data
    patient_data = pd.merge(ent_grouped, rad_grouped, on='patient_id', how='outer')
    patient_data = pd.merge(patient_data, surgery_df, on='patient_id', how='left')

    # Handle missing values
    patient_data['ent_notes'] = patient_data['ent_notes'].apply(lambda x: x if isinstance(x, list) else [])
    patient_data['radiology_reports'] = patient_data['radiology_reports'].apply(lambda x: x if isinstance(x, list) else [])
    patient_data['had_surgery'] = patient_data['had_surgery'].fillna(False)

    return patient_data

# Patient Data Processing 

def censor_surgical_plans(text):
    """Remove sentence that suggest surgical plans or recommendations."""
    import re
    if not isinstance(text, str): 
        return text  # Ensure input is a string
    
    text = text.lower()  # Normalize to lowercase for keyword matching
    # Define keywords that indicate surgical plans or recommendations
    censor_keywords = [
        'surgery', 'fess', 'ess', 'operative', 'operation', 'surgical', 
        'recommend proceeding with', 'plan for', 'schedule', 'consent for']
    
    # Split text into sentences and filter out those containing any of the keywords
    # This regex splits on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    censored = [s for s in sentences if not any(kw in s for kw in censor_keywords)]
    return " ".join(censored)

# 

# LLM Prompting

def query_openai(prompt):
    """Query GPT-4 for surgical decision based on input prompt."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are an expert otolaryngologist. "
                    "Provide a surgical recommendation in the requested JSON format."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return None

def generate_prompt(case):
    
    radiology_section = f"- Radiology Report: {case['radiology_text']}" if case['has_radiology_report'] else "- Radiology Report: Not available."

    """Generates a structured prompt for the LLM."""

    prompt = f"""
    You are an expert otolaryngologist evaluating a case of chronic sinusitis.
    Based ONLY on the information provided below, make a recommendation on sinus surgery.

    --- Case Details ---
    - Case Index: {case['case_index']}
    - Age: {case['age']}
    - Sex: {case['sex']}
    - Censored Clinical Summary from ENT Notes: {case['censored_summary_statement']}
    {radiology_section}
    t
    ---
    
    Provide your response as a JSON object with three keys:
    1. "decision": Your recommendation, either "Yes" or "No".
    2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
    3. "reasoning": A brief, 2-4 sentence explanation for your decision.
    """
    return prompt

def process_all_patients(dataframes, surgery_codes):
    """Processes patient data and returns a dataframe of features for LLM input."""
    processed = []

    for patient_id in tqdm(dataframes['demographics']['patient_id'].unique(), desc="Processing Patients"):
        demographics = dataframes['demographics'].loc[
            dataframes['demographics']['patient_id'] == patient_id
        ].iloc[-1]

        notes_df = dataframes['clinical_note'].query("patient_id == @patient_id")
        procedures_df = dataframes['procedures'].query("patient_id == @patient_id")
        radiology_df = dataframes['radiology_report'].query("patient_id == @patient_id")

        notes_df = notes_df.sort_values(by='date')
        surgery_dates = procedures_df.loc[
            procedures_df['code'].isin(SURGERY_CPT_CODES), 'date'
        ]
        earliest_surgery = pd.to_datetime(surgery_dates.min()) if not surgery_dates.empty else pd.NaT

        pre_surgery_notes = notes_df if pd.isna(earliest_surgery) else \
                            notes_df[pd.to_datetime(notes_df['date']) < earliest_surgery]

        ent_notes = extract_ent_notes(pre_surgery_notes.to_dict('records'))
        if ent_notes.empty:
            continue

        summary = "\n---\n".join(ent_notes['text'].dropna())
        censored_summary = censor_surgical_plans(summary)
        radiology_text = extract_radiology_report(radiology_df.to_dict('records'))
        procedures_extracted = process_procedures(procedures_df.to_dict('records'))

        processed.append({
            'patient_id': patient_id,
            'age': demographics.get('age'),
            'race': demographics.get('race'),
            'ethnicity': demographics.get('ethnicity'),
            'sex': demographics.get('legal_sex'),
            'longitudinal_summary_statement': summary,
            'censored_summary_statement': censored_summary,
            'radiology_text': radiology_text,
            'has_radiology_report': bool(radiology_text),
            'Had_Surgery': procedures_extracted.get('had_surgery', False),
            'Had_Endoscopy': procedures_extracted.get('had_endoscopy', False)
        })

    return pd.DataFrame(processed)


# Evaluation
def evaluate_predictions(eval_df):
    
    # Map LLM decision ('Yes'/'No') to boolean for comparison. Using .lower() is robust.
    eval_df['llm_decision_bool'] = eval_df['llm_decision'].str.lower() == 'yes'

    #  Use 'Had_Surgery' (the correct column name) for comparison
    y_true = eval_df['Had_Surgery']
    y_pred = eval_df['llm_decision_bool']
    
     # Accuracy, Classification Report, and Confusion Matrix
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Surgery', 'Surgery'], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # --- Confidence Analysis ---
    eval_df['is_correct'] = (y_true == y_pred)
    
    print("\n--- Confidence Analysis ---")
    print(f"Average confidence on CORRECT predictions: {eval_df[eval_df['is_correct']]['llm_confidence'].mean():.2f}")
    print(f"Average confidence on INCORRECT predictions: {eval_df[~eval_df['is_correct']]['llm_confidence'].mean():.2f}")

    # Break down confidence by prediction outcome (TP, TN, FP, FN)
    tp_mask = (eval_df['is_correct']) & (eval_df['Had_Surgery'])
    tn_mask = (eval_df['is_correct']) & (~eval_df['Had_Surgery'])
    fp_mask = (~eval_df['is_correct']) & (eval_df['llm_decision_bool'])
    fn_mask = (~eval_df['is_correct']) & (~eval_df['llm_decision_bool'])

    print("\n--- Average Confidence by Prediction Outcome ---")
    print(f"True Positives (Correctly recommended surgery):  {eval_df.loc[tp_mask, 'llm_confidence'].mean():.2f}")
    print(f"True Negatives (Correctly recommended no surgery): {eval_df.loc[tn_mask, 'llm_confidence'].mean():.2f}")
    print(f"False Positives (Wrongly recommended surgery):   {eval_df.loc[fp_mask, 'llm_confidence'].mean():.2f}")
    print(f"False Negatives (Missed needed surgery):       {eval_df.loc[fn_mask, 'llm_confidence'].mean():.2f}")