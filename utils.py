# Helper functions for processing patient data

# Censor surgical plans from free text clinical notes
def censor_surgical_plans(text):
    """Censors sentences containing surgical plans or recommendations."""
    import re
    if not isinstance(text, str): return text  # Ensure input is a string
    text = text.lower()  # Normalize to lowercase for keyword matching
    # Define keywords that indicate surgical plans or recommendations
    censor_keywords = ['surgery', 'fess', 'ess', 'operative', 'operation', 'surgical', 'recommend proceeding with', 'plan for', 'schedule', 'consent for']
    # Split text into sentences and filter out those containing any of the keywords
    # This regex splits on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    censored_sentences = [s for s in sentences if not any(kw in s.lower() for kw in censor_keywords)]
    return " ".join(censored_sentences)



# Parse nested ENT data
def extract_ent_notes(notes_list):
    """Extracts text from relevant ENT outpatient progress notes."""
    if not isinstance(notes_list, list): return ""
    ent_notes_text = []
    for note in notes_list:
        note_type = note.get('type', '').lower()
        author = note.get('author', '').lower()
        text = note.get('text', '').lower()
        if "progress note, outpatient" in note_type and ('ent' in author or 'otolaryngology' in author or 'ent' in text or 'otolaryngology' in text):
            ent_notes_text.append(note.get('text', ''))
    return "\n---\n".join(ent_notes_text)


# Check if any keyword exists in a list of dictionaries
def check_list_for_keywords(items, key_name, keywords):
    """Generic function to check if any keyword exists in a list of dictionaries."""
    if not isinstance(items, list): return False
    return any(keyword in str(item.get(key_name, '')).lower() for item in items for keyword in keywords)

def process_procedures(procedures_list):
    """Checks for specific diagnostic and surgical CPT codes."""
    if not isinstance(procedures_list, list): return False, False
    had_surgery = False
    had_endoscopy = False
    for proc in procedures_list:
        if proc.get('code_type') == 'CPT':
            code = proc.get('code')
            if code in SURGERY_CPT_CODES:
                had_surgery = True
            if code in DIAGNOSTIC_ENDOSCOPY_CPT_CODES:
                had_endoscopy = True
    return had_surgery, had_endoscopy


# Extract radiology report if present
def extract_radiology_report(reports_list):
    """Extracts text from relevant sinus CT reports."""
    if not isinstance(reports_list, list): return ""
    report_texts = []
    for report in reports_list:
        report_type = str(report.get('type', '')).lower()
        title = str(report.get('title', '')).lower()
        # Ensure we only include relevant CT reports of the sinuses
        if 'ct' in report_type and ('sinus' in title or 'paranasal' in title or 'nasal' in title):
            report_texts.append(report.get('text', ''))
    return "\n---\n".join(filter(None, report_texts))

# LLM setup 
def query_openai(prompt):
    """Sends a prompt to the GPT-4 model and returns the content of the response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert otolaryngologist. Your task is to provide a surgical recommendation based on the provided patient data. Respond only in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return None


# LLM Prompt

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
    
    ---
    
    Provide your response as a JSON object with three keys:
    1. "decision": Your recommendation, either "Yes" or "No".
    2. "confidence": Your confidence level from 1 (not confident) to 10 (very confident).
    3. "reasoning": A brief, 2-4 sentence explanation for your decision.
    """
    return prompt


# Create a dataframe of all patients with their longitudinal summaries and demographics
def preprocess_all_patients(dataframes, surgery_codes):
    processed_records = []
    # Get a unique list of all patients from the demographics table
    unique_patient_ids = dataframes['demographics']['patient_id'].unique()

    for patient_id in tqdm(unique_patient_ids, desc="Processing Patients"):
        # --- Fetch all records for this specific patient from each table ---
        patient_notes = dataframes['clinical_note'][dataframes['clinical_note']['patient_id'] == patient_id]
        patient_procedures = dataframes['procedures'][dataframes['procedures']['patient_id'] == patient_id]
        patient_labs = dataframes['labs'][dataframes['labs']['patient_id'] == patient_id]
        patient_meds = dataframes['med_orders'][dataframes['med_orders']['patient_id'] == patient_id]
        patient_radiology = dataframes['radiology_report'][dataframes['radiology_report']['patient_id'] == patient_id]
        patient_demographics = dataframes['demographics'][dataframes['demographics']['patient_id'] == patient_id].iloc[-1] # Most recent record

        # --- Perform the longitudinal logic ---
        # Sort notes by date to create the history
        all_notes_sorted = patient_notes.sort_values(by='date', ascending=True)

        # Find the earliest surgery date for this patient
        earliest_surgery_date = pd.NaT
        surgeries = patient_procedures[patient_procedures['code'].isin(SURGERY_CPT_CODES)]
        if not surgeries.empty:
            earliest_surgery_date = pd.to_datetime(surgeries['date']).min()
            
        # Filter notes to only those occurring BEFORE the surgery
        if pd.notna(earliest_surgery_date):
            notes_for_llm_df = all_notes_sorted[pd.to_datetime(all_notes_sorted['date']) < earliest_surgery_date]
        else:
            notes_for_llm_df = all_notes_sorted

        # Filter for relevant ENT progress notes
        ent_notes_mask = notes_for_llm_df['type'].str.contains("progress note, outpatient", case=False, na=False) & \
                        (notes_for_llm_df['author'].str.contains("ent|otolaryngology", case=False, na=False) | \
                        notes_for_llm_df['text'].str.contains("ent|otolaryngology", case=False, na=False))
        
        final_ent_notes = notes_for_llm_df[ent_notes_mask]

        if final_ent_notes.empty:
            continue # Skip patient if they have no relevant notes before the decision point

        # Combine and censor notes
        longitudinal_summary = "\n---\n".join(final_ent_notes['text'].dropna())
        censored_summary = censor_surgical_plans(longitudinal_summary)
        
        # Properly filter for relevant CT Sinus reports
        radiology_text = extract_radiology_report(patient_radiology.to_dict('records'))

        # Combine data for individual patient
        processed_patient_records.append({
            'patient_id': patient_id,
            'age': patient_demographics.get('age'), 'race': patient_demographics.get('race'),
            'ethnicity': patient_demographics.get('ethnicity'), 'sex': patient_demographics.get('legal_sex'),
            'longitudinal_summary_statement': longitudinal_summary,
            'censored_summary_statement': censored_summary,
            'radiology_text': radiology_text,
            'has_radiology_report': bool(radiology_text),
            'Had_Surgery': pd.notna(earliest_surgery_date),
        })
    return pd.DataFrame(processed_records)



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
