# Helper functions for processing patient data

# Data Extraction
def extract_ent_notes_dict(clinical_notes_df, note_types, note_titles):
    """
    Extract relevant ENT notes.
    """
    
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


def extract_radiology_report(reports_list):
    """Extract sinus CT radiology report text."""
    if not isinstance(reports_list, list):
        return ""

    relevant_reports = [
        report.get('text', '')
        for report in reports_list
        if 'ct' in str(report.get('type', '')).lower()
        and any(kw in str(report.get('title', '')).lower() for kw in ['sinus', 'paranasal', 'nasal'])
    ]
    return "\n---\n".join(filter(None, relevant_reports))

# Preprocessing 

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
def process_procedures(procedures_list):
    """Returns two flags: if patient had surgery or endoscopy (CPT based)."""
    if not isinstance(procedures_list, list):
        return False, False

    had_surgery = any(
        proc.get('code') in SURGERY_CPT_CODES
        for proc in procedures_list if proc.get('code_type') == 'CPT'
    )
    had_endoscopy = any(
        proc.get('code') in DIAGNOSTIC_ENDOSCOPY_CPT_CODES
        for proc in procedures_list if proc.get('code_type') == 'CPT'
    )
    return had_surgery, had_endoscopy

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