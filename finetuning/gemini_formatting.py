# Training Formatting for Gemini
import pandas as pd
import json
from google.cloud import storage
from datetime import datetime
from typing import List, Tuple

def format_demographics_from_row(row: pd.Series, exclude_vars: List[str] = None) -> str:
    """Format demographics from dataframe row."""
    if exclude_vars is None:
        exclude_vars = []

    DEMOGRAPHIC_VARS = [
        'legal_sex', 'age', 'race', 'ethnicity', 'recent_bmi',
        'smoking_hx', 'alcohol_use', 'zipcode', 'insurance_type', 'occupation'
    ]

    var_labels = {
        'legal_sex': 'Sex',
        'age': 'Age',
        'race': 'Race',
        'ethnicity': 'Ethnicity',
        'recent_bmi': 'BMI',
        'smoking_hx': 'Smoking History',
        'alcohol_use': 'Alcohol Use',
        'zipcode': 'Zipcode',
        'insurance_type': 'Insurance',
        'occupation': 'Occupation'
    }

    demographics = []
    for var in DEMOGRAPHIC_VARS:
        if var in exclude_vars:
            continue
        value = row.get(var)
        if pd.notna(value):
            label = var_labels.get(var, var)
            demographics.append(f"{label}: {value}")

    return "\n".join(demographics) if demographics else "No information available."


def create_user_message(row: pd.Series, demographic_exclusions: List[str] = None) -> str:
    """Create the user message with case details."""
    demographics = format_demographics_from_row(row, exclude_vars=demographic_exclusions)
    
    progress_text = row['formatted_progress_text']
    radiology_text = row['formatted_radiology_text']
    
    has_radiology = radiology_text and radiology_text.strip() and radiology_text != "No radiology reports available."
    radiology_section = f"- Radiology Report: {radiology_text}" if has_radiology else "- Radiology Report: Not available."
    
    message = f"""=== CASE DETAILS ===
- Case ID: {row['llm_caseID']}

=== PATIENT DEMOGRAPHICS ===
{demographics}

=== CLINICAL INFORMATION ===
- Clinical Summary: {progress_text}
- Radiology report: {radiology_section}

Please evaluate this ENT case and decide whether surgery is recommended."""
    
    return message


def prepare_jsonl_for_gemini(
    df: pd.DataFrame,
    output_file: str,
    demographic_exclusions: List[str] = None
) -> str:
    """
    Prepare JSONL file for Gemini fine-tuning using systemInstruction + contents format.
    
    Args:
        df: DataFrame with columns: llm_caseID, formatted_progress_text,
            formatted_radiology_text, had_surgery, and demographic columns
        output_file: Path to save JSONL file (e.g., "train_data.jsonl")
        demographic_exclusions: List of demographic variables to exclude
    
    Returns:
        Path to created JSONL file
    """
    
    # System instruction that applies to all examples
    system_instruction = {
        "role": "system",
        "parts": [
            {
                "text": """You are an expert otolaryngologist evaluating ENT cases.
Your task is to decide whether surgery is recommended based on the provided case information.

INSTRUCTIONS:
1. Rely strictly on the case details provided (do not invent information).
2. Respond with a single valid JSON object — no extra text, headings, or explanations outside the JSON.
3. Follow the schema exactly.

OUTPUT SCHEMA:
Respond only using this JSON structure:
{
  "DECISION": "Yes" | "No"  // Whether surgery is recommended
}"""
            }
        ]
    }
    
    print(f"Creating JSONL file: {output_file}")
    print(f"Number of examples: {len(df)}")
    print(f"Surgery rate: {df['had_surgery'].mean()*100:.1f}%")
    
    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            # Create user message with case details
            user_message = create_user_message(row, demographic_exclusions)
            
            # Create model response
            decision = "Yes" if row['had_surgery'] else "No"
            model_response = json.dumps({"DECISION": decision})
            
            # Gemini format
            example = {
                "systemInstruction": system_instruction,
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": user_message
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": model_response
                            }
                        ]
                    }
                ]
            }
            
            f.write(json.dumps(example) + '\n')
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx+1}/{len(df)} examples...")
    
    print(f"✓ JSONL file created: {output_file}")
    return output_file


def upload_to_gcs(
    local_file: str,
    bucket_name: str,
    gcs_path: str,
    project_id: str
) -> str:
    """Upload JSONL file to existing GCS bucket."""
    
    print(f"\nUploading to GCS...")
    print(f"  Local file: {local_file}")
    print(f"  Bucket: {bucket_name}")
    print(f"  Path: {gcs_path}")
    
    storage_client = storage.Client(project=project_id)
    
    # Upload directly
    blob = storage_client.bucket(bucket_name).blob(gcs_path)
    blob.upload_from_filename(local_file)
    
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"Upload complete: {gcs_uri}")
    
    return gcs_uri


def prepare_training_data_for_gemini(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    project_id: str,
    bucket_name: str = None,
    demographic_exclusions: List[str] = None
) -> Tuple[str, str]:
    """
    Complete pipeline: Create JSONL files and upload to GCS for Gemini fine-tuning.
    Returns GCS URIs you can use with Gemini API.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        project_id: GCP project ID
        bucket_name: GCS bucket name (default: {project_id}-gemini-tuning)
        demographic_exclusions: Demographics to exclude
    
    Returns:
        (train_gcs_uri, val_gcs_uri) - Use these for Gemini fine-tuning
    """
    
    if bucket_name is None:
        bucket_name = f"{project_id}-gemini-tuning"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("PREPARING DATA FOR GEMINI FINE-TUNING")
    print("="*80)
    print("Format: systemInstruction + contents (user/model turns)")
    print(f"Using Gemini's supervised fine-tuning format")
    
    # Create local JSONL files
    print("\n[1/4] Creating training JSONL...")
    train_file = f"train_{timestamp}.jsonl"
    prepare_jsonl_for_gemini(train_df, train_file, demographic_exclusions)
    
    print("\n[2/4] Creating validation JSONL...")
    val_file = f"val_{timestamp}.jsonl"
    prepare_jsonl_for_gemini(val_df, val_file, demographic_exclusions)
    
    # Upload to GCS
    print("\n[3/4] Uploading training data to GCS...")
    train_gcs_uri = upload_to_gcs(
        local_file=train_file,
        bucket_name=bucket_name,
        gcs_path=f"training_data/{train_file}",
        project_id=project_id
    )
    
    print("\n[4/4] Uploading validation data to GCS...")
    val_gcs_uri = upload_to_gcs(
        local_file=val_file,
        bucket_name=bucket_name,
        gcs_path=f"training_data/{val_file}",
        project_id=project_id
    )
    
    # Print instructions
    print("\n" + "="*80)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*80)
    print("\nYour data format:")
    print('  {"systemInstruction": {...}, "contents": [{"role": "user", ...}, {"role": "model", ...}]}')
    print(f"\n   Training dataset: {train_gcs_uri}")
    print(f"   Validation dataset: {val_gcs_uri}")
    
    return train_gcs_uri, val_gcs_uri

# Example usage:
if __name__ == "__main__":
    # Prepare data 
    train_uri, val_uri = prepare_training_data_for_gemini(
        train_df=train_df,
        val_df=val_df,
        project_id=PROJECT_ID,
        bucket_name = "starr-sinusitis_2016_2025",
        demographic_exclusions=None  # Include all demographics
    )

    # Save URIs for reference
    with open('gcs_uris.txt', 'w') as f:
        f.write(f"Training: {train_uri}\n")
        f.write(f"Validation: {val_uri}\n")

    print("\n✓ URIs saved to: gcs_uris.txt")