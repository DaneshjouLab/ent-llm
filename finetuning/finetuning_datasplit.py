# Load STARR data from GCS
llm_df = pd.read_parquet('gs://starr-sinusitis_2016_2025/llm_df_102725.parquet')
processed_df = pd.read_parquet('gs://starr-sinusitis_2016_2025/processed_df_102725.parquet')

# Filter out long cases
def estimate_tokens(text: str) -> int:
    return len(str(text)) // 3

def find_long_cases(llm_df: pd.DataFrame, max_tokens: int = 5000) -> pd.DataFrame:
    """Find cases that would exceed token limits."""

    long_cases = []

    for idx, row in llm_df.iterrows():
        case_id = row['llm_caseID']
        progress_text = str(row['formatted_progress_text'])
        radiology_text = str(row['formatted_radiology_text'])

        # Estimate total tokens (including prompt overhead)
        prompt_overhead = 400
        total_tokens = (
            estimate_tokens(progress_text) +
            estimate_tokens(radiology_text) +
            prompt_overhead
        )

        if total_tokens > max_tokens:
            long_cases.append({
                'llm_caseID': case_id,
                'progress_tokens': estimate_tokens(progress_text),
                'radiology_tokens': estimate_tokens(radiology_text),
                'total_tokens': total_tokens,
                'progress_chars': len(progress_text),
                'radiology_chars': len(radiology_text)
            })

    long_df = pd.DataFrame(long_cases)
    return long_df

def filter_processable_cases(llm_df: pd.DataFrame, max_tokens: int = 5000) -> tuple:
    """Split dataframe into processable and long cases."""

    print(f"Checking {len(llm_df)} cases for token length (max: {max_tokens})...")

    processable_cases = []
    long_case_ids = []

    for idx, row in llm_df.iterrows():
        progress_text = str(row['formatted_progress_text'])
        radiology_text = str(row['formatted_radiology_text'])

        # Conservative token estimate
        total_tokens = (
            estimate_tokens(progress_text) +
            estimate_tokens(radiology_text) +
            400  # prompt overhead + safety margin
        )

        if total_tokens <= max_tokens:
            processable_cases.append(idx)
        else:
            long_case_ids.append(row['llm_caseID'])

    processable_df = llm_df.iloc[processable_cases].copy()

    print(f"Results:")
    print(f"  Processable cases: {len(processable_df)} ({len(processable_df)/len(llm_df)*100:.1f}%)")
    print(f"  Too long cases: {len(long_case_ids)} ({len(long_case_ids)/len(llm_df)*100:.1f}%)")
    if long_case_ids:
        print(f"  Long case IDs: {sorted(long_case_ids)[:10]}{'...' if len(long_case_ids) > 10 else ''}")

    return processable_df, long_case_ids

# Filter cases
long_cases_info = find_long_cases(llm_df, max_tokens=3000)
print(f"Found {len(long_cases_info)} cases that are too long")
if not long_cases_info.empty:
    print("Sample long cases:")
    print(long_cases_info.head())

llm_df_filtered, long_case_ids = filter_processable_cases(llm_df, max_tokens=5000)

# Preprocessing training data
from sklearn.model_selection import train_test_split

# Subset processed_df to only patients in llm_df_filtered
# Get the case IDs that passed filtering
filtered_case_ids = llm_df_filtered['llm_caseID'].unique()

# Subset processed_df to only those cases
processed_df_filtered = processed_df[processed_df['llm_caseID'].isin(filtered_case_ids)].copy()

# Create llm_df_training by merging had_surgery into llm_df_filtered
llm_df_training = llm_df_filtered.merge(
    processed_df_filtered[['llm_caseID', 'had_surgery']],
    on='llm_caseID',
    how='left'
)

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    llm_df_training,
    train_size=0.8,
    stratify=llm_df_training['had_surgery'],
    random_state=42
)

# Second split: 50/50 of temp = 10% val, 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['had_surgery'],
    random_state=42
)

# Print split info
print(f"\nTRAIN: {len(train_df)} cases ({train_df['had_surgery'].mean()*100:.1f}% surgery)")
print(f"VAL:   {len(val_df)} cases ({val_df['had_surgery'].mean()*100:.1f}% surgery)")
print(f"TEST:  {len(test_df)} cases ({test_df['had_surgery'].mean()*100:.1f}% surgery)")

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)