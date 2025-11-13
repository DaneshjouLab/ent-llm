# Filtering out long cases
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

# Verify the filtering worked
print(f"\nVerification - checking max tokens in filtered data:")
max_tokens_in_filtered = 0
for idx, row in llm_df_filtered.iterrows():
    progress_text = str(row['formatted_progress_text'])
    radiology_text = str(row['formatted_radiology_text'])
    total_tokens = estimate_tokens(progress_text) + estimate_tokens(radiology_text) + 400
    max_tokens_in_filtered = max(max_tokens_in_filtered, total_tokens)

print(f"Max estimated tokens in filtered data: {max_tokens_in_filtered}")
print(f"Should be <= 5000: {max_tokens_in_filtered <= 5000}")