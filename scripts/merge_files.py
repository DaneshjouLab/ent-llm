import pandas as pd


def merge_csv_files(file_list, output_file="merged_file.csv"):
    """
    Merges multiple CSV files into a single sorted CSV file.

    Parameters:
    file_list (list of str): List of file paths to the CSV files to be merged.
    output_file (str): Path to the output CSV file (default: merged_file.csv).
    """
    # Initialize an empty list to hold dataframes
    dataframes = []

    # Read each CSV file and append the dataframe to the list
    for file in file_list:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded: {file} ({len(df)} rows)")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not dataframes:
        print("No dataframes to merge!")
        return

    # Concatenate all dataframes into a single dataframe
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")

    # Remove duplicates by llm_caseID
    if "llm_caseID" in merged_df.columns:
        duplicate_count = merged_df.duplicated(subset=["llm_caseID"], keep=False).sum()

        if duplicate_count > 0 and "decision" in merged_df.columns:
            # Custom deduplication: keep row with non-NaN decision, if all NaN keep first
            # Create a sort key: prioritize non-NaN decision values
            merged_df["_has_decision"] = merged_df["decision"].notna().astype(int)
            merged_df = merged_df.sort_values(by=["llm_caseID", "_has_decision"], ascending=[True, False])
            merged_df = merged_df.drop_duplicates(subset=["llm_caseID"], keep="first")
            merged_df = merged_df.drop(columns=["_has_decision"])
            print(f"Removed duplicate entries (kept row with non-NaN decision, or first if all NaN)")
        elif duplicate_count > 0:
            # If decision column doesn't exist, keep first occurrence
            merged_df = merged_df.drop_duplicates(subset=["llm_caseID"], keep="first")
            print(f"Removed duplicate entries (decision column not found, kept first occurrence)")

        print(f"Rows after deduplication: {len(merged_df)}")
    else:
        print("Warning: llm_caseID column not found. Skipping deduplication.")

    # Sort by llm_caseID if it exists
    if "llm_caseID" in merged_df.columns:
        merged_df = merged_df.sort_values(by="llm_caseID").reset_index(drop=True)
        print(f"Sorted by llm_caseID")
    else:
        print("Warning: llm_caseID column not found. Saving without sorting.")

    # Report NaN decision values in final dataframe
    if "decision" in merged_df.columns:
        nan_count = merged_df["decision"].isna().sum()
        non_nan_count = merged_df["decision"].notna().sum()
        print(f"Final decision column: {non_nan_count} non-NaN, {nan_count} NaN ({nan_count/len(merged_df)*100:.1f}%)")

        # Report llm_caseID values with NaN decisions
        if nan_count > 0:
            nan_case_ids = merged_df[merged_df["decision"].isna()]["llm_caseID"].tolist()
            print(f"\nllm_caseID values with NaN decision:")
            for case_id in nan_case_ids:
                print(f"  {case_id}")

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to: {output_file}")


def main():
    # Example usage
    files_to_merge = [
        "data/results/file1.csv",
        "data/results/file2.csv",
    ]
    merge_csv_files(files_to_merge, output_file="data/results/merged_results.csv")

if __name__ == "__main__":
    main()