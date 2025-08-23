import argparse
import pandas as pd
from dotenv import load_dotenv
from src.data.encryption import (
    generate_and_secure_key_in_local_env_file,
    write_pandas_to_encrypted_file,
)

RAW_DATA_PATH_WITH_LABELS = "data/data_2025/raw/Exploitable anonymised letters with labels.xlsx"
RAW_DATA_PATH_WITH_LETTERS = "data/data_2025/raw/Exploitable anonymised letters.xlsx"
ENCRYPTED_DATASET_PATH = "data/data_2025/processed/dataset.encrypted.csv"
COLUMNS_TO_KEEP = [
    "patient_id",
    "creation_date",
    "Age_creation_date",
    "Anonymised letter",
    "Label_student",
]


def main(args):
    """
    Main function to create and encrypt the dataset
    """
    # Create the dataset
    print("Creating dataset from raw data...")
    df = create_dataset(
        raw_data_path_with_labels=RAW_DATA_PATH_WITH_LABELS,
        raw_data_path_with_letters=RAW_DATA_PATH_WITH_LETTERS,
        columns_to_keep=COLUMNS_TO_KEEP,
    )
    
    # Save a key to an local environment file, once
    print("Generating encryption key...")
    generate_and_secure_key_in_local_env_file(
        key_var_name=args.encryption_key_var_name,
        env_file_path=args.local_env_file_path,
    )

    # Set the encryption key as an environment variable
    success = load_dotenv(args.local_env_file_path)
    if not success:
        print(
            f"Warning: Could not load '{args.local_env_file_path}'. "
            "Ensure the file exists and is formatted correctly."
        )

    # Save the dataset to an encrypted CSV file
    print("Saving dataset to encrypted CSV file...")
    write_pandas_to_encrypted_file(
        data_to_encrypt=df,
        encrypted_file_path=ENCRYPTED_DATASET_PATH,
        encryption_key_var_name=args.encryption_key_var_name,
    )


def create_dataset(
    raw_data_path_with_labels: str,
    raw_data_path_with_letters: str,
    columns_to_keep: list,
    output_dataset_path: str|None = None,
) -> pd.DataFrame:
    """
    Creates a dataset from an Excel file and saves it as a CSV file.

    Args:
        raw_data_path (str): The path to the raw Excel data file.
        csv_data_path (str): The path where the processed CSV data will be saved.
    """
    # Add student labels to the letters dataset
    df = pd.read_excel(raw_data_path_with_letters)
    df_labels = pd.read_excel(raw_data_path_with_labels)
    df["Label_student"] = df_labels["Label_student"]

    # Keep track of document, remove true patient id, and keep only used columns
    df["document_id"] = df.index
    df["patient_id"] = df.groupby("patient_id").ngroup()
    df = df[columns_to_keep]

    # Save the dataset to a CSV file if output path is provided
    if output_dataset_path is not None:
        df.to_csv(output_dataset_path, index=False)
        print(f"Dataset saved (not encrypted) to {output_dataset_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and encrypt dataset.")
    parser.add_argument(
        "--encryption_key_var_name",
        "-e",
        type=str,
        required=True,
        help="Name of the environment variable for encryption key retrieval.",
    )
    parser.add_argument(
        "--local_env_file_path",
        "-l",
        type=str,
        required=True,
        help="Path to the local environment file storing the encryption key.",
    )
    args = parser.parse_args()
    main(args)
