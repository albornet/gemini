import argparse
import pandas as pd
from dotenv import load_dotenv
from src.data.encryption import (
    generate_and_secure_key_in_local_env_file,
    write_pandas_to_encrypted_file,
)

RAW_DATA_PATH_WITH_LABELS = "data/data_2025/raw/Exploitable anonymised letters with new labels.xlsx"
ENCRYPTED_DATASET_PATH = "data/data_2025/processed/dataset.encrypted.csv"
COLUMNS_TO_KEEP = [
    "patient_id",
    "creation_date",
    "Age_creation_date",
    "Anonymised letter",
    "Label_extraction",
]
INPUT_LABEL_KEY_MAP = {
    "Anonymised letter": "input_text",
    "Label_extraction": "label",
}
LABEL_VALUE_MAP = {
    "No FU": float("nan"),
    "No FU yet": float("nan"),
}


def main(args):
    """
    Main function to create and encrypt the dataset
    """
    # Create the dataset
    print("Creating dataset from raw data...")
    df = create_dataset(
        raw_data_path_with_labels=RAW_DATA_PATH_WITH_LABELS,
        columns_to_keep=COLUMNS_TO_KEEP,
    )

    # Save a key to an local environment file, once
    print("Generating encryption key...")
    generate_and_secure_key_in_local_env_file(
        key_var_name=args.key_name,
        env_file_path=args.local_env_file_path,
    )

    # Set the encryption key in a local environment file
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
        encryption_key_var_name=args.key_name,
    )

    # Inform the user that the key should be sent to a remote server
    print(
        f"Encryption key stored in '{args.local_env_file_path}'. "
        "Please securely transfer this file to your remote server"
        ", then delete the local file."
    )


def create_dataset(
    raw_data_path_with_labels: str,
    columns_to_keep: list,
    output_dataset_path: str|None = None,
) -> pd.DataFrame:
    """
    Creates a dataset from an Excel file and saves it as a CSV file.

    Args:
        raw_data_path (str): The path to the raw Excel data file.
        csv_data_path (str): The path where the processed CSV data will be saved.
    """
    # Keep track of document, remove true patient id, and keep only used columns
    df = pd.read_excel(raw_data_path_with_labels)
    df["document_id"] = df.index
    df["patient_id"] = df.groupby("patient_id").ngroup()
    df = df[columns_to_keep]

    # Save the dataset to a CSV file if output path is provided
    if output_dataset_path is not None:
        df.to_csv(output_dataset_path, index=False)
        print(f"Dataset saved (not encrypted) to {output_dataset_path}")

    # Format data fields and label values
    if INPUT_LABEL_KEY_MAP is not None and len(INPUT_LABEL_KEY_MAP) > 0:
        df = df.rename(columns=INPUT_LABEL_KEY_MAP)
    if LABEL_VALUE_MAP is not None and len(LABEL_VALUE_MAP) > 0:
        df = df.replace({"label": LABEL_VALUE_MAP})

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and encrypt dataset.")
    parser.add_argument(
        "--key_name",
        "-kn",
        type=str,
        required=True,
        help="Name of the environment variable for encryption key retrieval.",
    )
    parser.add_argument(
        "--local_env_file_path",
        "-lp",
        type=str,
        required=True,
        help="Path to the local environment file storing the encryption key.",
    )
    args = parser.parse_args()
    main(args)
