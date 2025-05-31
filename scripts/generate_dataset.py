import os
import re
import pandas as pd
from docx import Document
from tqdm import tqdm


RAW_DATA_FOLDER = "data/raw_docxs"
OUTPUT_DATA_PATH = "data/dataset.csv"
DATA_FILE_PATTERN = re.compile(r"_\d\.docx$")
# SAMPLE_SEPARATION_LINE_PATTERN = re.compile(r"Exemple\s*\d+\s*:?")
SAMPLE_SEPARATION_LINE_PATTERN = re.compile(r"^(Exemple|EXEMPLE) \d.*$")


def main():
    # Define the regex pattern to detect "Exemple N" variations
    data = []
    for file_name in tqdm(os.listdir(RAW_DATA_FOLDER)):
        if DATA_FILE_PATTERN.search(file_name):
            file_path = os.path.join(RAW_DATA_FOLDER, file_name)
            data.extend(parse_docx_folder(file_path))
            
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    
    # Print dataset content
    total_samples = len(df)
    samples_per_label = df["label"].value_counts().sort_index()
    print(f"\nTotal number of samples: {total_samples}")
    print(f"\nNumber of samples per label:\n{samples_per_label}")
    
    
def parse_docx_folder(file_path):
    """ Extract data from a docx file

    Args:
        file_path (str): path to input docx file

    Returns:
        list[dict[str, str]]: list of (input_text, label) pairs
    """
    # Load data and labels
    label = file_path.split("_")[-1].replace(".docx", "")
    doc = Document(file_path)
    
    # Extract input text from file content
    samples_in_this_file = []
    current_text = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        
        # Text matching the "Exemple N" pattern is the start of a new sample
        if SAMPLE_SEPARATION_LINE_PATTERN.match(text):
            if current_text:
                samples_in_this_file.append({
                    "input_text": " ".join(current_text).strip(),
                    "label": label,
                })
                current_text = []
        else:
            # Append the paragraph's text to current example's input text (if not empty)
            if text:
                current_text.append(text)
                
    # After the loop, append the last collected input text
    if current_text:
        samples_in_this_file.append({
            "input_text": " ".join(current_text).strip(),
            "label": label,
        })
    
    return samples_in_this_file


if __name__ == "__main__":
    main()
