import random
import os
import glob
import re # Import the regex module

# Define a mapping from filename parts to the canonical labels used in training
FILENAME_TO_LABEL = {
    "Meal-Logging": "meal-logging",
    "Meal-Planning-Recipes": "meal-planning", # Assuming this maps to meal-planning
    "Personalized-Health-Advice": "personalized-health-advice",
    "Educational-Content": "general-education"
}

def load_and_split_data(data_directory="../../data/intents/", train_split=0.8, seed=42):
    """
    Parses intent dataset files from a directory, extracts labels from filenames,
    shuffles the examples, and splits them into training and testing sets.
    Only extracts lines starting with a number, period, and space (e.g., "1. ...").

    Args:
        data_directory (str): The path to the directory containing intent .txt files
                              relative to this script's location.
        train_split (float): The proportion of data to use for training.
        seed (int): Random seed for shuffling to ensure reproducibility.

    Returns:
        tuple: A tuple containing (train_examples, test_examples), where each
               is a list of (text, label) tuples.
    """
    examples = []
    script_dir = os.path.dirname(__file__) # Directory of the current script
    abs_data_dir = os.path.abspath(os.path.join(script_dir, data_directory))

    print(f"Looking for intent files in: {abs_data_dir}")

    if not os.path.isdir(abs_data_dir):
        print(f"Error: Data directory not found at {abs_data_dir}")
        return [], []

    # Find all .txt files starting with 'intent_'
    intent_files = glob.glob(os.path.join(abs_data_dir, "intent_*.txt"))

    if not intent_files:
        print(f"Error: No intent files found in {abs_data_dir}. Looked for pattern 'intent_*.txt'.")
        return [], []

    print(f"Found {len(intent_files)} intent files:")

    for file_path in intent_files:
        filename = os.path.basename(file_path)
        print(f"  Processing file: {filename}")
        current_label = None
        try:
            label_part = filename.split('_')[1].split('.txt')[0]
            current_label = FILENAME_TO_LABEL.get(label_part)
            if not current_label:
                print(f"    Warning: Could not map filename part '{label_part}' to a known label. Skipping file.")
                continue
            print(f"    Assigned label: {current_label}")
        except IndexError:
            print(f"    Warning: Could not parse label from filename '{filename}'. Skipping file.")
            continue

        try:
            # Add errors='ignore' to handle potential encoding issues
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines_processed = 0
                for line in f:
                    line = line.strip()
                    # Use regex to find lines starting with "number. " and capture the rest
                    match = re.match(r"^\d{1,3}\.\s+(.*)", line)
                    if match:
                        utterance = match.group(1).strip()
                        # Remove surrounding quotes if present
                        if utterance.startswith('"') and utterance.endswith('"'):
                            utterance = utterance[1:-1]
                        elif utterance.startswith("'") and utterance.endswith("'"):
                            utterance = utterance[1:-1]
                        
                        if utterance: # Ensure we don't add empty strings
                            examples.append((utterance, current_label))
                            lines_processed += 1
                print(f"    Processed {lines_processed} examples from this file.")

        except Exception as e:
            print(f"    An error occurred while reading file {filename}: {e}")
            continue # Skip to next file on error

    if not examples:
        print("Error: No examples were successfully parsed from any files.")
        return [], []

    # Set seed for reproducibility before shuffling
    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(train_split * len(examples))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    print(f"\nTotal examples parsed: {len(examples)}")
    print(f"Split into Train: {len(train_examples)}, Test: {len(test_examples)}")

    return train_examples, test_examples

if __name__ == "__main__":
    # Example usage: Load data from the default directory and print counts
    train_data, test_data = load_and_split_data()

    if train_data:
        print("\nFirst 3 Training examples:")
        for i, example in enumerate(train_data[:3]):
            print(f"{i+1}: {example}")

    if test_data:
        print("\nFirst 3 Testing examples:")
        for i, example in enumerate(test_data[:3]):
            print(f"{i+1}: {example}") 