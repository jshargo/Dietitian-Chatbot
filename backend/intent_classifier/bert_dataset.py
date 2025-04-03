import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Define label mapping globally or pass it around
LABEL_TO_ID = {
    "meal-logging": 0,
    "meal-planning": 1,
    "personalized-health-advice": 2,
    "general-education": 3
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_TO_ID)

class IntentDataset(Dataset):
    """PyTorch Dataset for BERT intent classification.

    Takes a list of (text, label) examples, tokenizes them, and returns
    tensors for input_ids, attention_mask, and label_id.
    """
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label_str = self.examples[idx]

        # Ensure label exists in mapping
        if label_str not in LABEL_TO_ID:
            raise ValueError(f"Unknown label '{label_str}' encountered in dataset.")

        label_id = torch.tensor(LABEL_TO_ID[label_str], dtype=torch.long)

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Squeeze batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_id
        }

def setup_tokenizer(model_name="bert-base-uncased"):
    """Loads and returns the BERT tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def create_data_loaders(train_examples, test_examples, tokenizer, batch_size=16, max_len=128):
    """Creates train and test DataLoaders from the examples."""
    print(f"Creating datasets with max_len={max_len}...")
    train_dataset = IntentDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataset = IntentDataset(
        examples=test_examples,
        tokenizer=tokenizer,
        max_len=max_len
    )

    print(f"Creating data loaders with batch_size={batch_size}...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False # No need to shuffle test data
    )

    return train_loader, test_loader

# Example usage (can be run directly for testing)
if __name__ == "__main__":
    # Import the data loading function from the other script
    # Note: Assumes data_processor.py is in the same directory
    try:
        from data_processor import load_and_split_data
    except ImportError:
        print("Error: Could not import 'load_and_split_data' from data_processor.py.")
        print("Make sure data_processor.py is in the same directory.")
        exit()

    # 1. Load raw data
    train_ex, test_ex = load_and_split_data()

    if not train_ex or not test_ex:
        print("Failed to load data. Exiting.")
        exit()

    # 2. Setup tokenizer
    tokenizer = setup_tokenizer()

    # 3. Create DataLoaders
    train_dl, test_dl = create_data_loaders(train_ex, test_ex, tokenizer)

    print("\nDataLoaders created successfully.")

    # 4. Inspect a batch from the training loader
    print("\nInspecting first batch from train_loader:")
    try:
        first_batch = next(iter(train_dl))
        print("Batch keys:", first_batch.keys())
        print("Input IDs shape:", first_batch["input_ids"].shape)
        print("Attention Mask shape:", first_batch["attention_mask"].shape)
        print("Labels shape:", first_batch["labels"].shape)
        print("First label ID:", first_batch["labels"][0].item())
        print("Corresponding label:", ID_TO_LABEL[first_batch["labels"][0].item()])
        # Decode first example in batch
        first_input_ids = first_batch["input_ids"][0]
        decoded_text = tokenizer.decode(first_input_ids, skip_special_tokens=False)
        print("\nDecoded first example in batch (incl. special tokens):")
        print(decoded_text)
        # Find original text (requires looking up the label)
        original_label = ID_TO_LABEL[first_batch["labels"][0].item()]
        print(f"(Original label was: {original_label})")

    except Exception as e:
        print(f"Error inspecting batch: {e}") 