import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
import time

# Import from our other modules
from data_processor import load_and_split_data
from bert_dataset import (
    setup_tokenizer,
    create_data_loaders,
    LABEL_TO_ID, # Use the same mapping
    ID_TO_LABEL,
    NUM_LABELS
)

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./trained_model" # Directory to save the model and tokenizer
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "intent_classifier_model.pt")
TOKENIZER_SAVE_PATH = OUTPUT_DIR
SEED = 42 # For reproducibility

def train_epoch(model, data_loader, optimizer, device):
    """Performs one epoch of training."""
    model.train() # Set model to training mode
    total_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad() # Reset gradients

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels # Pass labels to compute loss internally
        )

        loss = outputs.loss # The model returns CrossEntropyLoss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print progress
        if (i + 1) % 2 == 0: # Print every few batches
            elapsed = time.time() - start_time
            print(f'  Batch {i+1:>3}/{len(data_loader)} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s')

    avg_train_loss = total_loss / len(data_loader)
    epoch_time = time.time() - start_time
    print(f"\n  Average training loss: {avg_train_loss:.4f}")
    print(f"  Training epoch took: {epoch_time:.2f}s")
    return avg_train_loss

def evaluate_model(model, data_loader, device):
    """Evaluates the model on the given data loader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations
        for batch in data_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels # Also get loss in eval for comparison
            )

            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_eval_loss = total_loss / len(data_loader)
    eval_time = time.time() - start_time

    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=list(ID_TO_LABEL.values()),
        labels=list(ID_TO_LABEL.keys()), # Explicitly provide all possible label indices
        zero_division=0 # Avoid warnings for labels with no predictions
    )

    print(f"\n  Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2%}")
    print("  Classification Report:")
    print(report)
    print(f"  Evaluation took: {eval_time:.2f}s")

    return avg_eval_loss, accuracy, report

def main():
    print("Starting BERT Fine-Tuning for Intent Classification...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Load and split data
    print("\n--- Loading Data ---")
    train_examples, test_examples = load_and_split_data(seed=SEED)
    if not train_examples or not test_examples:
        print("Failed to load data. Exiting.")
        return

    # 2. Setup tokenizer
    print("\n--- Setting up Tokenizer ---")
    tokenizer = setup_tokenizer(MODEL_NAME)

    # 3. Create DataLoaders
    print("\n--- Creating DataLoaders ---")
    train_loader, test_loader = create_data_loaders(
        train_examples, test_examples, tokenizer, BATCH_SIZE, MAX_LEN
    )

    # 4. Load BERT model
    print("\n--- Loading Model ---")
    print(f"Loading pre-trained model: {MODEL_NAME} for {NUM_LABELS} labels.")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        # id2label and label2id can be useful for the Trainer API, but not strictly needed here
        # id2label=ID_TO_LABEL,
        # label2id=LABEL_TO_ID
    )
    model.to(device) # Move model to the appropriate device

    # 5. Setup optimizer
    print("\n--- Setting up Optimizer ---")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Using AdamW optimizer with learning rate: {LEARNING_RATE}")

    # 6. Training loop
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {EPOCHS} ========")
        print("Training...")
        train_loss = train_epoch(model, train_loader, optimizer, device)

        print("\nEvaluating...")
        eval_loss, accuracy, report = evaluate_model(model, test_loader, device)

    print("\n--- Training Complete ---")

    # 7. Save the model and tokenizer
    print(f"\n--- Saving Model & Tokenizer to {OUTPUT_DIR} ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save model state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model state dict saved to {MODEL_SAVE_PATH}")

    # Save tokenizer
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    print("\nFine-tuning process finished successfully.")

if __name__ == "__main__":
    main() 