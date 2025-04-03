from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import os

# --- Configuration ---
# These should match the paths used in train_bert.py or be configurable
MODEL_DIR = "./trained_model" # Directory where the model and tokenizer are saved
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier_model.pt")
TOKENIZER_PATH = MODEL_DIR # The directory containing tokenizer files
MAX_LEN = 128 # Should match the training configuration

# Define label mapping (should match the one used during training)
# We can import this from bert_dataset if structure allows, or redefine it
ID_TO_LABEL = {
    0: "meal-logging",
    1: "meal-planning",
    2: "personalized-health-advice",
    3: "general-education"
}
NUM_LABELS = len(ID_TO_LABEL)

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    query: str

class IntentResponse(BaseModel):
    intent: str
    confidence: float | None = None # Optional: Add confidence score

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Nutritional Chatbot Intent Classifier API",
    description="API to classify user queries into nutritional intents using a fine-tuned BERT model.",
    version="1.0.0"
)

# --- Global Variables for Model and Tokenizer ---
# These will be loaded at startup
tokenizer = None
model = None
device = None

# --- Startup Event Handler --- 
@app.on_event("startup")
def load_model_and_tokenizer():
    global tokenizer, model, device
    print("--- Loading Model and Tokenizer --- ")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model and tokenizer paths exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Model path ('{MODEL_PATH}') or Tokenizer path ('{TOKENIZER_PATH}') not found.")
        print("Please ensure the model is trained and saved correctly using train_bert.py")
        # You might raise an exception here or handle it depending on desired behavior at startup
        # For now, we'll allow the app to start but prediction will fail.
        return

    try:
        print(f"Loading tokenizer from: {TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        print(f"Loading model architecture (bert-base-uncased for {NUM_LABELS} labels)")
        # We need to know the base model to load the architecture before loading the state dict
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        
        print(f"Loading fine-tuned model weights from: {MODEL_PATH}")
        # Load the saved state dictionary
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("--- Model and Tokenizer Loaded Successfully ---")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        # Handle error appropriately (e.g., log and exit, or disable prediction endpoint)
        model = None # Ensure model is None if loading failed
        tokenizer = None

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health():
    """Check if the API service is running."""
    # Optionally, check if the model loaded correctly
    model_status = "loaded" if model is not None and tokenizer is not None else "not loaded"
    return {"status": "ok", "model_status": model_status}

@app.post("/predict", response_model=IntentResponse, summary="Predict Intent")
def predict_intent(request: QueryRequest):
    """
    Predicts the intent of a given user query.
    - **query**: The user's text input.
    Returns the predicted intent label.
    """
    global tokenizer, model, device

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Model is not loaded. Please check server logs."
        )

    text = request.query
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Invalid input: 'query' must be a non-empty string.")

    try:
        # Tokenize the input text
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )

        # Move tensors to the correct device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get prediction and confidence (optional)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_id_tensor = torch.max(probabilities, dim=1)
        
        pred_id = pred_id_tensor.item()
        pred_confidence = confidence.item()

        # Map prediction ID to label string
        if pred_id in ID_TO_LABEL:
            predicted_intent = ID_TO_LABEL[pred_id]
        else:
            # This case should ideally not happen if NUM_LABELS matches model output
            raise HTTPException(status_code=500, detail=f"Model predicted an invalid label ID: {pred_id}")

        return {"intent": predicted_intent, "confidence": pred_confidence}

    except Exception as e:
        print(f"Error during prediction for query '{text}': {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# --- Optional: Add main block to run with Uvicorn for local testing ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server locally...")
    # Note: Running this directly might have issues with relative paths
    # unless run from the `backend/intent_classifier` directory.
    # It's usually better to run uvicorn from the terminal:
    # cd backend/intent_classifier
    # uvicorn api:app --reload --port 8000
    
    # Ensure the script is run from the correct directory relative to the model
    # Or adjust MODEL_DIR to be absolute or relative to the expected CWD
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Update model dir to be relative to the script dir if needed
    # This assumes 'trained_model' is in the same dir as api.py
    MODEL_DIR = os.path.join(script_dir, "trained_model")
    MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier_model.pt")
    TOKENIZER_PATH = MODEL_DIR

    print(f"Expecting model in: {MODEL_PATH}")
    print(f"Expecting tokenizer in: {TOKENIZER_PATH}")

    uvicorn.run(app, host="0.0.0.0", port=8000) 