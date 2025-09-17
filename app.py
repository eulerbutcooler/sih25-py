import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
from model import MultiTaskDistilBERT  


# CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
hazard_labels = {0: "No Hazard", 1: "Hazard"}
hazard_type_labels = {0: "Cyclone", 1: "Flood", 2: "Hurricane",3:'Other'}
urgency_labels = {0: "High", 1: "Low", 2: "Medium"}



# LOAD MODEL + TOKENIZER

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = MultiTaskDistilBERT(num_types=len(hazard_type_labels),
                            num_urgency=len(urgency_labels))
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()





# FASTAPI APP

app = FastAPI(title="Hazard Detection API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    # Tokenize
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    # Run inference
    with torch.no_grad():
        hazard_logits, type_logits, urgency_logits = model(
            inputs["input_ids"], inputs["attention_mask"]
        )

    hazard_pred = torch.argmax(hazard_logits, dim=1).item()
    type_pred = torch.argmax(type_logits, dim=1).item()
    urgency_pred = torch.argmax(urgency_logits, dim=1).item()

    return {
        "is_hazard": hazard_labels[hazard_pred],
        "hazard_type": hazard_type_labels.get(type_pred, "Unknown"),
        "urgency": urgency_labels.get(urgency_pred, "Unknown")
    }
