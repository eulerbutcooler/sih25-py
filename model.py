import torch.nn as nn
from transformers import DistilBertModel

class MultiTaskDistilBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_types=4, num_urgency=3):
        super(MultiTaskDistilBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.hazard_classifier = nn.Linear(hidden_size, 2)      # Binary
        self.type_classifier = nn.Linear(hidden_size, num_types)  # Multiclass
        self.urgency_classifier = nn.Linear(hidden_size, num_urgency)  # Multiclass

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        hazard_logits = self.hazard_classifier(pooled_output)
        type_logits = self.type_classifier(pooled_output)
        urgency_logits = self.urgency_classifier(pooled_output)

        return hazard_logits, type_logits, urgency_logits