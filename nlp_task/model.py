import torch.nn as nn
from transformers import BertModel

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(TransformerSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', attn_implementation="eager")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))

        if output_attentions:
            return logits, outputs.attentions
        return logits, None