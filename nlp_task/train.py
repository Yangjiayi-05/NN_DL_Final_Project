import os
import sys
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW  # 修复报错：使用 PyTorch 原生的 AdamW

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import SentimentDataset
from model import TransformerSentimentClassifier


def train_nlp():
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    MAX_LEN = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"当前设备: {device}")
    os.makedirs('results', exist_ok=True)

    train_dataset = SentimentDataset(max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerSentimentClassifier(num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 NLP 任务 (Transformer)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {total_loss / len(train_loader):.4f}")

    print("生成可视化图表...")
    model.eval()
    all_preds, all_labels = [], []
    last_attentions = None
    last_input_ids = None

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, attn = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            last_attentions = attn
            last_input_ids = input_ids

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Sentiment Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/nlp_confusion_matrix.png')

    attention_matrix = last_attentions[-1][0][0].cpu().numpy()
    tokens = train_dataset.tokenizer.convert_ids_to_tokens(last_input_ids[0].cpu().numpy())

    seq_len = min(25, len(tokens))
    attention_matrix = attention_matrix[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title('Transformer Self-Attention Heatmap (Last Layer)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/nlp_attention.png')
    print("✅ NLP 任务完成，图表已保存至 results 文件夹")


if __name__ == '__main__':
    train_nlp()