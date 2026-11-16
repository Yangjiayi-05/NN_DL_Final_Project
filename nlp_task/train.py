import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保能找到同级目录的文件

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW  # <--- 修复报错的核心
from dataset import SentimentDataset
from model import TransformerSentimentClassifier


def train_sentiment(epochs=1, batch_size=32, lr=2e-5, max_len=128):  # 演示用，epoch=1即可出图
    os.makedirs('results', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载自动下载的数据集
    dataset = SentimentDataset(max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerSentimentClassifier(num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 (Transformer)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | 平均 Loss: {total_loss / len(dataloader):.4f}")

    print("生成可视化图表...")
    model.eval()
    all_preds, all_labels = [], []
    last_attentions, last_input_ids = None, None

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, attn = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            last_attentions = attn
            last_input_ids = input_ids

    # 1. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Sentiment Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/nlp_confusion_matrix.png')

    # 2. 绘制注意力热力图
    attention_matrix = last_attentions[-1][0][0].cpu().numpy()
    tokens = dataset.tokenizer.convert_ids_to_tokens(last_input_ids[0].cpu().numpy())
    seq_len = min(25, len(tokens))
    attention_matrix = attention_matrix[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title('Transformer Self-Attention Heatmap')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/nlp_attention.png')

    print("✅ NLP 任务完成，图表已保存至 results/ 目录。")


if __name__ == '__main__':
    train_sentiment()