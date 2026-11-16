import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='bert-base-chinese', max_len=128):
        self.df = pd.read_csv(csv_file)
        # 兼容处理：如果列名不是 text/label，尝试自动重命名
        if 'text' not in self.df.columns:
            # 假设第一列是文本，第二列是标签
            self.df.columns = ['text', 'label'] + list(self.df.columns[2:])

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = int(self.df.iloc[idx]['label'])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }