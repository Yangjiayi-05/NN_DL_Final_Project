import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset


class SentimentDataset(Dataset):
    def __init__(self, tokenizer_name='bert-base-chinese', max_len=128):
        print("正在从 Hugging Face 自动下载并加载 ChnSentiCorp 数据集...")
        # 为了快速出图，这里仅取前 3000 条样本，如果你需要跑全量，把 [:3000] 去掉即可
        raw_dataset = load_dataset("seamew/ChnSentiCorp", split="train[:3000]")

        self.df = raw_dataset.to_pandas()
        # 统一列名，兼容不同版本的字段命名
        if 'sentence' in self.df.columns and 'text' not in self.df.columns:
            self.df = self.df.rename(columns={'sentence': 'text'})

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