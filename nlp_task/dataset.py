from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, tokenizer_name='bert-base-chinese', max_len=128):
        print("正在从 Hugging Face 加载 ChnSentiCorp 数据集...")
        # 自动下载数据集，加入 trust_remote_code=True 解决高版本库报错问题
        raw_dataset = load_dataset(
            "lansinuote/ChnSentiCorp",
            split="train[:3000]"
        )

        self.data = raw_dataset
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx]['text'])
        label = int(self.data[idx]['label'])

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