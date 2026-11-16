from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, csv_file=None, tokenizer_name='bert-base-chinese', max_len=128):
        """
        注意：为了兼容 Kaggle 环境，这里去掉了本地读取 CSV 的逻辑，
        直接使用 Hugging Face datasets 库下载。
        传入的 csv_file 参数保留是为了不修改 train.py 的调用方式，实际未使用。
        """
        print("正在从 Hugging Face 加载 ChnSentiCorp 数据集...")

        # 核心修改点：加入了 trust_remote_code=True，解决高版本库不支持自定义脚本的问题
        raw_dataset = load_dataset(
            "seamew/ChnSentiCorp",
            split="train[:3000]",
            trust_remote_code=True
        )

        self.data = raw_dataset
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 提取文本和标签
        text = str(self.data[idx]['text'])
        label = int(self.data[idx]['label'])

        # Tokenizer 编码
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