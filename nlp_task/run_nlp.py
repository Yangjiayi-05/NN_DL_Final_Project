import os
from nlp_task.train import train_sentiment


def find_kaggle_dataset():
    """自动在 Kaggle 环境中寻找 csv 数据文件"""
    kaggle_input_path = '/kaggle/input'
    if os.path.exists(kaggle_input_path):
        for root, dirs, files in os.walk(kaggle_input_path):
            for file in files:
                if file.endswith('.csv'):
                    return os.path.join(root, file)
    return None


if __name__ == '__main__':
    # 1. 尝试寻找本地文件
    data_path = 'data.csv'

    # 2. 如果本地没有，且在 Kaggle 环境，自动探测
    if not os.path.exists(data_path):
        print("本地未找到 data.csv，尝试在 Kaggle input 中寻找...")
        kaggle_path = find_kaggle_dataset()
        if kaggle_path:
            data_path = kaggle_path
            print(f"✅ 成功找到 Kaggle 数据集: {data_path}")
        else:
            print("❌ 错误：未找到任何 csv 数据文件。请确保已上传数据。")
            exit()

    print(f"--- 开始执行 NLP 任务 (Transformer + 情感分析) ---")
    print(f"数据集路径: {data_path}")
    train_sentiment(data_path=data_path, epochs=3, batch_size=16, lr=2e-5)