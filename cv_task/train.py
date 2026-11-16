import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .model import ResNetMNIST


def train_mnist(epochs=5, batch_size=64, lr=0.001):
    # 确保 results 文件夹存在
    os.makedirs('results', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 自动下载并加载数据
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = ResNetMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / len(test_loader.dataset)
        test_accs.append(acc)
        print(f'Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}, Test Acc = {acc:.2f}%')

    # 绘图
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, epochs + 1), train_losses, color='tab:red', marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(range(1, epochs + 1), test_accs, color='tab:blue', marker='s', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('ResNet on MNIST')
    fig.tight_layout()
    plt.savefig('results/cv_metrics.png')
    print("✅ CV 任务完成，图表已保存至 results/cv_metrics.png")