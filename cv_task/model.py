import torch.nn as nn
import torchvision.models as models


class ResNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetMNIST, self).__init__()
        self.resnet = models.resnet18(weights=None)
        # 适配 MNIST 28x28 单通道
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)