import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================
# 1. SE Block（论文核心）
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // r)
        self.fc2 = nn.Linear(channels // r, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))  # GAP
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


# =========================
# 2. Physics Prior Injection
# =========================
class PhysicsPrior(nn.Module):
    """
    模拟论文中的：
    湍流强度 + SNR + 相位扰动 bias
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 轻量物理扰动注入（论文关键点）
        return x * self.gamma + self.beta


# =========================
# 3. SE-SAM3-PPI 主模型
# =========================
class SESAM3PPI(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        # ---- Backbone (ResNet50替代SAM encoder) ----
        backbone = models.resnet50(pretrained=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ---- SE + Physics ----
        self.se = SEBlock(2048)
        self.ppi = PhysicsPrior(2048)

        # ---- Classifier ----
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # SE + Physics Injection
        x = self.se(x)
        x = self.ppi(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x