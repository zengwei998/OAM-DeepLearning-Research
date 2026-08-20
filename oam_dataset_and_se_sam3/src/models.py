import torch.nn as nn
from torchvision import models


def build_model(num_classes=11):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model