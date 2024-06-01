import torch
import torchvision.models as models
import torch.nn as nn

def get_resnet_model(num_classes=10, in_channels=3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Використовуємо ResNet18 замість ResNet50 для пришвидшення
    if in_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

