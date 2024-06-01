import torch
import torchvision.models as models
import torch.nn as nn

def get_densenet_model(num_classes=10, in_channels=3):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)  # Використовуємо DenseNet121 як менш глибоку модель
    if in_channels == 1:
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model





