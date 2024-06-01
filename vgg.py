import torch
import torchvision.models as models
import torch.nn as nn

def get_vgg_model(num_classes=10, in_channels=3):
    model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)  # Використовуємо VGG11 з Batch Normalization для пришвидшення
    if in_channels == 1:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )
    return model



