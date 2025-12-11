import torch
import torch.nn as nn
from torchvision import models

def get_network(settings):
    name = settings.model
    
    if name == 'baseline':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif name == 'nodown':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        new_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        new_conv.weight = nn.Parameter(model.conv1.weight)
        model.conv1 = new_conv
        model.fc = nn.Linear(model.fc.in_features, 1)

    else:
        raise NotImplementedError('model not recognized')

    if settings.dropout > 0 and settings.task == 'train':
        def new_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = torch.nn.functional.dropout(x, settings.dropout)
            x = self.fc(x)

            return x

        model.forward = type(model.forward)(new_forward, model)

    if settings.task == 'test':
        for param in model.parameters():
            param.requires_grad = False
    else:
        if settings.freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

    return model
