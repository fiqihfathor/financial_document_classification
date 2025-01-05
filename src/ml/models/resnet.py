import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=True):
        """
        Initializes the ResNet model.
        
        Args:
            num_classes (int): Number of classes in the dataset.
            pretrained (bool): If True, loads the pretrained weights.
        
        Returns:
            None
        """
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x)->torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)