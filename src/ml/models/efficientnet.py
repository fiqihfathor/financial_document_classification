import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBase(nn.Module):
    def __init__(self, num_classes: int, model_name: str='efficientnet-b0', pretrained: bool=True):
        """
        Initializes the EfficientNetBase model.
        
        Args:
            num_classes (int): Number of classes in the dataset.
            model_name (str): Name of the EfficientNet model.
            pretrained (bool): If True, loads the pretrained weights.
        
        Returns:
            None
        """
        super(EfficientNetBase, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)