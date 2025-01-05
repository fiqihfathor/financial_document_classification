import torch
import torch.nn as nn
import timm 

class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, model_name: str="vit_base_patch16_224", pretrained: bool=True):
        """
        Vision Transformer model for image classification.
        
        Args:
            num_classes (int): Number of classes in the dataset.
            model_name (str): Name of the Vision Transformer model.
            pretrained (bool): Use pretrained weights or not.
        """
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x)-> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)