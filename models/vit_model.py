import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTPlantDisease(nn.Module):
    """
    Vision Transformer (ViT-Base-16) adapted for Plant Disease Detection.
    """
    def __init__(self, num_classes):
        super(ViTPlantDisease, self).__init__()
        # Load pre-trained ViT model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # We replace the classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

def get_vit_model(num_classes):
    return ViTPlantDisease(num_classes=num_classes)
