import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class SwinPlantDisease(nn.Module):
    """
    Swin Transformer v2 (Tiny) adapted for Plant Disease Detection.
    """
    def __init__(self, num_classes):
        super(SwinPlantDisease, self).__init__()
        # Load pre-trained Swin Transformer
        self.swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        
        # Replace the final classification head
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.swin(x)

def get_swin_model(num_classes):
    return SwinPlantDisease(num_classes=num_classes)
