import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

class CataractGoogLeNet(nn.Module):
    """
    GoogLeNet (GNet) Architecture adapted for Nuclear Cataract Detection & Grading.
    Classifier head outputs logits for 4 diagnostic classes.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(CataractGoogLeNet, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        
        # torchvision requires aux_logits=True when loading default pretrained weights
        self.base_model = googlenet(weights=weights, aux_logits=True)
        
        # Replace FC head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Disable aux_logits after loading weights for fast single-pass execution
        self.base_model.aux_logits = False
        self.base_model.aux1 = None
        self.base_model.aux2 = None
        
    def forward(self, x):
        return self.base_model(x)

    def get_target_layer(self):
        """Returns target conv layer for Grad-CAM ('inception5b')"""
        return self.base_model.inception5b

if __name__ == '__main__':
    model = CataractGoogLeNet(num_classes=4, pretrained=True)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("GoogLeNet output shape:", out.shape)
    print("Target layer:", model.get_target_layer())
