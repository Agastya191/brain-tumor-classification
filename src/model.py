import torch
import torch.nn
 as nn
from torchvision.models import resnet50, ResNet50_Weights

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        sper(TumorClassifier, self).__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet50()
            
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
