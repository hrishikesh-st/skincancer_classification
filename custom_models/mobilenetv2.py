import torch.nn as nn
import torchvision.models as models

class SkinCancerMobileNetV2(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(SkinCancerMobileNetV2, self).__init__()
        
        self.model = models.mobilenet_v2(pretrained=pretrained)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        in_features = self.model.last_channel  

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        return self.model(x)
