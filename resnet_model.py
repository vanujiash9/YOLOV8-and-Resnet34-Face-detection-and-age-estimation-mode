import torch
import torch.nn as nn
from torchvision import models

class UTKFaceResNet34(nn.Module):
    def __init__(self, num_age_classes=101, num_gender_classes=2, num_race_classes=5):
        super().__init__()
        base_model = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Bỏ layer cuối
        in_features = base_model.fc.in_features

        self.age_head = nn.Linear(in_features, num_age_classes)
        self.gender_head = nn.Linear(in_features, num_gender_classes)
        self.race_head = nn.Linear(in_features, num_race_classes)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        age = self.age_head(x)
        gender = self.gender_head(x)
        race = self.race_head(x)
        return age, gender, 


