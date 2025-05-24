from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
class CountryModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, image_features, return_logits = False):
        x = image_features
        logits = self.classifier(x)
        if return_logits:
            return logits
        probs = F.softmax(logits, dim=-1) 
        return probs

    def to(self, device):
        self.classifier.to(device)
        return self

    def parameters(self):
        return list(self.classifier.parameters())
    
    
class CityModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, num_classes)
        )

    def forward(self, image_features, return_logits = False):
        x = image_features
        logits = self.classifier(x)
        if return_logits:
            return logits
        probs = F.softmax(logits, dim=-1) 
        return probs

    def to(self, device):
        self.classifier.to(device)
        return self

    def parameters(self):
        return list(self.classifier.parameters())