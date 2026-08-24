from .Feature_extraction_abs import Feature_extraction_abs

from torchvision.models import resnet18
import torch.nn as nn
import torch
from Utils.device import DEVICE as device

DROPOUT_P = 0.5


class ResNet18_64x64_feature_extraction(Feature_extraction_abs):
    """
        Mirrors Models.ResNet18_backbone.ResNet18_tinyimgnet: a 3x3/stride1 stem
        with no maxpool, and a Sequential(Dropout, Linear) head. Because the
        parametros ficam iguais (conv1.*, bn1.*, layer1-4.*, fc.1.*), o state_dict de um
        ResNet18_tinyimgnet treinado carrega direto aqui via load_model().
    """

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet18()
        self.adjust_output()

    def adjust_output(self):
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        # Sequential(Dropout, Linear) rather than a bare Linear, so the classification weights stay
        # named fc.1.weight / fc.1.bias, matching the ResNet18_tinyimgnet checkpoint.
        self.model.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(self.model.fc.in_features, self.num_classes),
        )
        # Depois das trocas, para que conv1/maxpool/fc tambem vao para a GPU.
        self.model.to(device=device)

    def forward(self, x):
        return self.model(x)
    
    def classify_features(self, x):
        return self.model.fc(x)
    
    def extract_features(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

        return x
    
    def load_model(self, weights):
        self.model.load_state_dict(weights)
