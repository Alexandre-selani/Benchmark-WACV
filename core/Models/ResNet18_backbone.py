
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .Base_CAC import BaseCACClassifier


def ResNet18(num_classes, weights=None):
    
    model=None 

    if isinstance(weights, str) or weights is None:
        model=resnet18(weights=weights)
    else:
        model = resnet18()
        weights.pop('fc.weight', None)
        weights.pop('fc.bias', None)
        model.load_state_dict(weights,strict=False)

    model.fc = nn.Linear(512,num_classes)

    return model




class ResNet18_cac(BaseCACClassifier):
    def __init__(self, num_classes=20, weights=None, skip_distances=False, init_weights=False, **kwargs):
        # ResNet18's feature width is fixed at 512.
        self.weights = weights
        super(ResNet18_cac, self).__init__(
            num_classes=num_classes, 
            feat_dim=512, 
            skip_distances=skip_distances,
            init_weights=init_weights
        )

    def _build_encoder(self) -> nn.Module:
        # Instantiate the customized ResNet18.
        encoder = ResNet18(self.num_classes, self.weights)
        
        # Drop the final linear layer, turning it into an Identity.
        encoder.fc = nn.Identity()
        return encoder


class ResNet18Featurizer(nn.Module):
    """Wrapper que retorna (logits, features) no forward, similar ao LeNetFeaturizer.

    Uses the same layer names as the original ResNet18 (conv1, bn1, layer1, ...)
    so that state_dicts stay fully compatible with models trained through ResNet18().
    """

    def __init__(self, num_classes=10, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        # Feature blocks keeping the SAME names as the original ResNet18.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # classifier linear (substitui o fc original)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)                          # (batch, 512, 1, 1)
        feats = torch.flatten(x, 1)                  # (batch, 512)
        logits = self.fc(feats)                      # (batch, num_classes)
        return logits, feats

    def getPerClassWeights(self):
        """Weights of the final classification layer (fc)."""
        with torch.no_grad():
            return self.fc.weight.detach()


class ResNet18_tinyimgnet(nn.Module):
    """ResNet18 with a CIFAR/TinyImageNet-style stem and a dropout head.

    The original stem (7x7/stride2 conv + stride2 maxpool) was designed for 224x224 inputs;
    em imagens 64x64 ele reduz o mapa de features quase a nada antes da layer4, entao aqui ele
    it is replaced by a 3x3/stride1 conv with no maxpool (the usual CIFAR adaptation).
    """

    def __init__(self, num_classes=20, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.Identity()

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_tinyimgnet_featurizer(nn.Module):
    """Wrapper whose forward returns (logits, features), on the ResNet18_tinyimgnet stem.

    Keeps the layer names of ResNet18_tinyimgnet (conv1, bn1, layer1, ..., fc) so
    state_dicts stay compatible with models trained on that backbone.
    """

    def __init__(self, num_classes=20, weights=None):
        super().__init__()
        backbone = ResNet18_tinyimgnet(num_classes=num_classes, weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Same head (Dropout + Linear) as ResNet18_tinyimgnet.
        self.fc = backbone.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)                          # (batch, 512, 1, 1)
        feats = torch.flatten(x, 1)                  # (batch, 512)
        logits = self.fc(feats)                      # (batch, num_classes)
        return logits, feats

    def getPerClassWeights(self):
        """Weights of the classifier's final linear layer."""
        with torch.no_grad():
            if isinstance(self.fc, nn.Sequential):
                return self.fc[-1].weight.detach()
            return self.fc.weight.detach()


class ResNet18_tinyimgnet_cac(BaseCACClassifier):
    """CAC on the ResNet18_tinyimgnet backbone (3x3/stride1 stem, no maxpool, for 64x64).

    Same idea as ResNet18_cac, swapping the standard 7x7/stride2 + maxpool stem for the
    small-image one. The encoder returns the 512 avgpool features (the backbone head
    becomes an Identity) and projection into anchor space is left to BaseCACClassifier's
    `classify` layer.
    """

    def __init__(self, num_classes=20, weights=None, skip_distances=False, init_weights=False, **kwargs):
        # Guardado antes do super().__init__ porque _build_encoder e chamado la dentro.
        self.weights = weights
        super(ResNet18_tinyimgnet_cac, self).__init__(
            num_classes=num_classes,
            feat_dim=512,
            skip_distances=skip_distances,
            init_weights=init_weights
        )

    def _build_encoder(self) -> nn.Module:
        encoder = ResNet18_tinyimgnet(num_classes=self.num_classes, weights=self.weights)
        # Without the classification head the backbone's forward returns features.
        encoder.fc = nn.Identity()
        return encoder


class ResNet18_tinyimgnet_GFROR(nn.Module):
    """ResNet18 for GFROR on TinyImageNet: 6 input channels (x + x_hat), 64x64 images.

    Combines the two adaptations already present in this module:
      - the ResNet18_tinyimgnet stem (3x3/stride1 conv, no maxpool), suited to 64x64;
      - the twin heads of ResNet18_GFROR (classification + transformation prediction).

    As in ResNet18_GFROR, layer4_class and layer4_trans point at the same module: both
    heads share the layer4 weights and diverge only at the linear layers.

    Saidas:
        classification_out: (batch, num_classes)
        transformation_out: (batch, num_transforms)
    """

    def __init__(self, num_classes=20, num_transforms=8, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        # Small-image stem: 64x64 reaches 8x8 by the end of layer4.
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.Identity()

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Same behaviour as ResNet18_GFROR: both attributes point at the same
        # layer4, so the two heads share those weights.
        self.layer4_class = backbone.layer4
        self.avgpool_class = backbone.avgpool

        self.layer4_trans = backbone.layer4
        self.avgpool_trans = backbone.avgpool

        self.classification = nn.Linear(512, num_classes)
        self.transformation = nn.Linear(512, num_transforms)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_class = self.avgpool_class(self.layer4_class(x))
        x_trans = self.avgpool_trans(self.layer4_trans(x))

        x_class = torch.flatten(x_class, 1)
        x_trans = torch.flatten(x_trans, 1)

        classification_out = self.classification(x_class)
        transformation_out = self.transformation(x_trans)
        return classification_out, transformation_out


class ResNet18_GFROR(nn.Module):
    """ResNet18 adapted for GFROR: 6 input channels (x + x_hat), 32x32 images.

    Saidas:
        classification_out: (batch, num_classes)
        transformation_out: (batch, num_transforms)
    """

    def __init__(self, num_classes=10, num_transforms=10, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        # Conv1 — 6 input channels (x + x_hat concatenated), 32x32
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.layer4_class = backbone.layer4
        self.avgpool_class = backbone.avgpool       

        self.layer4_trans = backbone.layer4
        self.avgpool_trans = backbone.avgpool             # adaptive_avg_pool2d(1)


        self.classification = nn.Linear(512, num_classes)
        self.transformation = nn.Linear(512, num_transforms)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_class = self.layer4_class(x)
        x_class = self.avgpool_class(x_class)
        
        x_trans = self.layer4_trans(x)
        x_trans = self.avgpool_trans(x_trans)   
                                    
        x_class = torch.flatten(x_class, 1)
        x_trans = torch.flatten(x_trans,1)                          


        classification_out = self.classification(x_class)
        transformation_out = self.transformation(x_trans)
        return classification_out, transformation_out