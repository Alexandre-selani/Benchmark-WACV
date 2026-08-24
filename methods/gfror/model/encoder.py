# Adapted from https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/network_definitions.py
import torch
import torch.nn as nn

from .utils import clamp_to_unit_sphere

class Encoder_eucalyptus(nn.Module):
    def __init__(self, latent_size=512, normalize_latent=True):
        super().__init__()
        self.normalize_latent = normalize_latent
 
        self.block1 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),   # 224 -> 112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
 
        self.block2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
 
        self.block3 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
 
        self.block4 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
 
        self.block5 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),  # 14 -> 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
 
        self.linear = nn.Linear(512 * 7 * 7, latent_size)

    def forward(self, x):
            out = self.block1(x)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.block5(out)
            out = self.linear(out.reshape(out.shape[0], -1))
        
            out = clamp_to_unit_sphere(out)
            
            return out
    
class Encoder(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(3,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Dropout2d(0.2),
            nn.Conv2d(128,128,3,1,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,1,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),   
        )

        self.block2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(128,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.block3 = nn.Sequential(
            nn.Dropout2d(0.2), 
            nn.Conv2d(128,128,3,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.block1_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)
        self.block2_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)
        self.block3_out = nn.Conv2d(128,latent_size,3,1,1,bias=False)

        self.linear = nn.Linear(128*2*2, latent_size)

    def forward(self, x, out_scale=1):

        out = self.block1(x)

        if out_scale == 8:
            out = self.block1_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)
        
        out = self.block2(out)

        if out_scale == 4:
            out = self.block2_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)

        out = self.block3(out)

        if out_scale == 2:
            out = self.block3_out(out).view(out.shape[0],-1)
            return clamp_to_unit_sphere(out, out_scale*out_scale)
        
        out = self.linear(out.view(out.shape[0],-1))
        
        return clamp_to_unit_sphere(out)

import torch
import torch.nn as nn

class Encoder320(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.features = nn.Sequential(
            # 320x320 -> 160x160
            nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 160x160 -> 80x80
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 80x80 -> 40x40
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 40x40 -> 20x20
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Linear(512 * 20 * 20, latent_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return clamp_to_unit_sphere(out)



class Encoder64(nn.Module):
    """Encoder para TinyImageNet (64x64), no mesmo estilo do Encoder320.

    Spatial reduction: 64 -> 32 -> 16 -> 8 -> 4, with the latent projected
    na esfera unitaria (mesma convencao dos demais encoders do repo).
    """

    def __init__(self, latent_size=512):
        super().__init__()
        self.features = nn.Sequential(
            # 64x64 -> 32x32
            nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Linear(512 * 4 * 4, latent_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return clamp_to_unit_sphere(out)


class Encoder128(nn.Module):
    """Encoder for MNIST+Omniglot with ResNet18 (128x128 inputs).

    As transformacoes RESNET18_MNIST_OMNI_* do enum NAMES entregam imagens
    3x128x128 em [0, 1] (Grayscale(3) + Resize(128) + ToTensor()), entao o
    encoder faz cinco reducoes espaciais: 128 -> 64 -> 32 -> 16 -> 8 -> 4,
    com o latente projetado na esfera unitaria (mesma convencao dos demais
    encoders do repo).
    """

    def __init__(self, latent_size=256):
        super().__init__()
        self.features = nn.Sequential(
            # 128x128 -> 64x64
            nn.Dropout2d(0.2),
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Linear(512 * 4 * 4, latent_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return clamp_to_unit_sphere(out)
