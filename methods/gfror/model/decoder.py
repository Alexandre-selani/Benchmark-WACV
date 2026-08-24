# Adapted from https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/network_definitions.py
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Decoder — espelha o Encoder: 7 -> 14 -> 28 -> 56 -> 112 -> 224
# NO Sigmoid/Tanh on the output (see the module docstring)
# ---------------------------------------------------------------------------
class Decoder_eucalyptus(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.linear = nn.Linear(latent_size, 512 * 7 * 7, bias=False)
 
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # 14 -> 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 28 -> 56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 56 -> 112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        # Final layer: biased (no BN follows) and with no saturating activation.
        self.block5 = nn.ConvTranspose2d(64, 3, 4, 2, 1)        # 112 -> 224

        self.activation = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x).view(-1, 512, 7, 7)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        #out = self.activation(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.linear = nn.Linear(latent_size, 512*2*2, bias=False)

        self.block1_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )

        self.block2_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.block3_in = nn.ConvTranspose2d(latent_size,256,1,1,0,bias=False)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,2,1),
            nn.Sigmoid()
        )
        


    def forward(self, x, in_scale=1):

        if in_scale <= 1:
            out = self.linear(x).reshape(-1,512,2,2)

        if in_scale == 2:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block1_in(out)
        if in_scale <= 2:
            out = self.block1(out)

        if in_scale == 4:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block2_in(out)
        if in_scale <= 4:
            out = self.block2(out)

        if in_scale == 8:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block3_in(out)
        if in_scale <= 8:
            out = self.block3(out)

        return self.block4(out)




class Decoder320(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.linear = nn.Linear(latent_size, 512 * 20 * 20)
        
        self.deconv = nn.Sequential(
            # 20x20 -> 40x40
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 40x40 -> 80x80
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 80x80 -> 160x160
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 160x160 -> 320x320
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh() # Ou Sigmoid, dependendo do seu pré-processamento
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 512, 20, 20)
        return self.deconv(out)

class Decoder64(nn.Module):
    """Decoder espelhando o Encoder64: 4 -> 8 -> 16 -> 32 -> 64.

    Saida com Sigmoid porque as imagens do TinyImageNet entram apenas com
    ToTensor() (sem normalizacao), ou seja, ja estao no intervalo [0, 1].
    """

    def __init__(self, latent_size=512):
        super().__init__()
        self.linear = nn.Linear(latent_size, 512 * 4 * 4)

        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 512, 4, 4)
        return self.deconv(out)


class Decoder128(nn.Module):
    """Decoder espelhando o Encoder128: 4 -> 8 -> 16 -> 32 -> 64 -> 128.

    Saida com Sigmoid porque as transformacoes RESNET18_MNIST_OMNI_* do enum
    NAMES terminam em ToTensor() sem normalizacao, ou seja, as imagens de
    inputs already lie in [0, 1].
    """

    def __init__(self, latent_size=256):
        super().__init__()
        self.linear = nn.Linear(latent_size, 512 * 4 * 4)

        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 512, 4, 4)
        return self.deconv(out)
