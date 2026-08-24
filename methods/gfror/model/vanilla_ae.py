import torch
import torch.nn as nn
from .encoder import Encoder,Encoder320,Encoder64,Encoder128,Encoder_eucalyptus
from .decoder import Decoder,Decoder320,Decoder64,Decoder128,Decoder_eucalyptus

class VanillaAE(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))



class VanillaAE64(nn.Module):
    """Autoencoder para TinyImageNet (entradas 64x64)."""

    def __init__(self, latent_size=512):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder64(latent_size)
        self.decoder = Decoder64(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))



class VanillaAE320(nn.Module):
    def __init__(self, latent_size=1000):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder320(latent_size)
        self.decoder = Decoder320(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))



 
    
 
 

 
 
# ---------------------------------------------------------------------------
# VanillaAE
# ---------------------------------------------------------------------------
class VanillaAE_eucalyptus(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder_eucalyptus(latent_size)
        self.decoder = Decoder_eucalyptus(latent_size)
 
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VanillaAE128(nn.Module):
    """Autoencoder para MNIST+Omniglot com ResNet18 (entradas 128x128).

    Tamanho definido pelas transformacoes RESNET18_MNIST_OMNI_TRAIN_TRANSFORMS /
    RESNET18_MNIST_OMNI_EVAL_TRANSFORMS do enum NAMES: Grayscale(3 canais) +
    Resize(128) + ToTensor(), ou seja, tensores 3x128x128 em [0, 1].
    """

    def __init__(self, latent_size=256):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder128(latent_size)
        self.decoder = Decoder128(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))
