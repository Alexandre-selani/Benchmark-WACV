from PIL import Image
from torchvision.datasets import VisionDataset,ImageFolder
import numpy as np


class PanicumDataset_pytorchOOD(ImageFolder):
    """
    Classe usada para carregar o dataset panicum ja segmentado, PARA USO NO PYTORCH-OOD. As imagens vao estar em um diretorio separadas em classes por pastas dentro do diretorio
    """
    def __init__(self, root,transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        self.targets=np.array(self.targets)
        self.targets = self.targets-1