import torch.nn as nn
import torch.nn.functional as F

class SimpleMNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMNIST_CNN, self).__init__()
        
        # Bloco 1: (Input: 1x28x28) -> (Output: 16x14x14)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduz para 14x14

        # Bloco 2: (Input: 16x14x14) -> (Output: 32x7x7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        # Reutiliza self.pool, que reduz para 7x7

        # Camada classificadora
        # A entrada é 32 canais * 7 pixels de altura * 7 pixels de largura
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # Passagem pelo bloco 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Passagem pelo bloco 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Achatamento (flatten) para a camada linear
        x = x.view(-1, 32 * 7 * 7)
        
        # Camada de saída
        x = self.fc1(x)
        return x