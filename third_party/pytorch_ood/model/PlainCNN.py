import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainCNN(nn.Module):
    def __init__(self,num_classes=10, dropout_rate=0.5):
        super(PlainCNN, self).__init__()
        
        # 5 convolutional layers
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(100 * 7 * 7, 500)
        self.fc2 = nn.Linear(500, num_classes)

        # Pooling e Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Bloco conv 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = self.dropout(x)  # Dropout após pooling (regulariza feature maps)

        # Bloco conv 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)

        # Última conv
        x = F.relu(self.conv5(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)  # Dropout antes da camada final
        x = self.fc2(x)
        
        return x
