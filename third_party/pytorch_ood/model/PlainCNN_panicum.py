import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainCNN_panicum(nn.Module):
    def __init__(self,num_classes=10, dropout_rate=0.5):
        super(PlainCNN_panicum, self).__init__()
        
        # 5 convolutional layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
       # self.conv5 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(100 * 80 * 40, 500)
        self.fc2 = nn.Linear(500, num_classes)

        # Pooling e Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
    
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)  # Dropout antes da camada final
        x = self.fc2(x)
        
        return x
