from torchvision.models import resnet18
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
from Utils.device import DEVICE as device
class Feature_extraction_abs(nn.Module, ABC):
    def __init__(self,num_classes:int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Standard forward pass."""
        pass

    @abstractmethod
    def extract_features(self, x):
        """Returns the features before the classification layer."""
        pass
    
    @abstractmethod
    def classify_features(self,x):
        """Classifies already-extracted features."""
        pass
    @abstractmethod
    def adjust_output(self):
        """Adapts the output layer to num_classes."""
        pass
    
    def save_features(self,dataloader,save_dir,features_name,original_labels=None):
        all_features = []
        all_labels = []

        
        
        for X,y in dataloader:
            X = X.to(device)
            feat = self.extract_features(X)
            feat = feat.float()
            all_features.append(feat.cpu())
            all_labels.append(y.float())

        #print(all_labels)
        data_to_save = {
        'features': torch.cat(all_features, dim=0),
        'labels': torch.cat(all_labels, dim=0)
        }
      
        if original_labels is not None:
            data_to_save["original_labels"] = torch.cat([t.unsqueeze(0) for t in original_labels], dim=0)

        # Create the directory if it does not exist.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Write the file.
        file_path = os.path.join(save_dir, f"{features_name}_features.pt")
        torch.save(data_to_save, file_path)
        print(f"wrote {file_path} | shape {data_to_save['features'].shape}")
    
    @abstractmethod
    def load_model(self,weights):
        pass
