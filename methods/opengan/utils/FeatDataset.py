from torch.utils.data import Dataset

class FeatDataset(Dataset):
    def __init__(self, data):
        self.samples = data["features"]
        self.labels = data["labels"]
        self.current_set_len = self.samples.shape[0]        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.samples[idx],self.labels[idx]        
        return curdata
