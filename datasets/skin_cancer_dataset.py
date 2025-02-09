import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # image paths
        self.image_paths = [
            os.path.join(self.root_dir, fname)
            for fname in os.listdir(self.root_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # labels
        if 'Malignant' in root_dir:
            self.label = 1.0
        else:
            self.label = 0.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(self.label, dtype=torch.float)
