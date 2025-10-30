import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os

class AgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konverter til RGB

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['age'], dtype=torch.float32)
        return image, label
