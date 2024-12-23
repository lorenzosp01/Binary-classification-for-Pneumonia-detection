import pandas as pd

import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms as tt
import numpy as np

class CheXDataset(Dataset):
    def __init__(self, root_dir, transformations=[]):
        self.root_dir = root_dir

        print("Loading dataset...")
        self.data = pd.read_csv(os.path.join(root_dir, 'train.csv'))

        # Drop columns that are not needed
        self.data.drop(columns=['Sex', 'Age', 'AP/PA', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Edema', 'Consolidation', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'])
        
        # Filter to cases positive for pneumonia or negative for lung opacity
        # as Pneumonia implies Lung Opacity, we can have more samples negative to pneumonia
        self.data = self.data[(self.data['Pneumonia'] == 1) | (self.data['Lung Opacity'] == 0)]

        # Fill missing values
        self.data['Pneumonia'] = self.data['Pneumonia'].fillna(0)

        # Fix cases where Pneumonia is -1 but Lung Opacity is 0, as it implies Pneumonia is 0
        self.data['Pneumonia'] = np.where((self.data['Pneumonia'] == -1) & (self.data['Lung Opacity'] == 0), 0, self.data['Pneumonia'])

        # Convert to int
        self.data['Pneumonia'] = self.data['Pneumonia'].astype(int)

        self.targets = self.data['Pneumonia'].to_list()
        self.classes = self.data['Pneumonia'].unique()

        self.transformations = transformations

    def __len__(self):
        return len(self.data) + len(self.data) * (len(self.transformations) - 1)

    def __getitem__(self, idx):
        return self.loadItem(idx)

    def loadItem(self, idx):
        # If more transformations are specified, we use the modulo operator to index the image
        im_idx = idx % len(self.data)

        # Load image file
        img_path = self.data.iloc[im_idx]['Path']
        label = self.data.iloc[im_idx]['Pneumonia']

        relative_path = img_path[img_path.find('/') + 1:]

        img = Image.open(os.path.join(self.root_dir, relative_path)).convert('RGB')

        # Apply transformations
        if len(self.transformations) > 0:
            # We use the integer division operator to index the transformation, if more than one is specified
            transform_idx = idx // len(self.data)
            img = self.transformations[transform_idx](img)
            return img, int(label)

        raise Exception(f"Image {img_name} not found in dataset")