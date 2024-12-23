import pandas as pd

import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms as tt

class RSNADataset(Dataset):
    def __init__(self, root_dir, transformations=[]):
        self.root_dir = root_dir

        print("Loading dataset...")
        self.data = pd.read_csv(os.path.join(root_dir, 'stage_2_train_labels.csv'))

        self.classes = self.data['Target'].unique()
        self.targets = self.data['Target'].to_list()

        self.transformations = transformations

    def __len__(self):
        return len(self.data) + len(self.data) * (len(self.transformations) - 1)

    def __getitem__(self, idx):
        return self.loadItem(idx)

    def drop(self, drop_class, drop_ratio, seed):
        """ Drops a percentage of the data from a specific class """
        drop_indices = self.data[self.data['Target'] == drop_class].sample(frac=drop_ratio, random_state=seed).index
        self.data.drop(drop_indices, inplace=True)
        self.classes = self.data['Target'].unique()
        self.targets = self.data['Target'].to_list()

    def loadItem(self, idx):
        """ Loads an image from the dataset """
        # If more transformations are specified, we use the modulo operator to index the image
        im_idx = idx % len(self.data)

        # Load image file
        img_name = self.data.iloc[im_idx]['patientId']
        label = self.data.iloc[im_idx]['Target']
        
        # Find image file in the dataset
        image_dir_path = os.path.join(self.root_dir, "train_res_png")
        if os.path.isdir(image_dir_path) and os.path.isfile(os.path.join(image_dir_path, img_name + '.png')):
            img_path = os.path.join(image_dir_path, img_name + '.png')

            img = Image.open(img_path).convert('RGB')

            # Apply transformations
            if len(self.transformations) > 0:
                # We use the integer division operator to index the transformation, if more than one is specified
                transform_idx = idx // len(self.data)
                img = self.transformations[transform_idx](img)

            return img, int(label)
        raise Exception(f"Image {img_name} not found in dataset")