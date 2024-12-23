import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# class RSNADataset(Dataset):
#     def __init__(self, root_dir, transformations=[]):
#         self.root_dir = root_dir

#         print("Loading dataset...")
#         self.data = pd.read_csv(os.path.join(root_dir, 'stage_2_train_labels.csv'))

#         self.classes = self.data['Target'].unique()
#         self.targets = self.data['Target'].to_list()

#         self.transformations = transformations

#     def __len__(self):
#         return len(self.data) + len(self.data) * (len(self.transformations) - 1)

#     def __getitem__(self, idx):
#         return self.loadItem(idx)

#     def drop(self, drop_class, drop_ratio, seed):
#         drop_indices = self.data[self.data['Target'] == drop_class].sample(frac=drop_ratio, random_state=seed).index
#         self.data.drop(drop_indices, inplace=True)
#         self.classes = self.data['Target'].unique()
#         self.targets = self.data['Target'].to_list()

#     def loadItem(self, idx):
#         im_idx = idx % len(self.data)
#         # Load image file
#         img_name = self.data.iloc[im_idx]['patientId']
#         label = self.data.iloc[im_idx]['Target']

#         image_dir_path = os.path.join(self.root_dir, "train_res_png")
#         #print(image_dir_path)
#         #print(os.path.join(image_dir_path, img_name + '.png'))
#         if os.path.isdir(image_dir_path) and os.path.isfile(os.path.join(image_dir_path, img_name + '.png')):
#             img_path = os.path.join(image_dir_path, img_name + '.png')

#             img = Image.open(img_path).convert('RGB')

#             if len(self.transformations) > 0:
#                 transform_idx = idx // len(self.data)
#                 img = self.transformations[transform_idx](img)

#             return img, int(label)
#         raise Exception(f"Image {img_name} not found in dataset")
class RSNADataset():
    data_dir = None
    data = None
    train = None
    test = None

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.__loadImages()
        
    
    def __loadImages(self):
        print("Loading RSNA dataset...")
        data = pd.read_csv(os.path.join(self.data_dir, 'stage_2_train_labels.csv'))
        # classes = data['Target'].unique()
        # targets = data['Target'].to_list()
        # drop_indices = self.data[self.data['Target'] == drop_class].sample(frac=drop_ratio, random_state=seed).index
        # self.data.drop(drop_indices, inplace=True)
        # self.classes = self.data['Target'].unique()
        # self.targets = self.data['Target'].to_list()
        # Iterate through the data and load the images
        images = []
        labels = []
        self.data_dir = os.path.join(self.data_dir, 'train_res_png')
        for idx, row in data.iterrows():
            img_name = row['patientId']
            label = row['Target']
            if os.path.isdir(self.data_dir) and os.path.isfile(os.path.join(self.data_dir, img_name + '.png')):
                img_path = os.path.join(self.data_dir, img_name + '.png')
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping {img_path}: Unable to load image.")
                    continue
                img = cv2.resize(img, (64, 64))
                img = img.reshape(-1) 
                images.append(img)
                labels.append(int(label))
            # if len(self.transformations) > 0:
            #     transform_idx = idx // len(self.data)
            #     img = self.transformations[transform_idx](img)
        print(len(images))
        return np.array(images), np.array(labels)

class ChestXrayDataset():
    data_dir = None
    train = None
    test = None

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train = self.__loadImages('train')
        self.test = self.__loadImages('test')
    

    def __loadImages(self, sub_dir=None):
        print("Loading ChestXray dataset...")
        images = []
        labels = []
        pointed_dir = os.path.join(self.data_dir, sub_dir)
        for class_label in os.listdir(pointed_dir):
            class_path = os.path.join(pointed_dir, class_label)
            if os.path.isdir(class_path):  # Ensure it's a folder
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Skipping {img_path}: Unable to load image.")
                        continue
                    
                    # Resize the image to 64x64
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                    # Append to data and labels
                    img = img.reshape(-1)  # Flatten the image
                    images.append(img)
                    labels.append(class_label)
        print(len(images))
        return np.array(images), np.where(np.array(labels) == 'NORMAL', 0, 1)

class CheXDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.__loadImages()
        
    def __loadImages(self):
        print("Loading CheX dataset...")
        data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        data.drop(columns=['Sex', 'Age', 'AP/PA', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Edema', 'Consolidation', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'])
        data = data[(data['Pneumonia'] == 1) | (data['Lung Opacity'] == 0)]
        data['Pneumonia'] = data['Pneumonia'].fillna(0)
        data['Pneumonia'] = np.where((data['Pneumonia'] == -1) & (data['Lung Opacity'] == 0), 0, data['Pneumonia'])
        data['Pneumonia'] = data['Pneumonia'].astype(int)
        images = []
        labels = []
        for idx, row in data.iterrows():
            img_path = row['Path']
            relative_path = img_path[img_path.find('/') + 1:]
            label=row['Pneumonia'] 
            if os.path.isdir(self.data_dir) and os.path.isfile(os.path.join(self.data_dir, relative_path)):
                img_path = os.path.join(self.data_dir, relative_path)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping {img_path}: Unable to load image.")
                    continue
                img = cv2.resize(img, (64, 64))
                img = img.reshape(-1) 
                images.append(img)
                labels.append(int(label))
        print(len(images))
        return np.array(images), np.array(labels)
    