import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose

class MyAnimalDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        data_path = os.path.join(root, split)
    
        self.categories = ["butterfly", "cat", "chicken", "cow", "dogs", "elephant", "horse", "sheep", "spider", "squirrel"]
        self.images = []
        self.labels = []
        for idx, category in enumerate(self.categories):
            category_folder_path = os.path.join(data_path, category)
            for image_name in os.listdir(category_folder_path):
                image_path = os.path.join(category_folder_path, image_name)
                self.images.append(image_path)
                self.labels.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
