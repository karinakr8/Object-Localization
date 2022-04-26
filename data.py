import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform):
        self.annotations = pd.read_csv(annotations_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        labels = self.annotations.Label[index]
        if labels == "Scissors":
            labels = 0
        elif labels == "Panda":
            labels = 1
        else:
            labels = 2

        image_path = self.images_dir + "/" + str(self.annotations.ImageID[index]) + ".jpg"

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        coordinates = torch.tensor([self.annotations.XMin[index], self.annotations.YMin[index],
                                    self.annotations.YMax[index] - self.annotations.YMin[index],
                                    self.annotations.XMax[index] - self.annotations.XMin[index]])
        return img.float(), labels, coordinates.float()

    def unique(self):
        classes = [
            "Scissors",
            "Panda",
            "Snake"
        ]
        return classes