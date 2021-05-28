# %%
import os
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import draw
from skimage.io import imread
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torchvision.transforms.functional as TF
import transforms as my_T


class Microscopy_dataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transforms=None):
        
        #print("Path to image:" + str(image_dir))
        #print("Path to mask:" + str(mask_dir))
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir  
        #self.name = []
        self.transforms = transforms
        #self.model = model

        
        #list of all files in folder
        self.images = os.listdir(image_dir)
        self.images = [ f for f in self.images if os.path.isfile(os.path.join(self.image_dir, f))]
        self.masks = os.listdir(mask_dir)
        self.images = [ f for f in self.images if os.path.isfile(os.path.join(self.mask_dir, f))]

        
    def __len__(self):
        
        #Length of the dataset
        return len(self.images)
    

    def __getitem__(self, index):
        
       #Path for image and mask
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = Image.open(image_path)
        mask= Image.open(mask_path)
        
        self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()
        ])
        
        #print("size obr:"+str(image.size))
        #print("width obr:"+str(image.width))
        #print("height obr:"+str(image.height))

        image = self.transforms(image)
        mask = self.transforms(mask)
        
        #print("tensor size img:"+str(image.size()))
        #print("tensor size mask:"+str(mask.size()))

        
        return image, mask
