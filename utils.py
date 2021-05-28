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
import import_ipynb
import transforms as my_T
import dataset
from dataset import Microscopy_dataset


def get_transforms(train=False, rescale_size=(256, 256)):
    #transformation definition
    transforms = []
    if train:
        transforms.append(my_T.Rescale(rescale_size))
        transforms.append(my_T.Normalize())
        transforms.append(my_T.ToTensor())
    return my_T.Compose(transforms)

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    num_workers=1,
    pin_memory=True):
    
    #loader for Microscopy dataset - training and validation dataset
    
    train_ds = Microscopy_dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Microscopy_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader



def get_loaders_test(
    # loader fo testing dataset
    val_dir,
    val_maskdir,
    batch_size,
    num_workers=1,
    pin_memory=True):
    print("test get loaders test")
    #loader for Microscopy dataset - training and validation dataset
    
    val_ds = Microscopy_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    return val_loader



def check_acc(loader, model, device="cuda"):
    #check accuracy and dice score deffinition
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    print(type(loader))
    print("Breakpoint!")
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    
    print(f"Dice score: {dice_score/len(loader)}")
    
    model.train()

