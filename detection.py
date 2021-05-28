import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage import color
import sys
import skimage.io
import math

import import_ipynb
import transforms as my_T
import dataset
from dataset import Microscopy_dataset
import utils
from utils import get_transforms, my_collate, get_loaders_test, check_acc
import model
from model import UNET

#train the model
def detection(model, dataloader = None, image = None):
    
    model.eval()
    print("val_loader:   "+str(val_loader))
    vector_dice = 0
    vector_precission = 0
    vector_recall = 0
    DICE = []
    
    for i, (input, mask) in enumerate(val_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]
        print(f"cycle: {i}")
        
        #compute the model output
        output = model(input)
        
        #normalize the output from 0 to 1
        m = nn.Sigmoid()
        out = m(output)
        
        #reshaping for plot
        out = out.detach().numpy().reshape(256,256)
        array = np.where(out < 0.02, 0, out)
        array = np.reshape(array,(256,256))
        mask = mask.detach().numpy().reshape(256,256)
        input = np.reshape(input,(256,256))
        
        #h_maxima operation --> labels to input
        h = 0.02
        h_maxima = extrema.h_maxima(array, h)
        labels = label(h_maxima)
        coordinates = peak_local_max(labels, indices=True)
        
        #coordinates for x and y axis for labels
        indicesx_pred = coordinates[:, 1]
        indicesy_pred = coordinates[:, 0]
        
        #tracking pixels in mask to validate detection
        indicesx_mask, indicesy_mask=np.where(mask==1)
        coor = zip(indicesx_mask, indicesy_mask)
        revx_mask = indicesx_mask[::-1]
        revy_mask = indicesy_mask[::-1]
        rx_mask = revy_mask
        ry_mask = revx_mask
        
        fig = plt.figure(dpi=200, frameon=False)
        graf1 = fig.add_subplot(1, 2, 1)
        graf1.imshow(input.detach().cpu().numpy(), cmap="gray")
        graf1.plot(coordinates[:, 1], coordinates[:, 0], 'y+', markersize=4)
        graf2 = fig.add_subplot(1, 2, 2)
        graf2.imshow(mask, cmap="gray")
        fig.savefig(f"detected_img/det_{i}.png")

        #evaluate the detection
        """
        dist_matrix = np.zeros((len(rx_mask), len(ry_mask)))

        for row, (g_x, g_y) in enumerate(zip(rx_mask, ry_mask)):
                    for col, (p_x, p_y) in enumerate(zip(indicesx_pred, indicesy_pred)):
                        x = abs(g_x - p_x)
                        y = abs(g_y - p_y)
                        dist_matrix[row, col] = math.sqrt((x*x)+(y*y))

        min_dists = np.amin(dist_matrix, axis=0)

        tp = 0
        fp = 0
        for dist in min_dists:
            if dist <=3:
            #dist_threshold
                tp += 1
            else:
                fp += 1
                
        tp = len(rx_mask) if tp > len(rx_mask) else tp
        fn = len(rx_mask)-tp  

        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        recall = tp / (tp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)

        vector_dice += dice
        vector_precission += precision
        vector_recall += recall
        
        DICE.append(dice)
        
        print(f"TP (pocet detekovanych nanocastic: {tp}, FP: {fp}, FN: {fn}, precision: {precision}, recall: {recall}, dice: {dice}")

        
    
    precission = vector_precission / len(val_loader)
    recall = vector_recall / len(val_loader)
    dice = vector_dice / len(val_loader)
    
    return precission, recall, dice, DICE
    """

if __name__ == "__main__":
    
    #Path to validation masks and images
    VAL_IMG_DIR = 'val_img/'
    VAL_MASK_DIR = 'val_mask/'
    batch = 1
    workers = 1
    pin_memory = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unet_path = "modely_natrenovane/1/unet_1.pt"
            
    val_loader = get_loaders_test(
        val_dir = VAL_IMG_DIR,
        val_maskdir = VAL_MASK_DIR,
        batch_size = batch,
        num_workers = workers,
        pin_memory = pin_memory
    )
    
    model = UNET(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(unet_path))
    model.to(device=device)
    
    detection(model, val_loader)