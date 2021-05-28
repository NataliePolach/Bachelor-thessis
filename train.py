import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np


import import_ipynb
import transforms as my_T
import dataset
from dataset import Microscopy_dataset
import utils
from utils import get_transforms, my_collate, get_loaders, check_acc
import model
from model import UNET

#train the model
def train():
    running_loss = 0.0
    
    model.train()
    for i, (input, mask) in enumerate(train_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]
 
        input.requires_grad = True
        mask.requires_grad = True
        
        # clear the gradients
        optimizer.zero_grad()
        
        #compute the model output
        output = model(input)
        
        #Use this in case of BCE loss
        
        #normalization of output with sigmoid function
        sig = nn.Sigmoid()
        
        #calculate loss
        loss = criterion(sig(output), mask.detach())
        running_loss += loss
        loss.backward()
        # update model weights
        optimizer.step()
        print(f"Epoch: {epoch}, iteration: {i} of {len(train_loader)}, loss: {loss}")
        
        #print("out min:  "+str(out.min()))
        #print("out max:   "+str(out.max()))
        #print("typ out:  "+str(out.type))
        #print("typ mask:  "+str(mask.type))
        
        #Use this in general
            #loss = criterion(output, mask)
            #loss_item = loss.item
            #running_loss += loss_item
            #optimizer.step()
            #print(f"Epoch: {epoch}, iteration: {i} of {len(train_loader)}, loss: {loss_item}")
         

    training_loss.append(running_loss / len(train_loader))

#evaluate the model
def evaluate():
    running_loss_eval = 0.0
    model.eval()
    predictions, actuals = list(), list()
    
    for i, (input, mask) in enumerate(val_loader):
        input, mask = input[0].to(device=device)[None,:,:,:], mask[0].to(device=device)[None,:,:,:]

        with torch.no_grad():
            
            # evaluate the model on the val set
            output = model(input)
            
            #Use this in case of BCE loss
            #normalization of output with sigmoid function
            sig = nn.Sigmoid()
            #calculate loss
            loss = criterion(sig(output), mask.detach())
        
        running_loss_eval += loss
        
        print(f"Eval: {epoch}, iteration: {i} of {len(val_loader)}, loss: {loss}")
        
    eval_loss.append(running_loss_eval / len(val_loader))
            
            #print("out min:  "+str(out.min()))
            #print("out max:   "+str(out.max()))
            #print("typ out:  "+str(out.type))
            #print("typ mask:  "+str(mask.type))
            
            #Use this in general
                #loss = criterion(output, mask)
                #loss_item = loss.item
            #running_loss += loss_item
                #print(f"Epoch: {epoch}, iteration: {i} of {len(train_loader)}, loss: {loss_item}")
    

def plot_losses():
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'b-', label="Trénovací loss",)
    plt.plot(eval_loss, 'y-', label="Validační loss", )
    plt.title("Trénovací a validační loss")
    plt.xlabel('epochy')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f"results/plots/training_eval_loss_{epoch}.png")
    plt.close()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(training_loss, 'g-', label="trénovací loss", )
    plt.title("Trénovací loss")
    plt.xlabel('epochy')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig( f"results/training_loss_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    
    TRAIN_IMG_DIR = 'train_img/'
    TRAIN_MASK_DIR = 'train_mask/'
    VAL_IMG_DIR = 'val_img/'
    VAL_MASK_DIR = 'val_mask/'
    
    epoch = 3
    batch = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 1
    pin_memory = True

    print(f"Running on {device}")
        
    train_loader, val_loader = get_loaders(
        train_dir = TRAIN_IMG_DIR,
        train_maskdir = TRAIN_MASK_DIR,
        val_dir = VAL_IMG_DIR,
        val_maskdir = VAL_MASK_DIR,
        batch_size = batch,
        num_workers = workers,
        pin_memory = pin_memory,
    )
    
    model = UNET(n_channels=1, n_classes=1).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #criterion functions
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    
    training_loss = []
    eval_loss = []


    for epoch in range(epoch):
        
        train()
        evaluate()
        plot_losses()
        
        # check accuracy
        check_acc(val_loader, model, device=device)
                            
        if (epoch % 10) == 0:
            torch.save(model.state_dict(), f"unet__3.pt")
        else:
            torch.save(model.state_dict(), f"unet_3.pt")
    print("Done!")