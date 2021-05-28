import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import dataset
from dataset import Microscopy_dataset
import utils
from utils import get_transforms, my_collate, get_loaders
import model
from model import UNET
from torchvision import transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

model = UNET(1,1)
checkpoint = torch.load(Path('/'))
model.load_state_dict(checkpoint)

trans = T.Compose([
                T.Resize((256, 256)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()
            ])

image = Image.open(Path('/'))
mask = Image.open(Path('/'))
input = trans(image)
mask = trans(mask)

input = input.view(1, 1, 256, 256)

output = model(input)

fig = plt.figure(figsize=(40,40), frameon=False)
fig.add_subplot(1, 3, 1).set_axis_off()
plt.imshow(input.detach().cpu().numpy().squeeze(), cmap="gray")
fig.add_subplot(1, 3, 2).set_axis_off()
plt.imshow(mask.detach().cpu().numpy().squeeze(),cmap = "gray")
fig.add_subplot(1, 3, 3).set_axis_off()
plt.imshow(output.detach().cpu().numpy().squeeze())
plt.savefig('/.png')



