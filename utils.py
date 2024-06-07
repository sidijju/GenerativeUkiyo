import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_image(image, path):
    plt.cla()
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.savefig(path)

def plot_batch(batch, path):
    plt.cla()
    grid = vutils.make_grid(batch.cpu()[:25], nrow = 5, padding=2, normalize=True)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(path)

# utility function to iterate through model
# and initalize weights in layers rom N(0, 0.02)
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)