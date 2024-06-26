import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_image(image, path):
    plt.cla()
    plt.axis('off')
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.savefig(path, bbox_inches=0)

def plot_batch(batch, path):
    plt.cla()
    grid = vutils.make_grid(batch.cpu()[:25], nrow = 5, padding=2, normalize=True)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(path, bbox_inches=0)

def plot_compare_batch(batch_xy, batch_yhat, path):
    plt.cla()
    grid_images = []
    for i in range(5):
        grid_images += batch_xy[i].cpu(), batch_yhat[i].cpu()
    grid = vutils.make_grid(grid_images, nrow=2, padding=2, normalize=True)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(path, bbox_inches=0)

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