import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms.v2 as v2

to_512 = v2.Resize(512)

def extract(t):
    for _ in range(4 - len(t.shape)):
        t = torch.unsqueeze(t, -1)
    return t

def scale_0_1(image):
    # scale any Tensor to 0 to 1
    return v2.ToDtype(torch.float32, scale=True)(v2.ToDtype(torch.float64, scale=True)(image))

def scale_minus1_1(image):
    # scale 0 to 1 to -1 to 1
    return image * 2 - 1

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_image(image, path):
    plt.cla()
    plt.axis('off')
    new_image = to_512(image).cpu().permute(1, 2, 0)
    plt.imshow(new_image)
    plt.savefig(path, bbox_inches=0)

def plot_batch(batch, path):
    plt.cla()
    grid = vutils.make_grid(batch.cpu()[:16], nrow = 4, padding=2, normalize=True)
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

def is_outlier(points, thresh=3.5):
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

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