import os
import glob
import torch
import random
import argparse
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models.dc_gan import DCGAN
from models.cdc_gan import CDCGAN
from models.vae import VAE
from models.ddpm import DDPM
from dataset import *

from utils import make_dir

parser = argparse.ArgumentParser()

### General Flags

parser.add_argument('-n', '--n', type=int, default=5, help='number of training epochs')
parser.add_argument('--seed', type=int, default=128, help='manual random seed')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--latent', type=int, default=512, help='size of latent dimension')

### Model Flags

parser.add_argument('--vae', action='store_true', help='train vae model')
parser.add_argument('--cond', action='store_true', help='train conditional GAN using labels')
parser.add_argument('--ddpm', action='store_true', help='train denoising diffusion probablistic model')

### Dataset Flags

parser.add_argument('--ff', action='store_true', help='use Flicker Faces 128 x 128 dataset (assuming its already downloaded)')
parser.add_argument('--augment', type=str, default=None, help='augment dataset to input directory')

## Test Flags

parser.add_argument('--test', type=str, default=None, help='test model with weights from input path')

### Additional Flags

parser.add_argument('--fm', action='store_true', help='turn feature matching on for GANs')
parser.add_argument('--flip', action='store_true', help='flip label in GANs for better gradient flow')
parser.add_argument('--t', type=int, default=1000, help='noise timesteps for ddpm')
parser.add_argument('--b_0', type=float, default=1e-4, help='beta at timestep 0 for ddpm')
parser.add_argument('--b_t', type=float, default=0.02, help='beta at timestep t for ddpm')
parser.add_argument('--beta', type=float, default=1.0, help='beta coefficient for KL term in VAE loss')
parser.add_argument('--mse', default=False, action='store_true', help='flag for MSE loss in VAE')

args = parser.parse_args()

if not args.test:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    print("Using cuda")
    args.device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    print("Using mps")
    args.device = torch.device("mps")
else: 
    print("Using cpu")
    args.device = torch.device("cpu")

##### Dataset #####

# hardcode to 128 by 128 images for now
# TODO make this a parameter
args.dim = 128

if args.augment:
    make_dir(args.augment)

    to_float32 = v2.ToDtype(dtype=torch.float32, scale=True)

    print("### Augmenting Dataset ###")
    counter = 0
    new_counter = 0
    for f in glob.glob("jap-art/*/*.jpg"):
        counter += 1
        img = to_float32(read_image(f).to(args.device))
        dir_name = f.split('/')[-2]
        img_name = f.split('/')[-1][:-4]
        store_location = args.augment + dir_name + "/" + img_name
        if not os.path.exists(args.augment + dir_name):
            os.makedirs(args.augment + dir_name)
        save_image(img, store_location + ".jpg")

        augment_transforms = [
            v2.RandomHorizontalFlip(p=1.0),
            v2.RandomRotation(30, fill=1),
            v2.RandomResizedCrop(720),
            v2.RandomPerspective(distortion_scale = 0.25, p=1.0, fill=1.0),
        ]

        for i, transform in enumerate(augment_transforms):
            new_counter += 1
            aug_img = transform(img)
            save_image(aug_img, store_location + f"_aug{i}.jpg")

    print(f"Original Dataset Size: {counter}")
    print(f"Augmented Dataset Size: {counter + new_counter}")
    print("#########################")
    
    dataset = JapArtDataset(args)
elif args.ff:
    dataset = FlickerFacesDataset(args)
else:
    dataset = JapArtDataset(args)

# assuming channels first dataset
args.channel_size = len(dataset[0][0])
args.num_classes = len(dataset.labels_map)

dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

if args.ddpm:
    model = DDPM(args, dataloader)
elif args.vae:
    model = VAE(args, dataloader)
# elif args.cond:
#     model = CDCGAN(args, dataloader)
else:
    model = DCGAN(args, dataloader)

if args.test:
    model.generate(args.test)
else:
    model.train(num_epochs=args.n)
    model.generate(model.run_dir)
