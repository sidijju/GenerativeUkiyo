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
from dataset import JapArtDataset

parser = argparse.ArgumentParser()

### General Flags

parser.add_argument('-n', '--n', type=int, default=5, help='number of training epochs')
parser.add_argument('--seed', type=int, default=128, help='manual random seed')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
parser.add_argument('--latent', type=int, default=100, help='size of latent dimension')
parser.add_argument('--dim', type=int, default=128, help='output image dimension')

### Model Flags

parser.add_argument('--vae', action='store_true', help='train vae model')
parser.add_argument('--cond', action='store_true', help='train conditional GAN using labels')
parser.add_argument('--ddpm', action='store_true', help='train denoising diffusion probablistic model')

### Dataset Flags

parser.add_argument('--augment', action='store_true', help='augment dataset')
parser.add_argument('--new_dir', type=str, default='./augment/', help='directory to store after augment')

## Test Flags

parser.add_argument('--test', default=False, action='store_true', help='set testing mode to true')
parser.add_argument('--weights', type=str, default='train/run', help='path to folder with model weights')

### Additional Flags

parser.add_argument('--fm', action='store_true', help='turn feature matching on for GANs')
parser.add_argument('--flip', action='store_true', help='flip label in GANs for better gradient flow')
parser.add_argument('--t', type=int, default=1000, help='noise timesteps for ddpm')
parser.add_argument('--b_0', type=float, default=1e-4, help='beta at timestep 0 for ddpm')
parser.add_argument('--b_t', type=float, default=0.02, help='beta at timestep t for ddpm')

args = parser.parse_args()

if not args.test:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

if (torch.cuda.is_available()):
    print("Using cuda")
    args.device = torch.device("cuda:0")
else: 
    print("Using cpu")
    args.device = torch.device("cpu")

##### Dataset #####

if args.augment:
    if not os.path.exists(args.new_dir):
        os.makedirs(args.new_dir)

        to_float32 = v2.ToDtype(dtype=torch.float32, scale=True)

        print("### Augmenting Dataset ###")
        counter = 0
        new_counter = 0
        for f in glob.glob("jap-art/*/*.jpg"):
            counter += 1
            img = to_float32(read_image(f).to(args.device))
            dir_name = f.split('/')[-2]
            img_name = f.split('/')[-1][:-4]
            store_location = args.new_dir + dir_name + "/" + img_name
            if not os.path.exists(args.new_dir + dir_name):
                os.makedirs(args.new_dir + dir_name)
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
else:
    dataset = JapArtDataset(args)

args.num_classes = len(dataset.labels_map)

# assuming channels first dataset
args.channel_size = len(dataset[0][0])

dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

if args.ddpm:
    model = DDPM(args, dataloader)
elif args.vae:
    model = VAE(args, dataloader)
elif args.cond:
    model = CDCGAN(args, dataloader)
else:
    model = DCGAN(args, dataloader)

if not args.test:
    model.train(num_epochs=args.n)
    model.generate(model.run_dir)
else:
    model.generate(args.weights)
