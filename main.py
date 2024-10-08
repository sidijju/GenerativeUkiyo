import torch
import random
import argparse

from models.dc_gan import DCGAN
from models.vae import VAE
from models.vq_vae import VQVAE
from models.ddpm import DDPM
from models.pro_gan import ProGAN
from models.style_gan import StyleGAN
from data.dataset import *
from utils import read_conf

parser = argparse.ArgumentParser()

### General Flags

parser.add_argument('-n', '--n', type=int, default=5, help='number of training epochs')
parser.add_argument('--seed', type=int, default=128, help='manual random seed')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--dim', type=int, default=128, help='image dimension for input and output')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate for training')
parser.add_argument('--decay', type=float, default=0.9999, help='weight decay for EMA')
parser.add_argument('--log_dir', type=str, default=None, help='log dir for training')

### Model Flags

parser.add_argument('--vae', action='store_true', help='train vae model')
parser.add_argument('--vq_vae', action='store_true', help='train vq-vae model')
parser.add_argument('--ddpm', action='store_true', help='train denoising diffusion probablistic model')
parser.add_argument('--pro_gan', action='store_true', help='train progressive gan')
parser.add_argument('--style_gan', action='store_true', help='train style gan')

parser.add_argument('--checkpoint', type=str, default=None, help='train model from checkpoint')
parser.add_argument('--checkpoint_prior', type=str, default=None, help='train prior model from checkpoint')
parser.add_argument('--checkpoint_g', type=str, default=None, help='train generator model from checkpoint')
parser.add_argument('--checkpoint_d', type=str, default=None, help='train discriminator model from checkpoint')

### Dataset Flags

parser.add_argument('--ff', action='store_true', help='use Flicker Faces 128 x 128 dataset (assuming its already downloaded)')
parser.add_argument('--augment', default=False, action='store_true', help='augment dataset')

## Test Flags

parser.add_argument('--test', type=str, default=None, help='test model with weights from input path')
parser.add_argument('--test_n', type=int, default=16, help='number of images to generate')

### Additional Flags

### VAE

parser.add_argument('--beta', type=float, default=.0001, help='beta coefficient for KL term in VAE loss')
parser.add_argument('--mse', default=False, action='store_true', help='flag for MSE loss in VAE')
parser.add_argument('--annealing', default=False, action='store_true', help='flag for beta annealing in VAE loss')
parser.add_argument('--latent', type=int, default=512, help='size of latent dimension')

### VQVAE

parser.add_argument('--k', type=int, default=512, help='embedding dimensionality K for VQ-VAE. overloaded as number of D updates for GAN')
parser.add_argument('--prior_n', type=int, default=10, help='iterations to train PixelCNN prior for VQ-VAE')
parser.add_argument('--prior_lr', type=float, default=1e-06, help='lr to train PixelCNN prior for VQ-VAE')

### DDPM

parser.add_argument('--t', type=int, default=1000, help='noise timesteps for ddpm')
parser.add_argument('--b_0', type=float, default=1e-4, help='beta at timestep 0 for ddpm')
parser.add_argument('--b_t', type=float, default=0.02, help='beta at timestep t for ddpm')
parser.add_argument('--fixed_large', default=False, action='store_true', help='used fixed large for posterior variance')
parser.add_argument('--cosine_lr', default=False, action='store_true', help='use cosine learning rate schedule')

### ProgressiveGAN

parser.add_argument('--lambda_gp', type=int, default=10, help='lambda for Wasserstein gradient penalty loss')

### StyleGAN

parser.add_argument('--w_latent', type=int, default=512, help='latent size for style input in StyleGAN')


args = parser.parse_args()

##### Configs #####

if args.ddpm:
    args = read_conf('models/configs/ddpm.conf', parser)
elif args.vae:
    args = read_conf('models/configs/vae.conf', parser)
elif args.vq_vae:
    args = read_conf('models/configs/vq_vae.conf', parser)
elif args.pro_gan:
    args = read_conf('models/configs/pro_gan.conf', parser)
elif args.style_gan:
    args = read_conf('models/configs/style_gan.conf', parser)
else:
    args = read_conf('models/configs/dc_gan.conf', parser)

##### Setup #####

if not args.test:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

if torch.cuda.is_available():
    print("Using cuda")
    args.device = torch.device("cuda:0")
else: 
    print("Using cpu")
    args.device = torch.device("cpu")

##### Dataset #####

if args.ff:
    dataset = FlickerFacesDataset(args)
else:
    dataset = JapArtDataset(args)
print(f"Dataset Size: {len(dataset)}")

# assuming channels first dataset
args.channel_size = len(dataset[0][0])
args.num_classes = len(dataset.labels_map)

###################

if args.ddpm:
    model = DDPM(args, dataset)
elif args.vae:
    model = VAE(args, dataset)
elif args.vq_vae:
    model = VQVAE(args, dataset)
elif args.pro_gan:
    model = ProGAN(args)
elif args.style_gan:
    model = StyleGAN(args)
else:
    model = DCGAN(args, dataset)

if args.test:
    model.generate(args.test, n=args.test_n)
else:
    model.train()
    model.generate(model.run_dir)
