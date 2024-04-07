import os
import glob
import torch
import random
import argparse
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dcgan import DCGAN
from cdcgan import CDCGAN
from vae import VAE
from dataset import JapArtDataset

parser = argparse.ArgumentParser()
parser.add_argument('--vae', action='store_true', help="train and use vae model")
parser.add_argument('--train', action='store_true', help='Set training mode to true')

parser.add_argument('-n', '--n', type=int, default=5, help='number of training epochs')
parser.add_argument('--seed', type=int, default=128, help='Manual random seed')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
parser.add_argument('--latent', type=int, default=100, help='size of latent')
parser.add_argument('--img_dim', type=int, default=128, help='output image dimension')
parser.add_argument('--fm_on', action='store_true', help='Turn feature matching on')
parser.add_argument('--cond', action='store_true', help='train conditional GAN using labels')
parser.add_argument('--flip', action='store_true', help='flip label in gan for better gradient flow')

parser.add_argument('--augment', action='store_true', help='augment dataset')
parser.add_argument('--new_dir', type=str, default='./augment/', help='directory to store after augment')
parser.add_argument('--path', type=str, default='train/discriminator', help='Path to folder with d and g weights')

args = parser.parse_args()

if args.train:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.use_deterministic_algorithms(True)

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
                #v2.AutoAugment(),
                #v2.RandAugment(2),
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

dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

if args.cond:
    model = CDCGAN(args, dataloader)
elif args.vae:
    model = VAE(args, dataloader)
else:
    model = DCGAN(args, dataloader)

if args.train:
    model.train(num_epochs=args.n)
    model.generate(model.run_dir)
else:
    model.generate(args.path)
