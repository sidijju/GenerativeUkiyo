import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import torch.optim as optim
from torchvision.transforms import v2
from tqdm import tqdm
from models.gan import GAN
from utils import *
import math
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter
from data.dataset import JapArtDataset
from accelerate import Accelerator

##### ProGAN #####

class ProGAN(GAN):

    def __init__(self, 
                args
                ):
        
        self.args = args

        if not self.args.test:
            self.run_dir = f"train/progan-n={self.args.n}_lr={self.args.lr}_dim={self.args.dim}/"
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

            save_conf(self.args, self.run_dir)

        def res_to_batch(res):
            ratio = args.dim / res
            if ratio <= 1:
                return 4
            elif ratio <= 16:
                return 8
            else:
                return 16
            
        self.resolutions = [4 * (2 ** i) for i in range(args.dim.bit_length() - 2)]
        self.batch_sizes = [res_to_batch(res) for res in self.resolutions]
        self.dataloaders = [self.get_dataloader(res) for res in self.resolutions]
        self.accelerator = Accelerator()

    def get_dataloader(self, resolution):
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(resolution),
            v2.RandomHorizontalFlip(p=0.5) if self.args.augment else v2.Identity(),
        ])
        batch_size = self.batch_sizes[int(math.log2(resolution / 4))]
        return DataLoader(JapArtDataset(self.args, transform=transform), batch_size=batch_size, shuffle=True)

    def save_train_data(self, d_losses, g_losses, d_net, g_net):

        # save models
        torch.save(d_net.state_dict(), self.run_dir + '/discriminator.pt')
        torch.save(g_net.state_dict(), self.run_dir + '/generator.pt')

        filtered_g = savgol_filter(g_losses, 51, 3)
        filtered_d = savgol_filter(d_losses, 51, 3)

        # save losses
        plt.cla()
        plt.figure(figsize=(10,5))
        plt.yscale('log')
        plt.title("Training Losses")
        plt.plot(filtered_d + filtered_g, label="Loss")
        plt.plot(filtered_g, label="G")
        plt.plot(filtered_d, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")
 
        plt.cla()
        plt.figure(figsize=(10,5))
        plt.yscale('log')
        plt.title("Training Losses")
        plt.plot(filtered_g, label="G")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "g_losses")

        plt.cla()
        plt.figure(figsize=(10,5))
        plt.yscale('log')
        plt.title("Training Losses")
        plt.plot(filtered_d, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "d_losses")

    def generate(self, path, n=5):
        print("### Begin Generating Images ###")
        g_net = Generator(self.args, self.channel_size, self.latent_size)
        g_net.load_state_dict(torch.load(path + "/generator.pt"))
        g_net.to(self.args.device)
        g_net.eval()

        noise = torch.randn(n, self.latent_size, 1, 1, device=self.args.device)
        batch, _ = next(iter(self.dataloaders[-1]))

        with torch.no_grad():
            fake = self.accelerator.gather(g_net(noise, len(self.resolutions), 1))

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            plot_image(fake[i], path + f"/f_{i}")
        print("### Done Generating Images ###")
        
    def compute_gradient_penalty(self, d_net, batch, fake, p, alpha):
        B, C, H, W = batch.shape
        beta = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(batch.get_device())
        interpolated_images = batch * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)
        mixed_scores = d_net(interpolated_images, p, alpha)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        norm = gradient.norm(2, dim=1)
        return torch.mean((norm - 1) ** 2)
    
    def train(self):
        
        d_net = Discriminator(self.args)
        if self.args.checkpoint_d:
            d_net.load_state_dict(torch.load(self.args.checkpoint_d, map_location=self.accelerator.device))
            print("Loaded discriminator checkpoint from", self.args.checkpoint_d)
        d_optimizer = optim.Adam(d_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        g_net = Generator(self.args)
        if self.args.checkpoint_g:
            g_net.load_state_dict(torch.load(self.args.checkpoint_g, map_location=self.accelerator.device))
            print("Loaded generator checkpoint from", self.args.checkpoint_g)
        g_optimizer = optim.Adam(g_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.accelerator.device)

        d_net, g_net, d_optimizer, g_optimizer = self.accelerator.prepare(
            d_net, g_net, d_optimizer, g_optimizer
        )

        d_losses = []
        g_losses = []

        print("### Begin Training Procedure ###")

        for p, resolution in tqdm(enumerate(self.resolutions), position=0, desc=f"Progression"):
            make_dir(self.progress_dir + f"res:{resolution}")
            self.dataloaders[p] = self.accelerator.prepare(self.dataloaders[p])
            alpha = 0

            for epoch in tqdm(range(self.args.n), position=1, desc="Epoch", leave=False):
                for i, batch in enumerate(self.dataloaders[p]):
                    batch, _ = batch

                    if epoch == 0:
                        plot_batch(batch, self.progress_dir + f"res:{resolution}_train_example")

                    # generate fake batch for training
                    noise = torch.randn(batch.shape[0], self.latent_size, 1, 1, device=self.accelerator.device)
                    fake_batch = g_net(noise, p, alpha)

                    #############################
                    #### Train Discriminator ####
                    #############################

                    d_net.zero_grad()

                    dx = d_net(batch, p, alpha).view(-1)
                    dgz_1 = d_net(fake_batch.detach(), p, alpha).view(-1)
                    gp = self.compute_gradient_penalty(d_net, batch, fake_batch, p, alpha)
                    d_loss = -(torch.mean(dx) - torch.mean(dgz_1)) + self.args.lambda_gp * gp + (0.001 * torch.mean(dx ** 2))

                    self.accelerator.backward(d_loss)
                    d_optimizer.step()

                    #############################
                    ####   Train Generator   ####
                    #############################

                    g_net.zero_grad()

                    dgz_2 = d_net(fake_batch, p, alpha).view(-1)
                    g_loss = -torch.mean(dgz_2)

                    self.accelerator.backward(g_loss)
                    g_optimizer.step()

                    #############################

                    # update alpha
                    alpha += batch.shape[0] / (self.args.n * len(self.dataloaders[p]) * 0.5)
                    alpha = min(alpha, 1)

                    #############################
                    ####   Metrics Tracking  ####
                    #############################

                    if i % 1000 == 0:
                        print(f'[%d/%d][%d/%d]\td_loss: %.4f\tg_loss: %.4f\talpha: %.4f'
                            % (epoch, self.args.n, i, len(self.dataloaders[p]),
                            d_loss.item(), g_loss.item(), alpha))

                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())

                if epoch % 2 == 0 or epoch == self.args.n-1:
                    with torch.no_grad():
                        g_net.eval()
                        fake = g_net(fixed_latent, p, 1).detach()
                        g_net.train()
                    plot_batch(fake, self.progress_dir + f"res:{resolution}/epoch:{epoch}")

        print("### End Training Procedure ###")
        self.save_train_data(d_losses, g_losses, d_net, g_net)
                
###############

class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def forward(self, x):
        return x/torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()

        self.use_pixelnorm = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.pn = PixelNorm() if use_pixelnorm else nn.Identity()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.pn(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pn(x)
        return x

####   Generator   #####

class Generator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 1, 2, 4, 8, 16]):
        super(Generator, self).__init__()
        self.args = args

        self.embed = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(args.latent, args.latent, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(args.latent, args.latent, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        hidden_dims = [int(args.latent / mult) for mult in dim_mults]

        self.progressive_blocks = nn.ModuleList([
            *[
                ConvBlock(in_f, out_f) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        ])

        self.out_blocks = nn.ModuleList([
            WSConv2d(args.latent, args.channel_size, 1, 1, 0),
            *[
                WSConv2d(out_f, args.channel_size, 1, 1, 0) for out_f in hidden_dims[1:]
            ],
        ])

    def fade(self, lower, higher, alpha):
        return F.sigmoid(alpha * higher + (1 - alpha) * lower)

    def forward(self, x, p, alpha):
        out = self.embed(x)

        if p == 0:
            return self.out_blocks[0](out)
        
        for i in range(p):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.progressive_blocks[i](upscaled)

        upscaled = self.out_blocks[p-1](upscaled)
        out = self.out_blocks[p](out)
        
        return self.fade(upscaled, out, alpha)
    
#########################
    
##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 1, 2, 4, 8, 16]):
        super(Discriminator, self).__init__()
        self.args = args

        hidden_dims = [int(args.latent / mult) for mult in reversed(dim_mults)]

        self.progressive_blocks = nn.ModuleList([
            *[
                ConvBlock(in_f, out_f, use_pixelnorm=False) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        ])

        self.out_blocks = nn.ModuleList([
            *[
                WSConv2d(args.channel_size, in_f, 1, 1, 0) for in_f in hidden_dims
            ],
        ])

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.map = nn.Sequential(
            WSConv2d(args.latent + 1, args.latent, 3, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(args.latent, args.latent, 4),
            nn.LeakyReLU(0.2),
            WSConv2d(args.latent, 1),
        )

    def fade(self, lower, higher, alpha):
        return alpha * higher + (1 - alpha) * lower
    
    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)
    
    def forward(self, x, p, alpha):
        rev_p = len(self.progressive_blocks) - p
        out = F.leaky_relu(self.out_blocks[rev_p](x), 0.2)

        if p == 0:
            out = self.minibatch_std(out)
            return self.map(out).view((out.shape[0], -1))
        
        downscaled = F.leaky_relu(self.out_blocks[rev_p + 1](self.downsample(x)))
        out = self.downsample(self.progressive_blocks[rev_p](out))
        out = self.fade(downscaled, out, alpha)

        for i in range(rev_p + 1, len(self.progressive_blocks)):
            out = self.progressive_blocks[i](out)
            out = self.downsample(out)

        out = self.minibatch_std(out)
        return self.map(out).view((out.shape[0], -1))
    
#########################