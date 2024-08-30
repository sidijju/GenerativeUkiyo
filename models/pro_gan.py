import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import torch.optim as optim
from torchvision.transforms import v2
from tqdm import tqdm
from models.gan import GAN
from utils import *
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter
from data.dataset import JapArtDataset

##### ProGAN #####

class ProGAN(GAN):

    def __init__(self, 
                args
                ):
        
        self.args = args

        if not self.args.test:
            self.run_dir = f"train/progan_n={self.args.n}_lr={self.args.lr}_dim={self.args.dim}/"
            self.progress_dir = self.run_dir + "progress/"

            make_dir(self.run_dir)
            make_dir(self.progress_dir)
            save_conf(self.args, self.run_dir)

        self.resolutions = [4 * (2 ** i) for i in range(args.dim.bit_length() - 2)]
        self.dataloaders = [self.get_dataloader(res) for res in self.resolutions]

    def get_dataloader(self, resolution):
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(resolution),
            v2.RandomHorizontalFlip(p=0.5) if self.args.augment else v2.Identity(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return DataLoader(JapArtDataset(self.args, transform=transform), batch_size=self.args.batchsize, shuffle=True)

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
        g_net = Generator(self.args)
        g_net.load_state_dict(torch.load(path + "/generator.pt", map_location=torch.device(self.args.device)))
        g_net.to(self.args.device)
        g_net.eval()

        noise = torch.randn(n, self.args.latent, 1, 1, device=self.args.device)
        batch, _ = next(iter(self.dataloaders[-1]))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake = g_net(noise, len(self.resolutions)-1, 1)

        for i in range(n):
            plot_image((batch[i] + 1)/2, path + f"/r_{i}")
            plot_image((fake[i] + 1)/2, path + f"/f_{i}")
        print("### Done Generating Images ###")
        
    def compute_gradient_penalty(self, d_net, batch, fake, p, alpha):
        B, C, H, W = batch.shape
        beta = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(self.args.device)
        interpolated_images = batch * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)
        preds = d_net(interpolated_images, p, alpha)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=preds,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        norm = gradient.norm(2, dim=1)
        return torch.mean((norm - 1) ** 2)
    
    def train(self):
        
        d_net = Discriminator(self.args)
        if self.args.checkpoint_d:
            d_net.load_state_dict(torch.load(self.args.checkpoint_d, map_location=self.args.device))
            print("Loaded discriminator checkpoint from", self.args.checkpoint_d)
        d_net.to(self.args.device)
        d_optimizer = optim.Adam(d_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        g_net = Generator(self.args)
        if self.args.checkpoint_g:
            g_net.load_state_dict(torch.load(self.args.checkpoint_g, map_location=self.args.device))
            print("Loaded generator checkpoint from", self.args.checkpoint_g)
        g_net.to(self.args.device)
        g_optimizer = optim.Adam(g_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        d_losses = []
        g_losses = []

        fixed_latent = torch.randn(self.args.batchsize, self.args.latent, 1, 1, device=self.args.device)

        print("### Begin Training Procedure ###")
        for p, resolution in tqdm(enumerate(self.resolutions), position=0, desc=f"Progression"):
            make_dir(self.progress_dir + f"res_{resolution}/")
            alpha = 0
            
            for epoch in tqdm(range(self.args.n), position=1, desc="Epoch", leave=False):
                for i, batch in enumerate(self.dataloaders[p]):
                    batch, _ = batch
                    batch = batch.to(self.args.device)

                    if epoch == 0:
                        plot_batch(batch, self.progress_dir + f"res_{resolution}_train_example")

                    # latent for training
                    noise = torch.randn(batch.shape[0], self.args.latent, 1, 1, device=self.args.device)

                    #############################
                    #### Train Discriminator ####
                    #############################

                    d_net.zero_grad()

                    fake_batch = g_net(noise, p, alpha)
                    dx = d_net(batch, p, alpha).view(-1)
                    dgz_1 = d_net(fake_batch.detach(), p, alpha).view(-1)
                    gp = self.compute_gradient_penalty(d_net, batch, fake_batch, p, alpha)
                    d_loss = torch.mean(dgz_1) - torch.mean(dx)
                    d_loss += self.args.lambda_gp * gp
                    d_loss += 0.001 * torch.mean(dx ** 2)

                    d_loss.backward()
                    d_optimizer.step()

                    #############################
                    ####   Train Generator   ####
                    #############################

                    g_net.zero_grad()

                    fake_batch = g_net(noise, p, alpha)
                    dgz_2 = d_net(fake_batch, p, alpha).view(-1)
                    g_loss = -torch.mean(dgz_2)

                    g_loss.backward()
                    g_optimizer.step()

                    #############################

                    # update alpha
                    alpha += 2 / (self.args.n * len(self.dataloaders[p]))
                    alpha = min(alpha, 1)

                    #############################
                    ####   Metrics Tracking  ####
                    #############################

                    if i % 1000 == 0:
                        print(f'[%d/%d][%d/%d]\td_loss: %.4f\tg_loss %.4f\talpha: %.4f'
                            % (epoch, self.args.n, i, len(self.dataloaders[p]),
                            d_loss.item(), g_loss.item(), alpha))

                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())

                if epoch % 2 == 0 or epoch == self.args.n-1:
                    with torch.no_grad():
                        g_net.eval()
                        fake = g_net(fixed_latent, p, 1).detach()
                        g_net.train()
                    plot_batch(fake, self.progress_dir + f"res_{resolution}/epoch_{epoch}")

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
    
class WSConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
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
    
class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B) if B % self.group_size == 0 else 1
        
        y = x.view(G, -1, C, H, W)
        
        # compute std dev
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(torch.square(y), dim=0)
        y = torch.sqrt(y + 1e-8)
        
        # average over all other dimensions
        y = torch.mean(y, dim=[1,2,3], keepdim=True)
        
        # repeat value for feature map
        y = y.repeat(G, 1, H, W)
        
        # add to input
        return torch.cat([x,y], dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()

        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.pn = PixelNorm() if use_pixelnorm else nn.Identity()

    def forward(self, x):
        x = F.leaky_relu(self.pn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.pn(self.conv2(x)), 0.2)
        return x

####   Generator   #####

class Generator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 1, 2, 4, 4]):
        super(Generator, self).__init__()
        self.args = args

        hidden_dims = [int(args.dim / mult) for mult in dim_mults]

        self.embed = nn.Sequential(
            WSConvTranspose2d(args.latent, args.dim, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            WSConv2d(args.dim, args.dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.progressive_blocks = nn.ModuleList([
            *[
                ConvBlock(in_f, out_f) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        ])

        self.out_blocks = nn.ModuleList([
            *[
                WSConv2d(out_f, args.channel_size, 3, 1, 1) for out_f in hidden_dims
            ],
        ])

    def fade(self, lower, higher, alpha):
        return alpha * higher + (1 - alpha) * lower

    def forward(self, x, p, alpha):
        out = self.embed(x)
        
        for i in range(p):
            out_lower = out
            out = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.progressive_blocks[i](out)
            
        if p > 0:
            out_lower = self.out_blocks[p-1](out_lower)
            out_lower = F.interpolate(out_lower, scale_factor=2, mode="bilinear")
            out = self.out_blocks[p](out)
            final_out = self.fade(out_lower, out, alpha)
        else:
            final_out = self.out_blocks[p](out)
        
        return F.tanh(final_out)
    
#########################
    
##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 1, 2, 4, 4]):
        super(Discriminator, self).__init__()
        self.args = args

        hidden_dims = [int(args.dim / mult) for mult in reversed(dim_mults)]

        self.progressive_blocks = nn.ModuleList([
            *[
                ConvBlock(in_f, out_f, use_pixelnorm=False) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        ])

        self.in_blocks = nn.ModuleList([
            *[
                WSConv2d(args.channel_size, in_f, 1, 1, 0) for in_f in hidden_dims
            ],
        ])

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.map = nn.Sequential(
            WSConv2d(args.dim + 1, args.dim, 3, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(args.dim, args.latent, 4),
            nn.LeakyReLU(0.2),
        )

        self.linear = WSConv2d(args.latent, 1)

        self.minibatch_stddev = MiniBatchStdDev()

    def fade(self, lower, higher, alpha):
        return alpha * higher + (1 - alpha) * lower
    
    def forward(self, x, p, alpha):

        for i in range(p, -1, -1):
            rev_p = len(self.progressive_blocks) - i

            if i == p:
                out = self.in_blocks[rev_p](x)

            if i == 0:
                out = self.minibatch_stddev(out)
                out = self.map(out)
            else:
                out = self.progressive_blocks[rev_p](out)

            if i > 0:
                out = self.downsample(out)

                if i == p and alpha < 1:
                    downsampled = self.in_blocks[rev_p + 1](self.downsample(x))
                    out = self.fade(downsampled, out, alpha)

        out = self.linear(out)
        return out
    
#########################
