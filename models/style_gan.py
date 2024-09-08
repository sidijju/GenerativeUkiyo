import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models.pro_gan import ProGAN, WSConv2d, MiniBatchStdDev, PixelNorm, ConvBlock
from utils import *

##### StyleGAN #####

class StyleGAN(ProGAN):

    def __init__(self, 
                args
                ):
        self.args = args

        if not self.args.test:
            self.run_dir = f"train/stylegan_n={self.args.n}_lr={self.args.lr}_dim={self.args.dim}/"
            self.progress_dir = self.run_dir + "progress/"

            make_dir(self.run_dir)
            make_dir(self.progress_dir)
            save_conf(self.args, self.run_dir)

        self.resolutions = [8 * (2 ** i) for i in range(args.dim.bit_length() - 3)]
        self.dataloaders = [self.get_dataloader(res) for res in self.resolutions]

    def generate(self, path, n=5):
        print("### Begin Generating Images ###")
        g_net = Generator(self.args)
        g_net.load_state_dict(torch.load(path + "/generator.pt", map_location=torch.device(self.args.device)))
        g_net.to(self.args.device)
        g_net.eval()

        noise = torch.randn(n, self.args.latent, device=self.args.device)
        batch, _ = next(iter(self.dataloaders[-1]))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake = g_net(noise, len(self.resolutions)-1, 1)

        for i in range(n):
            plot_image((batch[i] + 1)/2, path + f"/r_{i}")
            plot_image((fake[i] + 1)/2, path + f"/f_{i}")
        print("### Done Generating Images ###")

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

        map_params = g_net.map.parameters()
        base_params = list(g_net.init_block.parameters())
        base_params += list(g_net.progressive_blocks.parameters())
        base_params += list(g_net.out_blocks.parameters())
        base_params += [g_net.starting_constant]

        g_optimizer = optim.Adam([
            {'params': base_params},
            {'params': map_params, 'lr': self.args.lr * .01}
            ], lr=self.args.lr, betas=(0.5, 0.999))

        d_losses = []
        g_losses = []

        fixed_latent = torch.randn(self.args.batchsize, self.args.latent, device=self.args.device)

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
                    noise = torch.randn(batch.shape[0], self.args.latent, device=self.args.device)

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

class WSLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
            
####   Generator   #####

class NoiseMappingNetwork(nn.Module):
    def __init__(self, args):
        super(NoiseMappingNetwork, self).__init__()
        self.args = args 

        self.model = nn.Sequential(
            PixelNorm(),
            WSLinear(args.latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent),
            nn.ReLU(),
            WSLinear(args.w_latent, args.w_latent)
        )
    def forward(self, x):
        return self.model(x)

class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, in_channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels)

        self.style_sigma = WSLinear(w_dim, in_channels)
        self.style_mu = WSLinear(w_dim, in_channels)
                                   
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_sigma = self.style_sigma(w).unsqueeze(2).unsqueeze(3)
        style_mu = self.style_mu(w).unsqueeze(2).unsqueeze(3)
        return style_sigma * x + style_mu

class NoiseInput(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GeneratorBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.noise1 = NoiseInput(out_channels)
        self.noise2 = NoiseInput(out_channels)
        self.adain1 = AdaptiveInstanceNormalization(out_channels, w_dim)
        self.adain2 = AdaptiveInstanceNormalization(out_channels, w_dim)

    def forward(self, x, w):
        x = self.adain1(F.leaky_relu(self.noise1(self.conv1(x)), 0.2), w)
        x = self.adain2(F.leaky_relu(self.noise2(self.conv2(x)), 0.2), w)
        return x
    
class InitGeneratorBlock(nn.Module):
    def __init__(self, in_channels, w_dim):
        super(InitGeneratorBlock, self).__init__()
        
        self.conv = WSConv2d(in_channels, in_channels)
        self.noise1 = NoiseInput(in_channels)
        self.noise2 = NoiseInput(in_channels)
        self.adain1 = AdaptiveInstanceNormalization(in_channels, w_dim)
        self.adain2 = AdaptiveInstanceNormalization(in_channels, w_dim)

    def forward(self, x, w):
        x = self.adain1(F.leaky_relu(self.noise1(x)), w)
        x = self.adain2(F.leaky_relu(self.noise2(self.conv(x))), w)
        return x

class Generator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 2, 4, 4]):
        super(Generator, self).__init__()
        self.args = args

        self.map = NoiseMappingNetwork(args)

        self.starting_constant = nn.Parameter(torch.ones((1, args.latent, 8, 8)))
        self.init_block = InitGeneratorBlock(args.latent, args.w_latent)

        hidden_dims = [int(args.latent / mult) for mult in dim_mults]

        self.progressive_blocks = nn.ModuleList([
            *[
                GeneratorBlock(in_f, out_f, args.w_latent) for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
        ])

        self.out_blocks = nn.ModuleList([
            *[
                WSConv2d(out_f, args.channel_size, 1, 1, 0) for out_f in hidden_dims
            ],
        ])

    def fade(self, lower, higher, alpha):
        return alpha * higher + (1 - alpha) * lower

    def forward(self, z, p, alpha):
        w = self.map(F.normalize(z, dim=1))
        out = self.init_block(self.starting_constant, w)

        if p == 0:
            return self.out_blocks[0](out)
        
        for i in range(p):
            upsampled = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.progressive_blocks[i](upsampled, w)
            
        final_upsampled = self.out_blocks[p-1](upsampled) 
        final_out = self.out_blocks[p](out)

        return F.tanh(self.fade(final_upsampled, final_out, alpha))
    
#########################
    
##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, args, dim_mults = [1, 1, 1, 2, 4, 4]):
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
                out = F.interpolate(out, scale_factor=0.5, mode="bilinear")

                if i == p and alpha < 1:
                    downsampled = F.interpolate(x, scale_factor=0.5, mode="bilinear")
                    downsampled = self.in_blocks[rev_p + 1](downsampled)
                    out = self.fade(downsampled, out, alpha)
        
        out = self.linear(out)
        return out
    
#########################


