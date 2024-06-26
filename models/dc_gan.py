import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.optim as optim
from tqdm import tqdm
from models.gan import GAN
from utils import *

##### DCGAN #####

class DCGAN(GAN):

    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        g_net = Generator(self.args, self.channel_size, self.latent_size)
        g_net.load_state_dict(torch.load(path + "/generator.pt"))
        g_net.to(self.args.device)
        g_net.eval()

        noise = torch.randn(n, self.latent_size, 1, 1, device=self.args.device)

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake = g_net(noise)

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            plot_image(fake[i], path + f"/f_{i}")
        print("### Done Generating Images ###")
        
    def train(self):
        
        d_lr = self.args.lr
        g_lr = self.args.lr
        
        d_net = Discriminator(self.args, self.channel_size)
        if self.args.checkpoint_d:
            d_net.load_state_dict(torch.load(self.args.checkpoint_d, map_location=self.args.device))
            print("Loaded discriminator checkpoint from", self.args.checkpoint_d)
        else:
            d_net.apply(weights_init)
        d_net.to(self.args.device)
        d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.5, 0.999))


        g_net = Generator(self.args, self.channel_size, self.latent_size)
        if self.args.checkpoint_g:
            g_net.load_state_dict(torch.load(self.args.checkpoint_g, map_location=self.args.device))
            print("Loaded generator checkpoint from", self.args.checkpoint_g)
        else:
            g_net.apply(weights_init)
        g_net.to(self.args.device)
        g_optimizer = optim.Adam(g_net.parameters(), lr=g_lr, betas=(0.5, 0.999))

        bce = nn.BCELoss()

        fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.args.device)

        # One-sided label smoothing from "Improved Techniques for Training GANs"
        real_label = 0.9
        fake_label = 0.

        d_losses_real = []
        d_losses_fake = []
        g_losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)
                batchsize = batch.shape[0]

                if iters == 0:
                    plot_batch(batch, self.progress_dir + f"train_example")

                # generate fake batch for training
                noise = torch.randn(batchsize, self.latent_size, 1, 1, device=self.args.device)
                fake_batch = g_net(noise)

                real_labels = torch.full((batchsize,), real_label, dtype=torch.float, device=self.args.device)
                fake_labels = torch.full((batchsize,), fake_label, dtype=torch.float, device=self.args.device)

                #############################
                #### Train Discriminator ####
                #############################

                d_net.zero_grad()

                # loss on real inputs
                output = d_net(batch).view(-1)
                d_loss_real = bce(output, real_labels)
                d_loss_real.backward()
                # D(x)
                dx = output.mean().item()

                # loss on fake inputs
                output = d_net(fake_batch.detach()).view(-1)
                d_loss_fake = bce(output, fake_labels)
                d_loss_fake.backward()
                # D(G(z))
                dgz_1 = output.mean().item()

                d_optimizer.step()

                #############################
                ####   Train Generator   ####
                #############################

                g_net.zero_grad()
                output = d_net(fake_batch).view(-1)

                g_loss = bce(output, real_labels)
                g_loss.backward()
                # D(G(z))
                dgz_2 = output.mean().item()

                g_optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\td_loss: %.4f\tg_loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader),
                            d_loss_real.item() + d_loss_fake.item(), g_loss.item(), dx, dgz_1, dgz_2))

                d_losses_real.append(d_loss_real.item())
                d_losses_fake.append(d_loss_fake.item())
                g_losses.append(g_loss.item())

                if (iters % 5000 == 0) or ((epoch == self.args.n-1) and (i == len(self.dataloader)-1)):

                    with torch.no_grad():
                        g_net.eval()
                        fake = g_net(fixed_latent).detach()
                        g_net.train()

                    plot_batch(fake, self.progress_dir + f"iter:{iters}")

                iters += 1

        print("### End Training Procedure ###")
        self.save_train_data(d_losses_real, d_losses_fake, g_losses, d_net, g_net)
                
###############
            
####   Generator   #####

class Generator(nn.Module):
    def __init__(self, args, channel_size, latent_size, dim_mults = (1, 2, 4, 8, 16)):
        super(Generator, self).__init__()
        self.args = args

        ngf = 64
        hidden_dims = [ngf * mult for mult in reversed(list(dim_mults))]

        self.model = nn.Sequential(
            self.conv_block(latent_size, hidden_dims[0], 4, stride=1, pad=0),
            *[
                self.conv_block(in_f, out_f, 4)
                for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
            P.spectral_norm(nn.ConvTranspose2d(ngf, channel_size, 4, 2, 1, bias=False)),
            nn.Sigmoid()
        )

    def conv_block(self, input, output, kernel, stride=2, pad=1):
        return nn.Sequential(
            P.spectral_norm(nn.ConvTranspose2d(input, output, kernel, stride, pad, bias=False)),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.model(input)
    
#########################
    
##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, args, channel_size, dim_mults = (1, 2, 4, 8, 16)):
        super(Discriminator, self).__init__()
        self.args = args

        ndf = 64
        hidden_dims = [ndf * mult for mult in list(dim_mults)]

        self.model= nn.Sequential(
            P.spectral_norm(nn.Conv2d(channel_size, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            *[
                self.conv_block(in_f, out_f, 4)
                for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ],
            P.spectral_norm(nn.Conv2d(hidden_dims[-1], 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def conv_block(self, input, output, kernel):
        return nn.Sequential(
            P.spectral_norm(nn.Conv2d(input, output, kernel, 2, 1, bias=False)),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, input):
        return self.model(input)
    
#########################
