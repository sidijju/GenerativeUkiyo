##### Resources #####
# https://www.nichibun.ac.jp/en/db/category/yokaigazou/
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://arxiv.org/abs/1606.03498
#####################
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from utils import *

##### DCGAN #####

class DCGAN:

    def __init__(self, 
                args,
                dataloader = None,
                workers = 2,
                channel_size = 3,
                ):
        
        self.args = args
        self.dataloader = dataloader
        self.workers = workers
        self.channel_size = channel_size
        self.latent_size = args.latent

        if self.args.train:
            self.run_dir = "train/gan-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)
        
    def train(self, 
            num_epochs = 5,
            g_lr = .0001,
            d_lr = .0004):

        assert self.args.train and self.dataloader
        
        d_net = Discriminator(self.args, self.channel_size)
        d_net.apply(weights_init)
        d_net.to(self.args.device)
        d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.5, 0.999))
        if self.args.fm:
            d_net.model.register_forward_hook(d_net.feature_activations)

        g_net = Generator(self.args, self.channel_size, self.latent_size)
        g_net.apply(weights_init)
        g_net.to(self.args.device)
        g_optimizer = optim.Adam(g_net.parameters(), lr=g_lr, betas=(0.5, 0.999))

        mse = nn.MSELoss()
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
        for epoch in tqdm(range(num_epochs)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, labels = batch
                batch = batch.to(self.args.device)
                labels = labels.to(self.args.device)
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
                if self.args.fm:
                    fx = d_net.features.detach().clone()
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

                if self.args.fm:
                    # Feature matching objective from "Improved Techniques for Training GANs"
                    e_fx = fx.mean(dim=0)
                    fgz = d_net.features
                    e_fgz = fgz.mean(dim=0)
                    g_loss = torch.square(mse(e_fx, e_fgz))
                else:
                    # Flip label option for better gradient flow
                    if self.args.flip:
                        g_loss = bce(output, fake_labels)
                    else:
                        g_loss = bce(output, real_labels)
                
                # to check for vanishing gradients or exploding gradients
                g_loss.backward()
                # D(G(z))
                dgz_2 = output.mean().item()

                g_optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\td_loss: %.4f\tg_loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(self.dataloader),
                            (d_loss_real.item() + d_loss_fake.item())/2, g_loss.item(), dx, dgz_1, dgz_2))

                d_losses_real.append(d_loss_real.item())
                d_losses_fake.append(d_loss_fake.item())
                g_losses.append(g_loss.item())

                if (iters % 5000 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):

                    with torch.no_grad():
                        fake = g_net(fixed_latent).detach()
                    plot_batch(fake, self.progress_dir + f"iter:{iters}")

                iters += 1

        print("### End Training Procedure ###")
        self.save_train_data(d_losses_real, d_losses_fake, g_losses, d_net, g_net)

    def save_train_data(self, d_losses_real, d_losses_fake, g_losses, d_net, g_net):

        # save models
        torch.save(d_net.state_dict(), self.run_dir + '/discriminator.pt')
        torch.save(g_net.state_dict(), self.run_dir + '/generator.pt')

        # save losses
        plt.figure(figsize=(10,5))
        plt.title("Training Losses")
        plt.plot(g_losses,label="G")
        plt.plot([sum(x)/2 for x in zip(d_losses_real, d_losses_fake)], label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")
                
    def generate(self, path, n = 5):
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

###############
            
####   Generator   #####

class Generator(nn.Module):
    def __init__(self, args, channel_size, latent_size):
        super(Generator, self).__init__()
        self.args = args

        ngf = 64

        self.model = nn.Sequential(
            self.conv_block(latent_size, ngf * 16, 4, stride=1, pad=0),
            #self.conv_block(ngf * 32, ngf * 16, 4),
            self.conv_block(ngf * 16, ngf * 8, 4),
            self.conv_block(ngf * 8, ngf * 4, 4),
            self.conv_block(ngf * 4, ngf * 2, 4),
            self.conv_block(ngf * 2, ngf * 1, 4),
            P.spectral_norm(nn.ConvTranspose2d(ngf, channel_size, 4, 2, 1, bias=False)),
            nn.Tanh()
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
    def __init__(self, args, channel_size):
        super(Discriminator, self).__init__()
        self.args = args

        ndf = 64

        self.model = nn.Sequential(
            self.conv_block(channel_size, ndf, 4, batchnorm=False),
            self.conv_block(ndf, ndf * 2, 4),
            self.conv_block(ndf * 2, ndf * 4, 4),
            self.conv_block(ndf * 4, ndf * 8, 4),
            self.conv_block(ndf * 8, ndf * 16, 4),
            #self.conv_block(ndf * 16, ndf * 32, 4),
        )

        self.conv = P.spectral_norm(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False))
        self.sig = nn.Sigmoid()

    def conv_block(self, input, output, kernel, batchnorm=True):
        if batchnorm:
            return nn.Sequential(
                P.spectral_norm(nn.Conv2d(input, output, kernel, 2, 1, bias=False)),
                nn.BatchNorm2d(output),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                P.spectral_norm(nn.Conv2d(input, output, kernel, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def feature_activations(self, model, input, output):
        self.features = output
    
    def forward(self, input):
        return self.sig(self.conv(self.model(input)))
    
#########################
