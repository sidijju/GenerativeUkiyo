import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
from models.gan import GAN
from utils import *

##### CDCGAN #####

class CDCGAN(GAN):
        
    def train(self, 
            num_epochs = 5,
            lr = .0002):

        if not self.dataloader:
            return 
            
        d_net = Discriminator(self.args, self.channel_size)
        d_net.apply(weights_init)
        d_net.to(self.args.device)
        d_optimizer = optim.Adam(d_net.parameters(), lr=lr, betas=(0.5, 0.999))
        if self.args.fm:
            d_net.l1.register_forward_hook(d_net.feature_activations)

        g_net = Generator(self.args, self.channel_size, self.latent_size)
        g_net.apply(weights_init)
        g_net.to(self.args.device)
        g_optimizer = optim.Adam(g_net.parameters(), lr=lr, betas=(0.5, 0.999))

        mse = nn.MSELoss()
        bce = nn.BCELoss()

        fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.args.device)

        # One-sided label smoothing from "Improved Techniques for Training GANs"
        real_label = 0.9
        fake_label = 0.

        images = []
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
                fake_batch = g_net(noise, labels)

                real_labels = torch.full((batchsize,), real_label, dtype=torch.float, device=self.args.device)
                fake_labels = torch.full((batchsize,), fake_label, dtype=torch.float, device=self.args.device)

                #############################
                #### Train Discriminator ####
                #############################

                d_net.zero_grad()

                # loss on real inputs
                output = d_net(batch, labels).view(-1)
                if self.args.fm:
                    fx = d_net.features.detach().clone()
                d_loss_real = bce(output, real_labels)
                d_loss_real.backward()
                # D(x)
                dx = output.mean().item()

                # loss on fake inputs
                output = d_net(fake_batch.detach(), labels).view(-1)
                d_loss_fake = bce(output, fake_labels)
                d_loss_fake.backward()
                # D(G(z))
                dgz_1 = output.mean().item()

                d_optimizer.step()

                #############################
                ####   Train Generator   ####
                #############################

                g_net.zero_grad()
                output = d_net(fake_batch, labels).view(-1)

                if self.args.fm:
                    # Feature matching objective from "Improved Techniques for Training GANs"
                    e_fx = fx.mean(dim=0)
                    fgz = d_net.features
                    e_fgz = fgz.mean(dim=0)
                    g_loss = torch.square(mse(e_fx, e_fgz))
                else:
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
                        % (epoch, num_epochs, i, len(self.dataloader),
                            d_loss_real.item() + d_loss_fake.item(), g_loss.item(), dx, dgz_1, dgz_2))

                d_losses_real.append(d_loss_real.item())
                d_losses_fake.append(d_loss_fake.item())
                g_losses.append(g_loss.item())

                if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = g_net(fixed_latent, labels).detach().cpu()
                    images.append(vutils.make_grid(fake, padding=2, normalize=True))
                    plt.axis('off')
                    plt.imshow(images[-1].permute(1, 2, 0))
                    plt.savefig(self.progress_dir + f"iter:{iters}")

                iters += 1
        print("### End Training Procedure ###")
        self.save_train_data(d_losses_real, d_losses_fake, g_losses, d_net, g_net)
                
    def generate(self, n = 5):
        pass

###############
            
#####   Generator   #####

class Generator(nn.Module):
    def __init__(self, args, channel_size, latent_size):
        super(Generator, self).__init__()
        self.args = args

        ngf = 64

        self.embed = self.conv_block(latent_size, 1024, 4, stride=1, pad=0)
        self.label_embedding = self.conv_block(self.args.num_classes, 1024, 4, stride=1, pad=0)

        self.model = nn.Sequential(
            self.conv_block(ngf * 32, ngf * 16, 4),
            self.conv_block(ngf * 16, ngf * 8, 4),
            self.conv_block(ngf * 8, ngf * 4, 4),
            self.conv_block(ngf * 4, ngf * 2, 4),
            nn.ConvTranspose2d(ngf * 2, channel_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def conv_block(self, input, output, kernel, stride=2, pad=1):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )

    def forward(self, input, label):
        input_embed = self.embed(input)
        one_hot_label = nn.functional.one_hot(label, num_classes=self.args.num_classes)
        one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1).float()
        label_embed = self.label_embedding(one_hot_label)
        label_input = torch.cat((input_embed, label_embed), dim=1)
        return self.model(label_input)
    
#########################
    
##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, args, channel_size):
        super(Discriminator, self).__init__()
        self.args = args

        ndf = 64

        self.label_embedding = nn.Sequential(
            nn.Linear(self.args.num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 128 * 128),
            nn.Unflatten(1, (1, 128, 128)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.l1 = nn.Sequential(
            self.conv_block(channel_size + 1, ndf, 4, batchnorm=False),
            self.conv_block(ndf, ndf * 2, 4),
        )
        self.l2 = nn.Sequential(
            self.conv_block(ndf * 2, ndf * 4, 4),
            self.conv_block(ndf * 4, ndf * 8, 4),
            self.conv_block(ndf * 8, ndf * 16, 4),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def conv_block(self, input, output, kernel, batchnorm=True):
        if batchnorm:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input, output, kernel, 2, 1, bias=False)),
                nn.BatchNorm2d(output),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input, output, kernel, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def feature_activations(self, model, input, output):
        self.features = output
    
    def forward(self, input, label):
        one_hot_label = nn.functional.one_hot(label, num_classes=self.args.num_classes)
        one_hot_label = one_hot_label.float()
        label_channel = self.label_embedding(one_hot_label)
        label_input = torch.cat((label_channel, input), dim=1)
        l1 = self.l1(label_input)
        return self.l2(l1)
    
#########################
