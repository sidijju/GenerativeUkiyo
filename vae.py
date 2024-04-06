import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
from datetime import datetime

##### VAE #####

class VAE:

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
            self.run_dir = "train/vae-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
            self.progress_dir = self.run_dir + "progress/"
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            if not os.path.exists(self.progress_dir):
                os.makedirs(self.progress_dir)
        
    def train(self, 
            num_epochs = 50,
            lr = .0002):

        assert self.args.train

        if not self.dataloader:
            return
        
        vae = VariationalAutoEncoder(self.args, self.channel_size)
        vae.apply(self.weights_init)
        vae.to(self.args.device)
        optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))

        fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.args.device)
        bce = nn.BCELoss(reduction='sum')

        images = []
        losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(num_epochs)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)
                batchsize = batch.shape[0]

                if iters == 0:
                    images.append(vutils.make_grid(batch.cpu()[:25], nrow = 5, padding=2, normalize=True))
                    plt.axis('off')
                    plt.imshow(images[-1].permute(1, 2, 0))
                    plt.savefig(self.progress_dir + f"train_example")

                vae.zero_grad()
                batch_hat, mu, logvar = vae(batch)

                # reproduction loss
                reproduction_loss = bce(batch_hat, batch)

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = reproduction_loss + kl_loss
                loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tr_loss: %.4f\tkl_loss: %.4f\tloss: %.4f'
                        % (epoch, num_epochs, i, len(self.dataloader),
                            reproduction_loss.item() / batchsize, kl_loss.item(), loss.item()))

                losses.append(loss.item())

                if (iters % 5000 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):

                    with torch.no_grad():
                        fake = vae.decode(fixed_latent).detach().cpu()
                    images.append(vutils.make_grid(fake[:25], nrow = 5, padding=2, normalize=True))
                    plt.axis('off')
                    plt.imshow(images[-1].permute(1, 2, 0))
                    plt.savefig(self.progress_dir + f"iter:{iters}")

                iters += 1

        print("### End Training Procedure ###")
        self.save_train_data(losses, vae)

    def save_train_data(self, losses, vae):

        # save models
        torch.save(vae.state_dict(), self.run_dir + '/vae.pt')

        # save losses
        plt.figure(figsize=(10,5))
        plt.title("Training Losses")
        plt.plot(losses,label="L")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")
                
    def generate(self, path, n = 5):
        vae = VariationalAutoEncoder(self.args, self.channel_size)
        vae.load_state_dict(torch.load(path + "/vae.pt"))
        vae.to(self.args.device)
        vae.eval()

        noise = torch.randn(n, self.latent_size, 1, 1, device=self.args.device)

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake_batch = vae.decode(noise)

        for i in range(n):
            plt.imshow(batch[i].cpu().permute(1, 2, 0))
            plt.savefig(path + f"/r_{i}")
            plt.imshow(fake_batch[i].cpu().permute(1, 2, 0))
            plt.savefig(path + f"/f_{i}")

    # utility function to iterate through model
    # and initalize weights in layers rom N(0, 0.02)
    def weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
###############

class VariationalAutoEncoder(nn.Module):
    def __init__(self, args, channel_size, h_dim=2048):
        super(VariationalAutoEncoder, self).__init__()
        self.args = args
        self.latent_size = args.latent

        nf = 16

        self.encoder = nn.Sequential(
            self.conv_block(channel_size, nf),
            self.conv_block(nf, nf * 2),
            self.conv_block(nf * 2, nf * 4),
            self.conv_block(nf * 4, nf * 4),
            self.conv_block(nf * 4, nf * 8),
            self.conv_block(nf * 8, nf * 16),
            self.conv_block(nf * 16, nf * 32),
            nn.Flatten()
        )

        self.mu = nn.Linear(h_dim, self.latent_size)
        self.logvar = nn.Linear(h_dim, self.latent_size)
        self.embed = nn.Linear(self.latent_size, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (h_dim, 1, 1)),
            self.conv_transpose_block(h_dim, nf * 32),
            self.conv_transpose_block(nf * 32, nf * 16),
            self.conv_transpose_block(nf * 16, nf * 8),
            self.conv_transpose_block(nf * 8, nf * 4),
            self.conv_transpose_block(nf * 4, nf * 4),
            self.conv_transpose_block(nf * 4, nf * 2),
            self.conv_transpose_block(nf * 2, nf),
            nn.ConvTranspose2d(nf, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return mu + std * torch.randn(*mu.size()).to(self.args.device)

    def conv_block(self, input, output, kernel=4, stride=2, pad=1):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )
    
    def conv_transpose_block(self, input, output, kernel=4, stride=2, pad=1):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(True),
        )
    
    def encode(self, input):
        embed = self.encoder(input)
        mu, logvar = self.mu(embed), self.logvar(embed)
        sample = self.reparameterize(mu, logvar)
        return sample, mu, logvar
    
    def decode(self, input):
        embed = self.embed(input.squeeze())
        return self.decoder(embed)
    
    def forward(self, input):
        sample, mu, logvar = self.encode(input)
        out = self.decode(sample)
        return out, mu, logvar
    
#########################
