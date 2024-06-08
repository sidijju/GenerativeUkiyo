import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from utils import *

##### VAE #####

class VAE:

    def __init__(self, 
                args,
                dataloader = None,
                workers = 2,
                ):
        
        self.args = args
        self.dataloader = dataloader
        self.workers = workers
        self.channel_size = args.channel_size
        self.latent_size = args.latent
        self.beta = args.beta

        if not self.args.test:
            self.run_dir = "train/vae-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)
        
    def train(self):

        if not self.dataloader:
            return
        
        vae = VariationalAutoEncoder(self.args, self.channel_size)
        vae.apply(weights_init)
        vae.to(self.args.device)
        optimizer = optim.Adam(vae.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        fixed_latent = torch.randn(64, self.latent_size, 1, 1, device=self.args.device)

        loss_fn = nn.MSELoss() if self.args.mse else nn.BCELoss()

        losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                if iters == 0:
                    plot_batch(batch, self.progress_dir + f"train_example")

                vae.zero_grad()
                batch_hat, mu, logvar = vae(batch)

                # reproduction loss
                reproduction_loss = loss_fn(batch_hat, batch)

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = reproduction_loss + self.beta * kl_loss
                loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tr_loss: %.4f\tkl_loss: %.4f\tloss: %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader),
                            reproduction_loss.item(), kl_loss.item(), loss.item()))

                losses.append(loss.item())

                if (iters % 5000 == 0) or ((epoch == self.args.n-1) and (i == len(self.dataloader)-1)):

                    plot_batch(batch, self.progress_dir + f"input-iter:{iters}")
                    plot_batch(batch_hat, self.progress_dir + f"output-iter:{iters}")

                    with torch.no_grad():
                        fake = vae.decode(fixed_latent).detach().cpu()
                    plot_batch(fake, self.progress_dir + f"gen-iter:{iters}")

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
        print("### Begin Generating Images ###")
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
            plot_image(batch[i], path + f"/r_{i}")
            plot_image(fake_batch[i], path + f"/f_{i}")
        print("### Done Generating Images ###")

###############

class VariationalAutoEncoder(nn.Module):
    def __init__(self, args, channel_size, dim_mults = (1, 2, 4, 8, 16)):
        super(VariationalAutoEncoder, self).__init__()
        self.args = args
        self.latent_size = args.latent

        nf = 64
        hidden_dims = [nf * mult for mult in list(dim_mults)]
        self.d_max = hidden_dims[-1]

        self.encoder = nn.Sequential(
            *[
                self.conv_block(in_f, out_f)
                for in_f, out_f in zip([channel_size] + hidden_dims[:-1], hidden_dims)
            ]
        )

        self.mu = nn.Linear(self.d_max * 4 * 4, self.latent_size)
        self.logvar = nn.Linear(self.d_max * 4 * 4, self.latent_size)
        self.embed = nn.Linear(self.latent_size, self.d_max * 4 * 4)

        self.decoder = nn.Sequential(
            *[
                self.conv_transpose_block(in_f, out_f)
                for in_f, out_f in zip(reversed(hidden_dims[1:]), reversed(hidden_dims[:-1]))
            ],
            nn.ConvTranspose2d(nf, channel_size, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.args.device)
        return mu + std * eps

    def conv_block(self, input, output, kernel=4, stride=2, pad=1):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(),
        )
    
    def conv_transpose_block(self, input, output, kernel=4, stride=2, pad=1):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(),
        )
    
    def encode(self, input):
        embed = self.encoder(input)
        embed = torch.flatten(embed, start_dim=1)
        mu, logvar = self.mu(embed), self.logvar(embed)
        sample = self.reparameterize(mu, logvar)
        return sample, mu, logvar
    
    def decode(self, input):
        embed = self.embed(input.squeeze())
        embed = embed.view(-1, self.d_max, 4, 4)
        return self.decoder(embed)
    
    def forward(self, input):
        sample, mu, logvar = self.encode(input)
        out = self.decode(sample)
        return out, mu, logvar
    
#########################
