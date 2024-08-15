import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import savgol_filter

from models.pixel_cnn import PixelCNN

from utils import *

##### VQVAE #####

class VQVAE:

    def __init__(self, 
                args,
                dataloader = None,
                ):
        
        self.args = args
        self.dataloader = dataloader

        if not self.args.test:
            self.run_dir = f"train/vqvae/"
            self.progress_dir = self.run_dir + "progress/"
            self.prior_dir = self.run_dir + "prior_progress/"

            make_dir(self.run_dir)
            make_dir(self.progress_dir)
            make_dir(self.prior_dir)

            save_conf(self.args, self.run_dir)
        
    def train(self):

        print("### Begin Training Procedure ###")
        print("Training VQ-VAE")
        losses, vq_vae = self.train_vae()
        print("Training PixelCNN prior")
        pixel_cnn_losses, pixel_cnn = self.train_prior(vq_vae=vq_vae)
        print("### End Training Procedure ###")
        
        self.save_train_data(losses, pixel_cnn_losses, vq_vae, pixel_cnn)

    def train_vae(self):
        sample_batch, _ = next(iter(self.dataloader))
        sample_batch = sample_batch.to(self.args.device)
        plot_batch(sample_batch, self.progress_dir + f"train_example")

        vq_vae = VectorQuantizedVariationalAutoEncoder(self.args)
        if self.args.checkpoint:
            vq_vae.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
            print("Loaded checkpoint from", self.args.checkpoint)
        vq_vae.to(self.args.device)
        optimizer = optim.Adam(vq_vae.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        losses = []

        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                vq_vae.zero_grad()

                batch_hat, z_e, z_q = vq_vae(batch)

                reproduction_loss = F.mse_loss(batch_hat, batch)
                dictionary_loss = F.mse_loss(z_e.detach(), z_q)
                commitment_loss = F.mse_loss(z_q.detach(), z_e)

                loss = reproduction_loss + dictionary_loss + self.args.beta * commitment_loss

                loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                losses.append((loss.item(), 
                               reproduction_loss.item(), 
                               dictionary_loss.item(),
                               commitment_loss.item()))

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f\tr_loss: %.4f\tdictionary_loss: %.4f\t c_loss: %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader),
                            loss.item(), reproduction_loss.item(), dictionary_loss.item(), commitment_loss.item()))
                    
            with torch.no_grad():
                vq_vae.eval()
                sample_batch_hat, _, _ = vq_vae(sample_batch)
                vq_vae.train()
            plot_compare_batch(sample_batch, sample_batch_hat, self.progress_dir + f"comp-epoch:{epoch}")
        
        return losses, vq_vae

    def train_prior(self, vq_vae):  
        prior = PixelCNN()
        if self.args.checkpoint_prior:
            prior.load_state_dict(torch.load(self.args.checkpoint_prior, map_location=self.args.device))
            print("Loaded prior checkpoint from", self.args.checkpoint_prior)
        prior.to(self.args.device)   
        prior_optimizer = optim.Adam(prior.parameters(), lr=self.args.prior_lr, betas=(0.5, 0.999))

        pixelcnn_losses = []

        for epoch in tqdm(range(self.args.prior_n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                with torch.no_grad():
                    latents = vq_vae.encode(batch)
                    latents = latents.detach()

                prior.zero_grad()
                logits = prior(latents)
                logits = logits.permute(0, 2, 3, 1).contiguous()

                loss = F.cross_entropy(logits.view(-1, self.args.k), latents.view(-1))
                loss.backward()
                prior_optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                pixelcnn_losses.append((loss.item()))

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.prior_n, i, len(self.dataloader), loss.item()))
                    
            sample_prior_batch, sample_uniform_prior_batch = self.sample(16, vq_vae, prior)
            plot_batch(sample_prior_batch, self.prior_dir + f"gen-epoch:{epoch}")
            plot_batch(sample_uniform_prior_batch, self.prior_dir + f"uniform-gen-epoch:{epoch}")

        return pixelcnn_losses, prior

    def save_train_data(self, losses, pixelcnn_losses, vq_vae, pixel_cnn):

        totals = [loss[0] for loss in losses]
        reconstructions = [loss[1] for loss in losses]
        dictionary_losses = [loss[2] for loss in losses]
        commitment_losses = [loss[3] for loss in losses]

        if self.args.n > 0:
            filtered_totals = savgol_filter(totals, 51, 2)
            filtered_recs = savgol_filter(reconstructions, 51, 2)
            filtered_dics = savgol_filter(dictionary_losses, 51, 2)
            filtered_cs = savgol_filter(commitment_losses, 51, 2)

            # save losses
            plt.cla()
            plt.yscale('log')
            plt.figure(figsize=(10,5))
            plt.title("Training Loss")
            plt.plot(filtered_totals,label="total")
            plt.plot(filtered_recs,label="reconstruction")
            plt.plot(filtered_dics,label="vq")
            plt.plot(filtered_cs,label="commitment")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.run_dir + "losses")

        if self.args.prior_n > 0:
            filtered_pix = savgol_filter(pixelcnn_losses, 51, 2)

             # save prior losses
            plt.cla()
            plt.yscale('log')
            plt.figure(figsize=(10,5))
            plt.title("Prior Training Loss")
            plt.plot(filtered_pix)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.savefig(self.run_dir + "prior_losses")

        # save models
        torch.save(vq_vae.state_dict(), self.run_dir + '/vq_vae.pt')

        # save models
        torch.save(pixel_cnn.state_dict(), self.run_dir + '/pixel_cnn.pt')
                
    @torch.inference_mode
    def sample(self, n, vqvae, pixel_cnn):
        latents = pixel_cnn.sample(n)
        g = vqvae.decode(latents.view(-1))

        # uniform sampling
        latents_uniform = torch.randint_like(latents, high=self.args.k)
        g_uniform = vqvae.decode(latents_uniform.view(-1))

        return g, g_uniform
    
    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        vq_vae = VectorQuantizedVariationalAutoEncoder(self.args)
        vq_vae.load_state_dict(torch.load(path + "vq_vae.pt", map_location=self.args.device))
        vq_vae.eval()

        prior = PixelCNN()
        prior.load_state_dict(torch.load(path + "pixel_cnn.pt", map_location=self.args.device))
        prior.eval()

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake_batch, _ = self.sample(n, vq_vae, prior)

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            plot_image(fake_batch[i], path + f"/f_{i}")
        print("### Done Generating Images ###")
    
class VectorQuantizer(nn.Module):

    def __init__(self, K=512, D=64):
        super(VectorQuantizer, self).__init__()
        self.K = K
        self.D = D
        self.embeddings = nn.Embedding(K, D)
        self.embeddings.weight.data.uniform_(-1./K, 1./K)

    def embed(self, indices, shape):
        quantized = torch.index_select(self.embeddings.weight, 0, indices).view(shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        flattened = x.view(-1, self.D)

        distances = torch.cdist(flattened,self.embeddings.weight)
        indices = torch.argmin(distances,dim=1) 
        indices = indices.reshape(x.shape[:-1])

        return indices

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVariationalAutoEncoder(nn.Module):

    def __init__(self, args, hidden_channels=256):
        super().__init__()

        input_dim = args.channel_size
        dim = hidden_channels
        K = args.k

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.vq = VectorQuantizer(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.vq(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.vq.embed(latents, (-1, 32, 32, 256))
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e = self.encoder(x)
        latents = self.vq(z_e).view(-1)
        z_q = self.vq.embed(latents, (-1, 32, 32, 256))

        # straight through gradient
        st_z_q = z_e + (z_q - z_e).detach()

        x_hat = self.decoder(st_z_q)
        return x_hat, z_e, z_q
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)