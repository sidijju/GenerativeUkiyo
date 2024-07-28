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
    
        vq_vae = VectorQuantizedVariationalAutoEncoder(self.args)
        if self.args.checkpoint:
            vq_vae.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
            print("Loaded checkpoint from", self.args.checkpoint)
        vq_vae.to(self.args.device)
        optimizer = optim.Adam(vq_vae.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        sample_batch, _ = next(iter(self.dataloader))
        sample_batch = sample_batch.to(self.args.device)
        plot_batch(sample_batch, self.progress_dir + f"train_example")

        losses = []

        print("### Begin Training Procedure ###")
        print("Training VQ-VAE")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                vq_vae.zero_grad()
                batch_hat, dictionary_loss, commitment_loss = vq_vae(batch)
                reproduction_loss = F.mse_loss(batch_hat, batch)
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

        pixelcnn_losses = []

        print("Training PixelCNN prior")
        for epoch in tqdm(range(self.args.prior_n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                with torch.no_grad():
                    _, labels, _, _ = vq_vae.encode(batch)

                logits = vq_vae.pixel_cnn(labels)
                loss = F.cross_entropy(logits, labels)

                loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                pixelcnn_losses.append((loss.item()))

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.prior_n, i, len(self.dataloader), loss.item()))
                    
            sample_batch = vq_vae.sample(sample_batch.shape[0])
            plot_batch(sample_batch, self.prior_dir + f"gen-epoch:{epoch}")

        print("### End Training Procedure ###")
        self.save_train_data(losses, pixelcnn_losses, vq_vae)

    def save_train_data(self, losses, pixelcnn_losses, vq_vae):

        totals = [loss[0] for loss in losses]
        reconstructions = [loss[1] for loss in losses]
        dictionary_losses = [loss[2] for loss in losses]
        commitment_losses = [loss[3] for loss in losses]
        pixelcnn_losses = [loss[0] for loss in pixelcnn_losses]

        filtered_totals = savgol_filter(totals, 51, 2)
        filtered_recs = savgol_filter(reconstructions, 51, 2)
        filtered_dics = savgol_filter(dictionary_losses, 51, 2)
        filtered_cs = savgol_filter(commitment_losses, 51, 2)
        filtered_pix = savgol_filter(pixelcnn_losses, 51, 2)

        # save models
        torch.save(vq_vae.state_dict(), self.run_dir + '/vq_vae.pt')

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
        # save prior losses
        plt.cla()
        plt.yscale('log')
        plt.figure(figsize=(10,5))
        plt.title("Prior Training Loss")
        plt.plot(filtered_pix)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "prior_losses")
                
    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        vq_vae = VectorQuantizedVariationalAutoEncoder(self.args)
        vq_vae.load_state_dict(torch.load(path + "/vq_vae.pt", map_location=self.args.device))
        vq_vae.eval()

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)

        with torch.no_grad():
            fake_batch = vq_vae.sample()

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            plot_image(fake_batch[i], path + f"/f_{i}")
        print("### Done Generating Images ###")

###############

class VectorQuantizedVariationalAutoEncoder(nn.Module):
    def __init__(self, args, hidden_channels=256):
        super(VectorQuantizedVariationalAutoEncoder, self).__init__()

        in_channels = args.channel_size
        num_embeddings = args.k
        self.embedding_dim = args.latent

        self.encoder = Encoder(in_channels, hidden_channels=hidden_channels)

        self.to_vq = nn.Sequential(
            nn.Conv2d(hidden_channels, self.embedding_dim, 3, 1, 1),
            nn.BatchNorm2d(self.embedding_dim),
        )

        self.vq = VectorQuantizer(num_embeddings, self.embedding_dim)

        self.decoder = Decoder(self.embedding_dim, in_channels, hidden_channels=hidden_channels)

        self.pixel_cnn = PixelCNN(dim=self.embedding_dim, k=num_embeddings)        

    @torch.inference_mode
    def sample(self, n):
        latents = self.pixel_cnn.sample(n)
        shape = (latents.shape[0], self.embedding_dim, *latents.shape[-2:])
        z_q = self.vq.select_embeddings(latents.view(-1), shape)
        g = self.decode(z_q)
        return g
    
    def encode(self, x):
        z_e = self.encoder(x)
        z_q, z_q_indices, dictionary_loss, commitment_loss = self.vq(self.to_vq(z_e))
        return z_q, z_q_indices, dictionary_loss, commitment_loss
    
    def decode(self, z_q):
        x_hat = self.decoder(z_q)
        return x_hat
    
    def forward(self, x):
        z_q, _, dictionary_loss, commitment_loss = self.encode(x)
        x_hat = self.decode(z_q)
        return x_hat, dictionary_loss, commitment_loss
    
#########################

class ResidualConvBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = self.bn1(self.conv1(F.relu(x)))
        h = self.bn2(self.conv2(F.relu(h)))
        return x + h

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels = 256,
                 residual_hidden_channels = 64,
                 ):
        super(Encoder, self).__init__()

        self.down1 = self.conv_block(in_channels, hidden_channels // 2)
        self.down2 = self.conv_block(hidden_channels // 2, hidden_channels)
        self.down3 = self.conv_block(hidden_channels, hidden_channels, kernels=3, stride=1)

        self.res1 = ResidualConvBlock(hidden_channels, residual_hidden_channels)
        self.res2 = ResidualConvBlock(hidden_channels, residual_hidden_channels)

    def conv_block(self, in_channels, out_channels, kernels=4, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernels, stride, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.down3(x)
        x = self.res1(x)
        x = self.res2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hidden_channels = 256,
                 residual_hidden_channels = 64,
                 ):
        super(Decoder, self).__init__()

        self.from_vq = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
        )

        self.res1 = ResidualConvBlock(hidden_channels, residual_hidden_channels)
        self.res2 = ResidualConvBlock(hidden_channels, residual_hidden_channels)

        self.up1 = self.conv_block(hidden_channels, hidden_channels // 2)
        self.up2 = self.conv_block(hidden_channels // 2, out_channels)

    def conv_block(self, in_channels, out_channels, kernels=4, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernels, stride, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.from_vq(x)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.up1(x))
        x = F.sigmoid(self.up2(x))
        return x
    
class VectorQuantizer(nn.Module):

    def __init__(self, K=512, D=64):
        super(VectorQuantizer, self).__init__()
        self.K = K
        self.D = D
        self.embeddings = nn.Embedding(K, D)
        self.embeddings.weight.data.uniform_(-1./K, 1./K)

    def select_embeddings(self, indices, shape):
        quantized = torch.index_select(self.embeddings.weight, 0, indices).view(shape)
        return quantized

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        flattened = x.view(-1, self.D)

        distances = torch.cdist(flattened,self.embeddings.weight)
        indices = torch.argmin(distances,dim=1) 

        quantized = self.select_embeddings(indices, x.shape)
        indices = indices.reshape(x.shape[:-1])

        dictionary_loss = F.mse_loss(x.detach(), quantized)
        commitment_loss = F.mse_loss(quantized.detach(), x)

        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, indices, dictionary_loss, commitment_loss