import matplotlib.pyplot as plt
import torch
from abc import ABC
from utils import *
from scipy.signal import savgol_filter

class GAN(ABC):

    def __init__(self, 
                args,
                dataloader = None,
                ):
        
        self.args = args
        self.dataloader = dataloader
        self.channel_size = args.channel_size
        self.latent_size = args.latent

        if not self.args.test:
            self.run_dir = f"train/gan-n={self.args.n}_lr={self.args.lr}_batch={self.args.batchsize}/"
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

            save_conf(self.args, self.run_dir)

    def save_train_data(self, d_losses_real, d_losses_fake, g_losses, d_net, g_net):

        # save models
        torch.save(d_net.state_dict(), self.run_dir + '/discriminator.pt')
        torch.save(g_net.state_dict(), self.run_dir + '/generator.pt')

        filtered_g = savgol_filter(g_losses, 51, 3)
        filtered_d_real = savgol_filter(d_losses_real, 51, 3)
        filtered_d_fake = savgol_filter(d_losses_fake, 51, 3)
        filtered_d = filtered_d_real + filtered_d_fake

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
        plt.plot(filtered_d_real, label="D_real")
        plt.plot(filtered_d_fake, label="D_fake")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "d_losses")
                
    def generate(self, path, n = 5):
        pass

    def train(self):
        pass
