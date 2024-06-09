import matplotlib.pyplot as plt
import torch
from datetime import datetime
from abc import ABC
from utils import *

class GAN(ABC):

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

        if not self.args.test:
            self.run_dir = "train/gan-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

    def save_train_data(self, d_losses_real, d_losses_fake, g_losses, d_net, g_net):

        # save models
        torch.save(d_net.state_dict(), self.run_dir + '/discriminator.pt')
        torch.save(g_net.state_dict(), self.run_dir + '/generator.pt')

        # save losses
        plt.cla()
        plt.figure(figsize=(10,5))
        plt.yscale('log')
        plt.title("Training Losses")
        plt.plot(g_losses,label="G")
        plt.plot([sum(x)/2 for x in zip(d_losses_real, d_losses_fake)], label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")
                
    def generate(self, path, n = 5):
        pass

    def train(self, g_lr = .0001, d_lr = .0004):
        pass
