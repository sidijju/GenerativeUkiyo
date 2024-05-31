import torch
import datetime
from utils import *

class DDPM:
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
            self.run_dir = "train/ddpm-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

    def save_train_data(self, losses, noise_net):

        # save models
        torch.save(noise_net.state_dict(), self.run_dir + '/noise_net.pt')

        # save losses
        plt.figure(figsize=(10,5))
        plt.title("Training Losses")
        plt.plot(losses,label="p")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")
                
    def generate(self, path, n = 5):
        pass

    def train(self, num_epochs = 5, lr = 0.0001):
        pass