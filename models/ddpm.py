import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from utils import *
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from tqdm import tqdm

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

        if not self.args.test:
            self.run_dir = "train/ddpm-t={self.args.t}_lr={self.args.lr}/"
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

        self.prepare_noise_schedule()

    def __extract(self, t):
        for _ in range(4 - len(t.shape)):
            t = torch.unsqueeze(t, -1)
        return t

    def prepare_noise_schedule(self):
        self.b_0 = self.args.b_0
        self.b_t = self.args.b_t
        self.t = self.args.t
        self.beta = torch.linspace(self.args.b_0, self.args.b_t, self.args.t, device=self.args.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.args.device)
        self.alpha_bar_prev = torch.roll(self.alpha_bar, 1, 0).to(self.args.device)
        self.alpha_bar_prev[0] = 1.0
        
        # noise coefficients
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        # denoise coefficients
        self.sqrt_recip_alpha = 1 / self.sqrt_alpha_bar
        self.sqrt_recip_one_minus_alpha_bar = 1 / self.sqrt_one_minus_alpha_bar

        print(self.beta.shape)
        print(self.alpha.shape)
        print(self.alpha_bar.shape)
        print(self.alpha_bar_prev.shape)
        print(self.sqrt_alpha_bar.shape)
        print(self.sqrt_one_minus_alpha_bar.shape)
        print(self.sqrt_recip_alpha.shape)
        print(self.sqrt_recip_one_minus_alpha_bar.shape)

    def noise_t(self, x, t):
        print(t)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]
        noise = torch.randn_like(x, device=self.args.device)
        noise_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return noise_x, noise
    
    def sample_t(self, noise_net, x, t, noise):
        with torch.no_grad():
            noise_pred = noise_net(x, t)
        beta = self.beta[t]
        alpha = self.alpha[t]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t]
        sqrt_recip_one_minus_alpha_bar = self.sqrt_recip_one_minus_alpha_bar[t]
        sample = sqrt_recip_alpha * (x - ((1 - alpha) / sqrt_recip_one_minus_alpha_bar) * noise_pred) + torch.sqrt(beta) * noise
        return sample.clamp(-1, 1)
    
    def sample(self, noise_net, shape):
        noise_net.eval()
        images = torch.randn(shape, device=self.args.device)
        images_list = []

        for t in tqdm(reversed(range(1, self.args.t+1)), position=0):
            if t > 1:
                z = torch.randn(shape, device=self.args.device)
            else:
                z = torch.zeros(shape, device=self.args.device)
            ts = torch.ones((len(images), 1), dtype=int, device=self.args.device) * t
            images = self.sample_t(noise_net, images, ts, z)
            if t % (self.args.t // 10) == 0:
                images_list.append(images.cpu())
        noise_net.train()
        return images_list

    def save_train_data(self, losses, noise_net):

        # save models
        torch.save(noise_net.state_dict(), self.run_dir + '/noise_net.pt')

        # save losses
        plt.cla()
        plt.figure(figsize=(10,5))
        plt.title("Training Losses")
        plt.plot(losses,label="p")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")

    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        shape = (n, self.channel_size, self.args.dim, self.args.dim)
        noise_net = NoiseNet(self.args)
        noise_net.load_state_dict(torch.load(path + "/noise_net.pt"))
        noise_net.to(self.args.device)
        noise_net.eval()

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)
        fake_progress = self.sample(noise_net, shape)
        make_dir(path + "/f_progress")

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            for t in range(len(fake_progress)):
                if t == len(fake_progress) - 1:
                    plot_image(fake_progress[t][i], path + f"/f_{i}")
                plot_image(fake_progress[t][i], path + f"/f_progress/f_{i}_{t * (self.args.t // 10)}")
        print("### Done Generating Images ###")

    def train(self):
        noise_net = NoiseNet(self.args)
        noise_net.to(self.args.device)
        optimizer = optim.Adam(noise_net.parameters(), lr=self.args.lr)
        mse = nn.MSELoss()

        losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, labels = batch
                batch_t = torch.randint_like(labels, high=self.args.t+1, device=self.args.device)
                batch = batch.to(self.args.device)
                labels = labels.to(self.args.device)

                print(batch.shape)
                print(batch_t.shape)
                print(labels.shape)

                batch_noised_images, batch_noise = self.noise_t(batch, batch_t)
                batch_t = batch_t[:, None]

                if iters == 0:
                    plot_batch(batch, self.progress_dir + f"train_example")

                noise_net.zero_grad()
                batch_noise_pred = noise_net(batch_noised_images, batch_t)
                mse_loss = mse(batch_noise_pred, batch_noise)
                mse_loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader), mse_loss.item()))

                losses.append(mse_loss.item())

                if (iters > 0 and iters % 5000 == 0) or ((epoch == self.args.n-1) and (i == len(self.dataloader)-1)):

                    with torch.no_grad():
                        noise_net.eval()
                        fake = self.sample(noise_net, batch.shape)[-1]
                        noise_net.train()
                    plot_batch(fake, self.progress_dir + f"iter:{iters}")

                iters += 1

        print("### End Training Procedure ###")
        self.save_train_data(losses, noise_net)

class NoiseNet(nn.Module):
    def __init__(self, args, init_dim=64, dim_mults = (1, 2, 4, 8, 16), attn_resolutions = [16]):
        super(NoiseNet, self).__init__()

        self.args = args
        self.attn_resolutions = attn_resolutions

        time_dim = args.dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(dim = args.dim, device=self.args.device),
            nn.Linear(args.dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_conv = nn.Conv2d(self.args.channel_size, init_dim, 7, padding=3)

        self.downs = []
        self.ups = []

        num_resolutions = len(dim_mults)
        dims = [init_dim] + [init_dim * mult for mult in dim_mults]
        resolutions = [int(args.dim * r) for r in torch.cumprod(torch.ones(num_resolutions) * 0.5, dim=0).tolist()]
        in_out_res = list(enumerate(zip(dims[:-1], dims[1:], resolutions)))

        for i, (dim_in, dim_out, res) in in_out_res:
            is_last = (i == (num_resolutions - 1))
            self.downs.append(
                nn.ModuleList([
                    ConvNextBlock(dim_in, dim_in, time_dim),
                    ConvNextBlock(dim_in, dim_in, time_dim),
                    SelfAttention(dim_in) if res in attn_resolutions else nn.Identity(),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                    ]
                )
            )

        self.mid_conv_1 = ConvNextBlock(dims[-1], dims[-1], time_dim)
        self.mid_attn = SelfAttention(dims[-1])
        self.mid_conv_2 = ConvNextBlock(dims[-1], dims[-1], time_dim)

        for i, (dim_in, dim_out, res) in reversed((in_out_res)):
            is_first = (i == 0)
            self.ups.append(
                nn.ModuleList([
                    ConvNextBlock(dim_out + dim_in, dim_out, time_dim),
                    ConvNextBlock(dim_out + dim_in, dim_out, time_dim),
                    SelfAttention(dim_out) if res in attn_resolutions else nn.Identity(),
                    Upsample(dim_out, dim_in) if not is_first else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        self.output_res = ConvNextBlock(init_dim * 2, init_dim, time_dim)
        self.output_conv = nn.Conv2d(init_dim, self.args.channel_size, 1)

        for module in self.downs:
            module.to(self.args.device)
        for module in self.ups:
            module.to(self.args.device)

    def forward(self, x, t):
        t = self.time_embedding(t)
        x = self.input_conv(x)
        residual = x.clone()

        res_stack = []
        for down1, down2, attn, downsample in self.downs:
            x = down1(x, t)
            res_stack.append(x)
            x = down2(x, t)
            x = attn(x)
            res_stack.append(x)
            x = downsample(x)

        x = self.mid_conv_1(x, t)
        x = self.mid_attn(x)
        x = self.mid_conv_2(x, t)

        for up1, up2, attn, upsample in self.ups:
            x = torch.cat((x, res_stack.pop()), dim=1)
            x = up1(x, t)
            x = torch.cat((x, res_stack.pop()), dim=1)
            x = up2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, residual), dim=1)
        x = self.output_res(x, t)
        x = self.output_conv(x)
        return x

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.time_input = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embedding_dim, in_channels)
        )

        self.input_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        self.conv_block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, out_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        )

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        h = self.input_conv(x)
        h += self.time_input(t)[:, :, None, None]
        h = self.conv_block(x)
        h += self.residual_connection(x)
        return h
    
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.GroupNorm(1, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), 'b (c qkv) h w -> qkv b c h w', qkv=3)
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(in_channels * 4, out_channels, 1),
        )

    def forward(self, x):
        return self.model(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device, theta = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("'dim' must be even, but received {dim}")
        
        self.dim = dim
        self.device = device
        self.theta = theta

    def forward(self, t):
        ks = torch.arange(0, self.dim, 2, device=self.device)
        w_k = torch.exp(math.log(self.theta) * -ks / self.dim)
        emb = torch.cat((torch.sin(t * w_k), torch.cos(t * w_k)), dim=-1)
        return emb