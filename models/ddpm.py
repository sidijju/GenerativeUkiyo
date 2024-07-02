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
                 ):
        
        self.args = args
        self.dataloader = dataloader
        self.channel_size = args.channel_size

        if not self.args.test:
            self.run_dir = f"train/ddpm-n={self.args.n}_t={self.args.t}_lr={self.args.lr}/"
            self.progress_dir = self.run_dir + "progress/"
            make_dir(self.run_dir)
            make_dir(self.progress_dir)

    def save_train_data(self, losses, ddpm):

        # save models
        torch.save(ddpm.state_dict(), self.run_dir + '/ddpm.pt')

        # save losses
        plt.cla()
        plt.figure(figsize=(10,5))
        plt.title("Training Losses")
        plt.plot(losses,label="p")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")

    @torch.inference_mode()
    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        shape = (n, self.channel_size, self.args.dim, self.args.dim)

        ddpm = DenoisingDiffusionModel(self.args)
        ddpm.load_state_dict(torch.load(path + "/noise_net.pt"))
        ddpm.to(self.args.device)
        ddpm.eval()

        batch, _ = next(iter(self.dataloader))
        batch = batch.to(self.args.device)
        fake_progress = ddpm.sample(shape)
        make_dir(path + "/f_progress")

        for i in range(n):
            plot_image(batch[i], path + f"/r_{i}")
            for t in range(len(fake_progress)):
                if t == len(fake_progress) - 1:
                    plot_image(fake_progress[t][i], path + f"/f_{i}")
                plot_image(fake_progress[t][i], path + f"/f_progress/f_{i}_{t * (self.args.t // 10)}")
        print("### Done Generating Images ###")

    def train(self):
        ddpm = DenoisingDiffusionModel(self.args)
        ddpm.to(self.args.device)
        optimizer = optim.Adam(ddpm.parameters(), lr=self.args.lr)
        mse = nn.MSELoss()

        losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)

                ddpm.zero_grad()
                batch_noise_hat, batch_noise = ddpm(batch)
                mse_loss = mse(batch_noise_hat, batch_noise)
                mse_loss.backward()
                optimizer.step()

                losses.append(mse_loss.item())

                #############################
                ####   Metrics Tracking  ####
                #############################

                if iters == 0:
                    plot_batch(batch, self.progress_dir + f"train_example")

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader), mse_loss.item()))

                if (iters % 1000 == 0) or ((epoch == self.args.n-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = ddpm.sample(batch.shape)[-1]
                    plot_batch(scale_0_1(fake), self.progress_dir + f"iter:{iters}")

                iters += 1

        print("### End Training Procedure ###")
        self.save_train_data(losses, ddpm)

class DenoisingDiffusionModel(nn.Module):
    def __init__(self, args):
        super(DenoisingDiffusionModel, self).__init__()

        self.args = args
        self.noise_net = NoiseNet(self.args)

        beta = torch.linspace(self.args.b_0, self.args.b_t, self.args.t, device=self.args.device)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0).to(self.args.device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        recip_sqrt_alpha = 1 / torch.sqrt(alpha)
        recip_sqrt_one_minus_alpha_bar = 1 / sqrt_one_minus_alpha_bar

        self.register_buffer("beta", beta.to(torch.float32))
        self.register_buffer("alpha", alpha.to(torch.float32))
        self.register_buffer("alpha_bar", alpha_bar.to(torch.float32))
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar.to(torch.float32))
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar.to(torch.float32))
        self.register_buffer("recip_sqrt_alpha", recip_sqrt_alpha.to(torch.float32))
        self.register_buffer("recip_sqrt_one_minus_alpha_bar", recip_sqrt_one_minus_alpha_bar.to(torch.float32))
    
    def noise_t(self, x0, t):
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar[t])
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar[t])
        noise = torch.randn_like(x0, device=self.args.device)
        noise_x = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return noise_x, noise
    
    @torch.inference_mode()
    def sample_t(self, x, t, noise):
        noise_pred = self.noise_net(x, t)
        beta = extract(self.beta[t])
        alpha = extract(self.alpha[t])
        recip_sqrt_alpha = extract(self.recip_sqrt_alpha[t])
        recip_sqrt_one_minus_alpha_bar = extract(self.recip_sqrt_one_minus_alpha_bar[t])
        noise_removed_x = x - (1 - alpha) * recip_sqrt_one_minus_alpha_bar * noise_pred
        return recip_sqrt_alpha * noise_removed_x + torch.sqrt(beta) * noise
    
    @torch.inference_mode()
    def sample(self, shape):
        images = torch.randn(shape, device=self.args.device)
        images_list = []
        record_ts = torch.linspace(0, self.args.t-1, 10, dtype=torch.uint8)

        for t in tqdm(reversed(range(0, self.args.t)), position=0):
            z = torch.randn(shape, device=self.args.device) if t > 0 else torch.zeros(shape, device=self.args.device)
            ts = torch.ones((len(images), 1), dtype=int, device=self.args.device) * t
            images = self.sample_t(self.noise_net, images, ts, z)

            if t in record_ts:
                images_list.append(scale_0_1(images).cpu())
        return images_list
    
    def forward(self, x):
        print(x.shape[0])
        print((x.shape[0], ))
        t = torch.randint(self.args.t, (x.shape[0], ), device=self.args.device)
        x = scale_minus1_1(x)
        x_t, noise = self.noise_t(x, t)
        t = t[:, None]
        noise_hat = self.noise_net(x_t, noise)
        return noise_hat, noise

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