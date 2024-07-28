import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from einops.layers.torch import Rearrange
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
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

            save_conf(self.args, self.run_dir)

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

    def generate(self, path, n = 5):
        print("### Begin Generating Images ###")
        shape = (n, self.channel_size, self.args.dim, self.args.dim)

        ddpm = DenoisingDiffusionModel(self.args, self.args.device)
        ddpm.load_state_dict(torch.load(path + "/ddpm.pt"))
        ddpm.to(self.args.device)
        ddpm.eval()

        batch, _ = next(iter(self.dataloader))
        batch = scale_0_1(batch.to(self.args.device))
        fake_progress = ddpm.sample(shape)

        make_dir(path + "/real")
        make_dir(path + "/fake")
        for i in range(n):
            plot_image(batch[i], path + f"/real/r_{i}")

            make_dir(path + f"/fake/progress")
            make_dir(path + f"/fake/progress/f_{i}")
            for t in range(len(fake_progress)):
                if t == len(fake_progress) - 1:
                    plot_image(fake_progress[t][i], path + f"/fake/f_{i}")
                plot_image(fake_progress[t][i], path + f"/fake/progress/f_{i}/f_{i}_{t * (self.args.t // 10)}")
        print("### Done Generating Images ###")

    def train(self):
        ddpm = DenoisingDiffusionModel(self.args, self.args.device)
        ddpm.to(self.args.device)
        optimizer = optim.Adam(ddpm.parameters(), lr=self.args.lr)

        if self.args.cosine_lr:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=500,
                num_training_steps=(len(self.dataloader) * self.args.n),
            )
        else:
            scheduler = get_constant_schedule(
                optimizer=optimizer
            )

        losses = []

        print("### Begin Training Procedure ###")
        for epoch in range(self.args.n):

            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for i, batch in enumerate(self.dataloader, 0):
                batch, _ = batch
                batch = batch.to(self.args.device)
                batch = scale_minus1_1(batch)

                optimizer.zero_grad()
                batch_noise_hat, batch_noise = ddpm(batch)
                loss = F.mse_loss(batch_noise_hat, batch_noise)
                loss.backward()

                nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.n, i, len(self.dataloader), loss.item()))
                    
                progress_bar.update(1)
                    
            if epoch == 0:
                ts = torch.randint(self.args.t, (batch.shape[0],), device=self.args.device)
                onnx_program = torch.onnx.dynamo_export(ddpm.noise_net, batch, ts)
                onnx_program.save(self.run_dir + "/ddpm.onnx")
                plot_batch(scale_0_1(batch), self.progress_dir + f"train_example")

            fake = ddpm.sample(batch.shape)[-1]
            plot_batch(scale_0_1(fake), self.progress_dir + f"epoch:{epoch:04d}")

        print("### End Training Procedure ###")
        self.save_train_data(losses, ddpm)

class DenoisingDiffusionModel(nn.Module):
    def __init__(self, args, device):
        super(DenoisingDiffusionModel, self).__init__()

        self.args = args
        self.device = device

        self.noise_net = NoiseNet(self.args, init_dim=128, dim_mults=[1, 1, 2, 2, 4, 4], attn_resolutions=[8])

        T = torch.tensor(self.args.t).to(torch.float32)
        beta = torch.linspace(self.args.b_0, self.args.b_t, self.args.t, dtype=torch.float32, device=self.device)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        one_minus_alpha_bar = 1.0 - alpha_bar

        sqrt_alpha = torch.sqrt(alpha)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

        self.register_buffer("T", T)
        self.register_buffer("beta", beta)
        
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("one_minus_alpha_bar", one_minus_alpha_bar)

        self.register_buffer("sqrt_alpha", sqrt_alpha)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)

    def _get_variance(self, t):
        beta = self.beta[t]
        if self.args.fixed_large:
            return beta
        else:
            one_minus_alpha_bar_prev = self.one_minus_alpha_bar[t-1] if t >= 0 else torch.tensor(0.0)
            one_minus_alpha_bar = self.one_minus_alpha_bar[t]
            return beta * one_minus_alpha_bar_prev / one_minus_alpha_bar
    
    def _noise_t(self, x0, t):
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar[t])
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar[t])
        noise = torch.randn_like(x0, device=self.device)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise
    
    def _get_sample_ts(self):
        T = self.T.int().item()
        ts = torch.linspace(0, T, steps=11)
        ts -= torch.ones_like(ts)
        ts[0] = 0.0
        return torch.round(ts)
    
    # @torch.inference_mode()
    # def _sample_t(self, xt, t):
    #     ts = torch.ones((len(xt), 1), dtype=torch.float32, device=self.device) * t
    #     beta = extract(self.beta[t])
    #     sqrt_alpha = extract(self.sqrt_alpha[t])
    #     sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar[t])

    #     #noise_pred = self.noise_net(xt, ts)
    #     noise_pred = self.noise_net(xt, ts.squeeze()).sample
    #     xt_prev = (xt - noise_pred * beta / sqrt_one_minus_alpha_bar) / sqrt_alpha

    #     if t > 0:
    #         posterior_variance = self._get_variance(t) ** 0.5
    #         xt_prev += posterior_variance * torch.randn_like(xt_prev, device=self.device)
    #     return xt_prev

    @torch.inference_mode()
    def _sample_t(self, xt, t):
        ts = torch.ones((len(xt), ), dtype=torch.float32, device=self.device) * t
        beta = extract(self.beta[t])

        sqrt_alpha = extract(self.sqrt_alpha[t])
        sqrt_alpha_bar = extract(self.sqrt_alpha_bar[t])
        sqrt_alpha_bar_prev = extract(self.sqrt_alpha_bar[t-1] if t >= 0 else torch.tensor(1.0))

        one_minus_alpha_bar = extract(self.one_minus_alpha_bar[t])
        one_minus_alpha_bar_prev = extract(self.one_minus_alpha_bar[t-1] if t >= 0 else torch.tensor(0.0))
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alpha_bar[t])
        
        x0_coeff = sqrt_alpha_bar_prev * beta / one_minus_alpha_bar
        xt_coeff = sqrt_alpha * one_minus_alpha_bar_prev / one_minus_alpha_bar

        noise_pred = self.noise_net(xt, ts)
        x0_pred = (xt - noise_pred * sqrt_one_minus_alpha_bar) / sqrt_alpha_bar
        x0_pred = torch.clamp(x0_pred, min=-1, max=1)
        xt_prev = x0_coeff * x0_pred + xt_coeff * xt

        if t > 0:
            posterior_variance = self._get_variance(t) ** 0.5
            xt_prev += posterior_variance * torch.randn_like(xt, device=self.device)
        return xt_prev
    
    @torch.inference_mode()
    def sample(self, shape):
        images_list = []
        xt = torch.randn(shape, device=self.device)
        T = self.T.int().item()
        sample_ts = self._get_sample_ts()

        for t in tqdm(reversed(range(0, T)), position=0):
            xt = self._sample_t(xt, t)
            if t in sample_ts:
                images_list.append(scale_0_1(xt).cpu())
        return images_list
    
    def forward(self, x):
        T = self.T.int().item()
        ts = torch.randint(T, (x.shape[0],), device=self.device)
        xt, noise = self._noise_t(x, ts)
        noise_hat = self.noise_net(xt, ts)
        return noise_hat, noise

class NoiseNet(nn.Module):
    def __init__(self, args, init_dim=64, dim_mults = [1, 2, 4, 8, 16], attn_resolutions = [16]):
        super(NoiseNet, self).__init__()

        self.args = args
        self.attn_resolutions = attn_resolutions
        self.input_conv = nn.Conv2d(args.channel_size, init_dim, 7, padding=3)

        dim_time = args.dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(dim = args.dim, device=args.device),
            nn.Linear(args.dim, dim_time),
            nn.GELU(),
            nn.Linear(dim_time, dim_time)
        )

        num_resolutions = len(dim_mults)
        dims = [init_dim] + [init_dim * mult for mult in dim_mults]
        resolutions = [init_dim] + [int(args.dim * r) for r in torch.cumprod(torch.ones(num_resolutions) * 0.5, dim=0).tolist()]
        in_out_res = list(enumerate(zip(dims[:-1], dims[1:], resolutions)))

        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out, res) in in_out_res:
            downsample = (i < (num_resolutions - 1))
            attn = (res in attn_resolutions)

            self.downs.append(
                DownBlock(dim_in, dim_out, dim_time, attn, downsample)
            )

        dim_mid = dims[-1]
        self.mid = MidBlock(dim_mid, dim_time)

        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out, res) in reversed((in_out_res)):
            upsample = (i > 0)
            attn = (res in attn_resolutions)

            self.ups.append(
                UpBlock(dim_in, dim_out, dim_time, attn, upsample)
            )

        self.output_res = ResnetBlock(init_dim * 2, init_dim, dim_time)
        self.output_conv = nn.Conv2d(init_dim, args.channel_size, 1)

    def forward(self, x, t):
        t = self.time_embedding(t)
        x = self.input_conv(x)
        res_stack = [x.clone()]

        for down in self.downs:
            x, residuals = down(x, t)
            res_stack += residuals

        x = self.mid(x, t)

        for up in self.ups:
            x = up(x, t, res_stack)

        x = torch.cat((x, res_stack.pop()), dim=1)
        x = self.output_res(x, t)
        x = self.output_conv(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = dim_head * heads
        self.norm = nn.GroupNorm(1, dim)
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, h * w), qkv)

        q = q * self.scale
        sim = torch.einsum("b h c i, b h c j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attention = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h c j -> b h i c", attention, v)
        out = out.permute(0, 1, 3, 2).reshape((b, self.hidden_dim, h, w))
        return self.to_out(out)

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_time, attn=False, upsample=True):
        super().__init__()

        self.block1 = ResnetBlock(dim_out + dim_in, dim_out, dim_time)
        self.block2 = ResnetBlock(dim_out + dim_in, dim_out, dim_time)
        self.attn = Attention(dim_out) if attn else nn.Identity()
        self.us = Upsample(dim_out, dim_in) if upsample else nn.Conv2d(dim_out, dim_in, 3, padding=1) 

    def forward(self, x, t, r):
        x = torch.cat((x, r.pop()), dim=1)
        x = self.block1(x, t)
        x = torch.cat((x, r.pop()), dim=1)
        x = self.block2(x, t)
        x = self.attn(x)
        x = self.us(x)
        return x 
    
class MidBlock(nn.Module):
    def __init__(self, dim, dim_time):
        super().__init__()

        self.conv1 = ResnetBlock(dim, dim, dim_time)
        self.attn = Attention(dim)
        self.conv2 = ResnetBlock(dim, dim, dim_time)

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.attn(x)
        x = self.conv2(x, t)
        return x

class DownBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_time, attn=False, downsample=True):
        super().__init__()

        self.block1 = ResnetBlock(dim_in, dim_in, dim_time)
        self.block2 = ResnetBlock(dim_in, dim_in, dim_time)
        self.attn = Attention(dim_in) if attn else nn.Identity()
        self.ds = Downsample(dim_in, dim_out) if downsample else nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x, t):
        residuals = []
        x = self.block1(x, t)
        residuals.append(x.clone())
        x = self.block2(x, t)
        x = self.attn(x)
        residuals.append(x.clone())
        x = self.ds(x)
        return x, residuals   

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.time_fc = nn.Linear(time_embedding_dim, in_channels)
        self.conv1 = ConvBlock(in_channels, out_channels) 
        self.conv2 = ConvBlock(out_channels, out_channels) 

        self.conv_res = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        t_emb = self.time_fc(F.silu(t))[:, :, None, None]
        h = self.conv1(x + t_emb)
        h = self.conv2(h)
        r = self.conv_res(x)
        return h + r

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()

        self.norm = nn.GroupNorm(groups, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.silu(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device, theta=10000):
        super().__init__()
        self.dim = dim
        self.device = device
        self.theta = theta

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
