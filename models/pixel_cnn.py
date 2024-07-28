import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
    
class GatedMaskedConv2d(nn.Module):
    def __init__(self, dim, kernel, mask_type, residual):
        super(GatedMaskedConv2d, self).__init__()
        assert kernel % 2 == 1, "kernel size must be odd"

        # equivalent to masked conv for each stack
        v_kernel_shape = (kernel // 2 + 1, kernel)
        v_padding_shape = (kernel // 2, kernel // 2)
        h_kernel_shape = (1, kernel // 2 + 1)
        h_padding_shape = (0, kernel // 2)

        self.v_stack = nn.Conv2d(dim, dim * 2, v_kernel_shape, 1, v_padding_shape)

        self.v_to_h = nn.Conv2d(2 * dim, 2 * dim, 1)

        self.h_stack = nn.Conv2d(dim, dim * 2, h_kernel_shape, 1, h_padding_shape)

        self.h_residual = nn.Conv2d(dim, dim, 1)
        self.residual = residual

        if mask_type ==' A':
            self.v_stack.weight.data[:, :, -1].zero_()
            self.h_stack.weight.data[:, :, :, -1].zero_()
        
    def gate(self, x):
        x_f, x_g = x.chunk(2, dim=1)
        return F.tanh(x_f) * F.sigmoid(x_g)

    def forward(self, x_v, x_h):
        v = self.v_stack(x_v)
        v = v[:, :, :x_v.shape[-1], :]
        v_out = self.gate(v)

        h = self.h_stack(x_h)
        h = h[:, :, :, :x_h.shape[-2]]
        v_to_h = self.v_to_h(v)
        h_out = self.gate(h + v_to_h)
        h_out = self.h_residual(h_out)

        if self.residual:
            h_out += x_h

        return v_out, h_out
    
class PixelCNN(nn.Module):
    def __init__(self, dim=64, n_layers=15, k=512):
        super(PixelCNN, self).__init__()

        self.dim = dim
        self.k = k
        
        self.embedding = nn.Embedding(k, dim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(dim, kernel, mask_type, residual)
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, 1024, 1),
            nn.ReLU(True),
            nn.Conv2d(1024, k, 1)
        )

    @torch.inference_mode
    def sample(self, n, shape=(32, 32)):
        device = next(self.parameters()).device
        logits = torch.zeros((n, *shape), dtype=torch.int64, device=device)

        print("### Generating prior ###")
        with tqdm(total=shape[0] * shape[1], position=tqdm._get_free_pos()) as pbar:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    l = self.forward(logits)
                    probs = F.softmax(l[:, :, i, j], -1)
                    e = probs.multinomial(1).squeeze().data
                    logits.data[:, i, j].copy_(e)
                    pbar.update(1)

        return logits

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)
        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)
        out = self.out_conv(x_h)
        return out

