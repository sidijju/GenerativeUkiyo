import torch
import torch.nn as nn
import torch.nn.functional as F
    
class GatedMaskedConv2d(nn.Module):
    def __init__(self, dim, kernel, mask_type, residual):
        super(GatedMaskedConv2d, self).__init__()
        assert kernel % 2 != 0, "kernel size must be odd"

        # equivalent to masked conv for each stack
        v_kernel_shape = (kernel // 2 + 1, kernel)
        v_padding_shape = (kernel // 2, kernel // 2)
        h_kernel_shape = (1, kernel // 2 + 1)
        h_padding_shape = (0, kernel // 2)

        self.v_stack = nn.Conv2d(dim, dim * 2, v_kernel_shape, 1, v_padding_shape)

        self.v_to_h = nn.Conv2d(2 * dim, 2 * dim, 1)

        self.h_stack = nn.Conv2d(dim, dim * 2, h_kernel_shape, 1, h_padding_shape)

        if self.h_residual:
            self.h_residual = nn.Conv2d(dim, dim, 1)

        if mask_type ==' A':
            self.v_stack.weight.data[:, :, -1].zero_()
            self.h_stack.weight.data[:, :, :, -1].zero_()
        
    def gate(x):
        x_f, x_g = torch.chunk(2, dim=1)
        return F.tanh(x_f) * F.sigmoid(x_g)

    def forward(self, x_v, x_h):
        v = self.v_stack(x_v)
        v = self.gate(v)

        h = self.h_stack(x_h)
        v_to_h = self.v_to_h(v)
        h = self.gate(h + v_to_h)
        h = self.h_residual(h)

        if self.h_residual:
            h += x_h

        return v, h
    
class PixelCNN(nn.Module):
    def __init__(self, dim=32, n_layers=15, k=512):
        super(PixelCNN, self).__init__()

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
    def sample(self, n):
        pass

    def forward(self, x):
        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)
        out = self.out_conv(x_h)
        return out

