import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual):
        super(GatedMaskedConv2d, self).__init__()
        assert (kernel % 2) == 1, "kernel size must be odd"

        self.residual = residual
        self.mask_type = mask_type

        # equivalent to masked conv for each stack
        v_kernel_shape = (kernel // 2 + 1, kernel)
        v_padding_shape = (kernel // 2, kernel // 2)
        h_kernel_shape = (1, kernel // 2 + 1)
        h_padding_shape = (0, kernel // 2)

        self.vert_stack = nn.Conv2d(dim, dim * 2, v_kernel_shape, 1, v_padding_shape)

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        self.horiz_stack = nn.Conv2d(dim, dim * 2, h_kernel_shape, 1, h_padding_shape)

        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        
    def gate(self, x):
        x_f, x_g = x.chunk(2, dim=1)
        return F.tanh(x_f) * F.sigmoid(x_g)

    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.vert_stack.weight.data[:, :, -1].zero_()
            self.horiz_stack.weight.data[:, :, :, -1].zero_()
            
        v = self.vert_stack(x_v)
        v = v[:, :, :x_v.shape[-1], :]
        v_out = self.gate(v)

        h = self.horiz_stack(x_h)
        h = h[:, :, :, :x_h.shape[-2]]
        vert_to_horiz = self.vert_to_horiz(v)
        h_out = self.gate(h + vert_to_horiz)
        h_out = self.horiz_resid(h_out)

        if self.residual:
            h_out += x_h

        return v_out, h_out

class PixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=64, n_layers=15):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 1024, 1),
            nn.ReLU(True),
            nn.Conv2d(1024, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x):
        shp = x.size() + (-1, )
        
        x = self.embedding(x.view(-1).to(torch.long)).view(shp)
        x = x.permute(0, 3, 1, 2)

        x_v, x_h = (x, x)
        for _, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h)

        return self.output_conv(x_h)

    @torch.inference_mode
    def sample(self, n, shape=(32, 32)):
        param = next(self.parameters())
        x = torch.zeros(
            (n, *shape),
            dtype=torch.int64, device=param.device
        )

        print("### Generating prior ###")
        with tqdm(total=shape[0] * shape[1], position=tqdm._get_free_pos()) as pbar:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x)
                    probs = F.softmax(logits[:, :, i, j], -1)
                    x.data[:, i, j].copy_(
                        probs.multinomial(1).squeeze().data
                    )
                    pbar.update(1)
        return x

