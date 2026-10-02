import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class Blk(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8, o)
        self.n2 = nn.GroupNorm(8, o)
    def forward(self, x):
        x = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(x)))

class UFNO(nn.Module):
    def __init__(self, in_c, out_c, w=64):
        super().__init__()
        self.e1 = Blk(in_c, w)
        self.e2 = Blk(w, 2*w)
        self.e3 = Blk(2*w, 4*w)
        
        self.fno1 = SpectralConv2d(4*w, 4*w, modes1=4, modes2=8)
        self.fno_conv1 = nn.Conv2d(4*w, 4*w, 1)
        self.fno2 = SpectralConv2d(4*w, 4*w, modes1=4, modes2=8)
        self.fno_conv2 = nn.Conv2d(4*w, 4*w, 1)
        
        self.d3 = Blk(8*w, 2*w)
        self.d2 = Blk(4*w, w)
        self.d1 = Blk(2*w, w)
        
        self.mu_head = nn.Conv2d(w, out_c, 1)
        self.logvar_head = nn.Conv2d(w, out_c, 1)
        
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        
        b_f1 = F.gelu(self.fno1(e3) + self.fno_conv1(e3))
        b = F.gelu(self.fno2(b_f1) + self.fno_conv2(b_f1))
        
        d3 = self.d3(torch.cat([b, e3], 1))
        
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], 1))
        
        mu = self.mu_head(d1)
        logvar = self.logvar_head(d1)
        
        return mu, logvar
