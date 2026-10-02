import torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Blk(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8, o)
        self.n2 = nn.GroupNorm(8, o)
        self.se = SEBlock(o)
    def forward(self, x):
        x = F.gelu(self.n1(self.c1(x)))
        x = F.gelu(self.n2(self.c2(x)))
        return self.se(x)

class UNet(nn.Module):
    def __init__(self, ci, co, w):
        super().__init__()
        self.e1 = Blk(ci, w)
        self.e2 = Blk(w, 2 * w)
        self.e3 = Blk(2 * w, 4 * w)
        self.b = Blk(4 * w, 4 * w)
        self.d3 = Blk(8 * w, 2 * w)
        self.d2 = Blk(4 * w, w)
        self.d1 = Blk(2 * w, w)
        self.out = nn.Conv2d(w, co, 1)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        u = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u, e3], 1))
        u = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u, e2], 1))
        u = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u, e1], 1))
        return self.out(d1)
