import re

with open("joint_asym_attn.py", "r") as f:
    text = f.read()

attn_code = """
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // r), channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
        s.se = SEBlock(o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return s.se(F.gelu(s.n2(s.c2(x))))
"""

# Replace the Blk class
text = re.sub(r'class Blk\(nn\.Module\):.*?(?=class UNet)', attn_code, text, flags=re.DOTALL)

with open("joint_asym_attn.py", "w") as f:
    f.write(text)
