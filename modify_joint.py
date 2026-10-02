import re

with open('joint_asym.py', 'r') as f:
    content = f.read()

cbam_code = """
class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(1, c//r), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, c//r), c, 1, bias=False)
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca
        avg_sp = torch.mean(x, 1, keepdim=True)
        max_sp, _ = torch.max(x, 1, keepdim=True)
        sa = torch.sigmoid(self.conv(torch.cat([avg_sp, max_sp], 1)))
        return x * sa

class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c1=nn.Conv2d(i,o,3,padding=1); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.n1=nn.GroupNorm(8,o); s.n2=nn.GroupNorm(8,o)
        s.cbam=CBAM(o)
    def forward(s,x): x=F.gelu(s.n1(s.c1(x))); return s.cbam(F.gelu(s.n2(s.c2(x))))
"""

content = re.sub(r'class Blk\(nn\.Module\):.*?(?=class UNet)', cbam_code, content, flags=re.DOTALL)

with open('joint_asym_attn.py', 'w') as f:
    f.write(content)
