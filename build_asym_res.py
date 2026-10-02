import re
with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym.py', 'r') as f:
    content = f.read()

replacement = """
with torch.no_grad():
    for i in range(0,len(RS),16):
        xi_flat = flat(XI[i:i+16]).to(DEV)
        pr_flat = flat(PR[i:i+16]).to(DEV)
        x = torch.cat([xi_flat, pr_flat], 1)
        o=net(x)
        c=o[:,:40].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        
        u_std = xi_flat[:, :20].std(dim=1, keepdim=True).clamp(min=1e-4)
        v_std = xi_flat[:, 20:40].std(dim=1, keepdim=True).clamp(min=1e-4)
        l_std = torch.cat([u_std, v_std], 1).unsqueeze(1).expand(-1, 20, -1, -1, -1).permute(0,1,3,4,2)
        
        w_d = o[:,40:80].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        w_u = o[:,80:].reshape(-1,20,2,32,64).permute(0,1,3,4,2)
        
        # apply residual scaling logic during build
        w_d = (w_d * float(ck["sd_"]) + float(ck["mu"])) + torch.log(l_std)
        w_u = (w_u * float(ck["sd_"]) + float(ck["mu"])) + torch.log(l_std)
        # Note: the rest of build_asym expects Ww_d and Ww_u to just be the logits it seems? 
        # Wait, the original build_asym just takes the raw outputs! 
        # Let's revert w_d and w_u to what the LUT builder expects: The raw output of the UNet which is the log bounds!
        # Ah wait, the original `build_asym.py` literally fits LUTs over `w_d` and `w_u` directly (which were the outputs). 
        # Since I've changed the network's output to mean `w_d_out = log(error) - log(l_std) - mu / sd`, 
        # the true log(error) is `(w_d_out * sd + mu) + log(l_std)`. 
        # So I will just supply this `true log(error)` as the "Ww" to fit the LUT over!
        
        Cc[i:i+16]=c.cpu().numpy()
        Ww_d[i:i+16]=w_d.cpu().numpy()
        Ww_u[i:i+16]=w_u.cpu().numpy()
"""

# replace the inference block
content = re.sub(r'with torch\.no_grad\(\):.*?print\("\[scored\]', replacement + '\nprint("[scored]', content, flags=re.DOTALL)

with open('/Users/aryamannsrivastava/Desktop/IMPORTANT/UGP/build_asym_res_fixed.py', 'w') as f:
    f.write(content)
