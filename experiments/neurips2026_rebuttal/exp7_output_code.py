import math, torch
from flash_sd_kde.reference import sample_gaussian_mixture_16d, silverman_bandwidth_nd
torch.backends.cuda.matmul.allow_tf32 = True
EPS = 1e-12
def torch_score(tr, h):
    inv_h2 = 1.0 / (h * h)
    xn = (tr * tr).sum(1, keepdim=True)
    d2 = torch.clamp(xn + xn.T - 2.0 * (tr @ tr.T), min=0)
    phi = torch.exp(-0.5 * d2 * inv_h2)
    return tr + 0.5 * h * h * ((phi @ tr) / (phi.sum(1, keepdim=True) + EPS) - tr) * inv_h2
xtr = sample_gaussian_mixture_16d(32768)
h = silverman_bandwidth_nd(xtr)
tr = torch.as_tensor(xtr, device="cuda")
c = torch.compile(torch_score)
c(tr, h)
torch.cuda.synchronize()
