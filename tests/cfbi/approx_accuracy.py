"""Accuracy of the three PTX approximations on this device (informational; used to size the error budget)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from flash_kda.cfbi import load_ext

ext = load_ext()
def err(name, x, y, ref):
    rel = ((y.double() - ref).abs() / ref.abs().clamp_min(1e-300)).max().item()
    print(f"{name:34s} max rel err {rel:.3e}  (2^{torch.log2(torch.tensor(rel)).item():.1f})")

x = torch.linspace(-20, 20, 4_000_001, device="cuda")
err("tanh.approx   x in [-20,20]", x, ext.probe_transc(x, 1), torch.tanh(x.double()))
xs = torch.linspace(-3, 3, 4_000_001, device="cuda")
err("tanh.approx   x in [-3,3]", xs, ext.probe_transc(xs, 1), torch.tanh(xs.double()))
y = torch.linspace(-60, 5, 4_000_001, device="cuda")
err("ex2.approx    x in [-60,5]", y, ext.probe_transc(y, 0), torch.exp2(y.double()))
r = torch.logspace(-8, 8, 4_000_001, device="cuda")
err("rsqrt.approx  x in [1e-8,1e8]", r, ext.probe_transc(r, 2), 1 / torch.sqrt(r.double()))
# the gate as composed: sigmoid(x) = fma(tanh(x/2), .5, .5), absolute and relative error
g = torch.linspace(-12, 12, 4_000_001, device="cuda")
th = ext.probe_transc((g * 0.5).contiguous(), 1)
sig = (0.5 + th.double() * 0.5).float().double()
ref = torch.sigmoid(g.double())
print(f"sigmoid via tanh.approx            max abs err {(sig - ref).abs().max().item():.3e}  max rel err {((sig-ref).abs()/ref).max().item():.3e}")
