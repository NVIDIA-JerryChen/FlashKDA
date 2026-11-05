"""Dense single-token decode with caller-owned in-place continuation state."""

import torch
from flash_kda import prepare_consistent_decode

with torch.inference_mode():
    b, h = 4, 12
    q, k, v, g = [
        torch.randn((b, 1, h, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn((b, 1, h), device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(h, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((h, 128), device="cuda", dtype=torch.float32)
    state = torch.zeros((b, h, 128, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    plan = prepare_consistent_decode(
        q, k, v, g, beta, a_log, dt_bias, state, out, state
    )
    plan.run()
    torch.cuda.synchronize()
    print(out.shape, state.shape)
