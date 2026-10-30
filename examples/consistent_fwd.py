"""Minimal packed forward with caller-owned output and continuation state."""
import torch
from flash_kda import consistent_fwd


def main():
    torch.manual_seed(0)
    h, lengths = 12, [3, 0, 2]
    t = sum(lengths)
    q, k, v, g = [torch.randn(1, t, h, 128, device="cuda", dtype=torch.bfloat16)
                   for _ in range(4)]
    beta = torch.zeros(1, t, h, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(h, device="cuda")
    dt_bias = torch.zeros(h, 128, device="cuda")
    cu = torch.tensor([0, 3, 3, 5], device="cuda", dtype=torch.int64)
    state = torch.zeros(3, h, 128, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    with torch.no_grad():
        consistent_fwd(q, k, v, g, beta, a_log, dt_bias, cu, state, out, state)
    torch.cuda.synchronize()
    print(out.shape, state.shape)


if __name__ == "__main__":
    main()
