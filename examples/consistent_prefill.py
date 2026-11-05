"""Dense fused prefill with BF16 continuation state."""

import torch
from flash_kda import prepare_consistent_prefill


def main():
    device = "cuda"
    b, length, h = 2, 129, 12

    def rand(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * 0.2).to(dtype)

    q, k, v, g = [rand((b, length, h, 128)) for _ in range(4)]
    beta = rand((b, length, h))
    a = rand((h,), torch.float32)
    bias = rand((h, 128), torch.float32)
    initial = rand((b, h, 128, 128))
    state = torch.empty_like(initial)
    out = torch.empty_like(q)
    plan = prepare_consistent_prefill(q, k, v, g, beta, a, bias, initial, out, state)
    with torch.no_grad():
        plan.run()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    print(
        "output:",
        tuple(out.shape),
        "state:",
        tuple(state.shape),
        "workspace bytes:",
        plan.workspace_bytes,
    )


if __name__ == "__main__":
    main()
