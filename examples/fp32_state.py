"""Run FP32-state prefill and Q8 decode using only public APIs."""

import torch
from flash_kda import prepare_fp32_decode, prepare_fp32_prefill


@torch.no_grad()
def main():
    b, q, h = 2, 8, 12
    xs = [
        torch.randn(b, q, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    beta = torch.randn(b, q, h, device="cuda", dtype=torch.bfloat16)
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    initial = torch.zeros(b, h, 128, 128, device="cuda", dtype=torch.float32)
    output, terminal = torch.empty_like(xs[0]), torch.empty_like(initial)
    prefill = prepare_fp32_prefill(*xs, beta, a, bias, initial, output, terminal)
    prefill.run()
    checkpoints = torch.empty(b, q, h, 128, 128, device="cuda", dtype=torch.float32)
    decoded = torch.empty_like(output)
    decode = prepare_fp32_decode(
        *xs, beta, a, bias, initial, decoded, checkpoint_states=checkpoints
    )
    decode.run()
    assert torch.equal(output.view(torch.int16), decoded.view(torch.int16))
    assert torch.equal(
        terminal.view(torch.int32), checkpoints[:, -1].contiguous().view(torch.int32)
    )
    accepted = torch.tensor([0, 4], device="cuda")
    chosen = checkpoints[torch.arange(b, device="cuda"), (accepted - 1).clamp_min(0)]
    continuation = torch.where(
        (accepted == 0)[:, None, None, None], initial, chosen
    ).contiguous()
    print(
        f"Q8/prefill match; continuation state: {tuple(continuation.shape)}, {continuation.dtype}"
    )


if __name__ == "__main__":
    main()
