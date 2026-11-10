"""Run packed recurrent prefill and replay its prepared plan in a CUDA Graph."""

import torch
from flash_kda import prepare_unified_prefill


@torch.no_grad()
def main():
    torch.manual_seed(7)
    lengths, heads = [0, 8, 127, 257], 12
    total = sum(lengths)
    xs = [
        torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(1, total, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda")
    dt_bias = torch.zeros(heads, 128, device="cuda")
    state = torch.zeros(
        len(lengths), heads, 128, 128, device="cuda", dtype=torch.bfloat16
    )
    out, final = torch.empty_like(xs[0]), torch.empty_like(state)
    cu = torch.tensor([0, 0, 8, 135, total], device="cuda", dtype=torch.int32)
    plan = prepare_unified_prefill(
        *xs, beta, a_log, dt_bias, state, out, final, cu_seqlens=cu
    )
    plan.run()
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    print(
        f"output={tuple(out.shape)}, state={tuple(final.shape)}, workspace={plan.workspace_bytes} bytes"
    )


if __name__ == "__main__":
    main()
