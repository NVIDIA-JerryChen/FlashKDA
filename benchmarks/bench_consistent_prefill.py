"""Paired full-API prefill comparison: fused, T1 replay, generic and FlashKDA."""

import gc
import json
from pathlib import Path
import re
import statistics
import time

import torch
import flash_kda
from tests.consistency.flashinfer_reference import compiled_reference


def progress(phase, **kw):
    print(json.dumps(dict(phase=phase, wall=time.time(), **kw)), flush=True)


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def capture(call, path):
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        call()
    graph.instantiate()
    graph.debug_dump(str(path))
    dot = path.read_text()
    audit = dict(
        kernel_nodes=len(re.findall(r'label="\{\s*KERNEL\b', dot)),
        memcpy_nodes=len(re.findall(r'label="\{\s*MEMCPY\b', dot)),
        recurrent_nodes=dot.count("kernel_cutlass__decode_kernel"),
    )
    return graph, audit


def measure(calls, samples, repeats, warmup=3):
    for call in calls.values():
        for _ in range(warmup):
            call()
    torch.cuda.synchronize()
    raw = {n: dict(gpu_ms=[], wall_ms=[]) for n in calls}
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for sample in range(samples):
        for name in list(calls) if sample % 2 == 0 else list(calls)[::-1]:
            torch.cuda.synchronize()
            wall = time.perf_counter()
            begin.record()
            for _ in range(repeats):
                calls[name]()
            end.record()
            end.synchronize()
            raw[name]["gpu_ms"].append(begin.elapsed_time(end) / repeats)
            raw[name]["wall_ms"].append((time.perf_counter() - wall) * 1000 / repeats)
    return {
        n: {
            k: dict(median=statistics.median(v), min=min(v), max=max(v), samples=v)
            for k, v in record.items()
        }
        for n, record in raw.items()
    }


def run_case(h, b, length, output, smoke):
    label = f"h{h}-b{b}-l{length}"
    progress("prefill_fixture", case=label)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype)

    xs = [*[rand((b, length, h, 128)) for _ in range(4)], rand((b, length, h))]
    a = torch.zeros(h, device="cuda")
    bias = rand((h, 128), torch.float32)
    initial = (rand((b, h, 128, 128), torch.float32) * 0.1).bfloat16()
    initial_check = initial.clone()
    state = torch.empty_like(initial)
    out = torch.empty_like(xs[0])
    fstate, fout = torch.empty_like(initial), torch.empty_like(out)
    copies = []
    token_inputs = []
    for x in xs[:4]:
        if b == 1:
            token_inputs.append(x.view(length, b, 1, h, 128))
        else:
            packed = torch.empty((length, b, 1, h, 128), device="cuda", dtype=x.dtype)
            copies.append((packed, x.transpose(0, 1).unsqueeze(2)))
            token_inputs.append(packed)
    beta_stride = ((b * h + 15) // 16) * 16
    if b == 1 and beta_stride == b * h:
        packed_beta = xs[4].view(length, b, 1, h)
    else:
        beta_storage = torch.empty(
            (length, beta_stride), device="cuda", dtype=xs[4].dtype
        )
        packed_beta = beta_storage[:, : b * h].view(length, b, 1, h)
        copies.append((packed_beta, xs[4].transpose(0, 1).unsqueeze(2)))
    token_inputs.append(packed_beta)
    token_outputs = (
        out.view(length, 1, 1, h, 128)
        if b == 1
        else torch.empty((length, b, 1, h, 128), device="cuda", dtype=out.dtype)
    )
    unpack = token_outputs.squeeze(2).transpose(0, 1)
    progress(
        "prefill_prepare_plans",
        case=label,
        input_packs=len(copies),
        output_pack=int(b != 1),
    )
    plans = []
    for token in range(length):
        plans.append(
            flash_kda.prepare_consistent_decode(
                *(x[token] for x in token_inputs),
                a,
                bias,
                state,
                token_outputs[token],
                state,
            )
        )
        if token and token % 8192 == 0:
            progress("prefill_plan_progress", case=label, tokens=token)

    def pack():
        for dst, src in copies:
            dst.copy_(src)

    def recurrent():
        state.copy_(initial)
        pack()
        for plan in plans:
            plan.run()
        if b != 1:
            out.copy_(unpack)

    fused = flash_kda.prepare_consistent_prefill(*xs, a, bias, initial, out, state)
    direct = flash_kda.prepare_consistent_prefill(
        *xs, a, bias, initial, out, state, mode="direct"
    )

    def flashkda():
        flash_kda.fwd(
            *xs,
            128**-0.5,
            fout,
            a,
            bias,
            -5.0,
            initial_state=initial,
            final_state=fstate,
        )

    progress("prefill_correctness", case=label)
    recurrent()
    # Independent generic recurrent implementation is an oracle only, never timed.
    offsets = torch.arange(b + 1, device="cuda", dtype=torch.int64) * length
    oracle_out, oracle_state = torch.empty_like(out), torch.empty_like(initial)
    canonical = [
        *[x.view(1, b * length, h, 128) for x in xs[:4]],
        xs[4].view(1, b * length, h),
    ]
    flash_kda.consistent_fwd(
        *canonical,
        a,
        bias,
        offsets,
        initial,
        oracle_out.view(1, b * length, h, 128),
        oracle_state,
    )
    exact(out, oracle_out)
    exact(state, oracle_state)
    recurrent_expected, recurrent_final = out.clone(), state.clone()
    for plan in (fused, direct):
        plan.run()
        exact(out, recurrent_expected)
        exact(state, recurrent_final)
    # Independent pinned FI checks every state and output in a 17-token prefix.
    fi = compiled_reference(
        torch.cuda.current_device(), torch.cuda.get_device_capability()
    )
    dummy = torch.zeros(1, device="cuda", dtype=torch.int32)
    fi_state, fi_out = initial.clone(), torch.empty_like(token_outputs[0])
    state.copy_(initial)
    for token in range(min(17, length)):
        plans[token].run()
        fi(
            *(x[token] for x in token_inputs),
            fi_state,
            fi_state,
            dummy,
            fi_out,
            a,
            bias.view(-1),
            dummy,
            dummy,
            dummy,
            128**-0.5,
            1e-6,
            -5.0,
        )
        exact(token_outputs[token], fi_out)
        exact(state, fi_state)
    flashkda()
    assert bool(torch.isfinite(fout).all()) and bool(torch.isfinite(fstate).all())
    # Compare the public wrapper to the same original extension with caller-owned workspace.
    import flash_kda_C

    workspace = torch.empty(
        flash_kda_C.get_workspace_size(b * length, h, b),
        device="cuda",
        dtype=torch.uint8,
    )
    flash_kda_C.fwd(
        *xs,
        128**-0.5,
        oracle_out,
        workspace,
        a,
        bias,
        -5.0,
        initial_state=initial,
        final_state=oracle_state,
    )
    exact(fout, oracle_out)
    exact(fstate, oracle_state)
    exact(initial, initial_check)
    del workspace, fi_state, fi_out

    def generic():
        flash_kda.consistent_fwd(
            *canonical,
            a,
            bias,
            offsets,
            initial,
            oracle_out.view(1, b * length, h, 128),
            oracle_state,
        )

    calls = dict(
        fused=fused.run,
        direct=direct.run,
        replay=recurrent,
        generic=generic,
        flashkda=flashkda,
    )
    graphs, audits = {}, {}
    for name, call in calls.items():
        progress("prefill_capture", case=label, implementation=name)
        graphs[name], audits[name] = capture(call, output / f"{label}-{name}.dot")
    assert audits["replay"]["recurrent_nodes"] == length, audits
    assert audits["replay"]["kernel_nodes"] >= length
    assert audits["replay"]["kernel_nodes"] + audits["replay"]["memcpy_nodes"] == (
        length + 1 + len(copies) + int(b != 1)
    ), audits
    assert audits["fused"]["kernel_nodes"] == len(fused._calls), audits
    assert audits["fused"]["memcpy_nodes"] == 0, audits
    assert (
        audits["direct"]["kernel_nodes"] == 1 and audits["direct"]["memcpy_nodes"] == 0
    ), audits
    # Dense batched FlashKDA has three nodes; the packed-varlen path has a fourth.
    assert audits["flashkda"] == dict(
        kernel_nodes=3, memcpy_nodes=0, recurrent_nodes=0
    ), audits
    # Validate the captured complete call, including initialization and layout copies.
    graphs["fused"].replay()
    graphs["flashkda"].replay()
    torch.cuda.synchronize()
    exact(out, recurrent_expected)
    exact(state, recurrent_final)
    exact(initial, initial_check)
    del recurrent_expected, recurrent_final
    progress("prefill_measure", case=label, audits=audits)
    graph = measure(
        {n: g.replay for n, g in graphs.items()}, 3 if smoke else 9, 1 if smoke else 5
    )
    eager = measure(calls, 3 if smoke else 9, 1, warmup=1)
    result = dict(
        h=h,
        b=b,
        length=length,
        graph=graph,
        eager=eager,
        audit=audits,
        replay_kernel_calls=length,
        input_pack_calls=len(copies),
        output_pack_calls=int(b != 1),
        state_initialization_calls=1,
        ratio=graph["fused"]["gpu_ms"]["median"]
        / graph["flashkda"]["gpu_ms"]["median"],
        speedup_vs_replay=graph["replay"]["gpu_ms"]["median"]
        / graph["fused"]["gpu_ms"]["median"],
        speedup_vs_generic=graph["generic"]["gpu_ms"]["median"]
        / graph["fused"]["gpu_ms"]["median"],
        workspace_bytes=fused.workspace_bytes,
        correctness="All recurrent outputs/final state bitwise match independent generic oracle; first 17 token outputs/states match pinned FI; FlashKDA wrapper/core match. No cross-contract bitwise claim.",
    )
    (output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(
        "prefill_complete",
        case=label,
        times={n: v["gpu_ms"]["median"] for n, v in graph.items()},
        ratio=result["ratio"],
    )
    return result


def main(output, smoke):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261105)
    cases = (
        [(6, 8, 8192)]
        if smoke
        else [(6, 1, 8192), (6, 8, 8192), (6, 64, 8192), (12, 1, 4096), (12, 1, 32768)]
    )
    records = []
    for h, b, length in cases:
        records.append(run_case(h, b, length, output, smoke))
        gc.collect()
        torch.cuda.empty_cache()
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                records=records,
                protocol="Canonical B,L,H,D inputs and outputs. Complete calls include coefficient generation and scanning for fused prefill; packing, unpacking and initialization for T1 replay; metadata helpers for generic; all FlashKDA stages. Fixed-address warm cache, no manual clock lock, 9 alternating paired samples. JIT/binding/plan-owned workspace allocation outside Graph replay. Eager convenience-wrapper allocations are included. Distinct FlashKDA numerical contract; no cross-contract bitwise claim.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.output, args.smoke)
