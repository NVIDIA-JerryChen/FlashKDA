#!/usr/bin/env python3
"""Assemble csrc/replay/replay_kernel.cuh from the prefill kernels (csrc/smxx) and tools/replay_kernel.cuh.in.

The K1 arithmetic (fwd_kernel1.cuh:265-511), K1's layouts / shared-memory members and the K2 MMA block
(fwd_kernel2.cuh:463-732) are copied byte for byte; only the edits below are applied, each must match exactly once
(so a change in the prefill kernel makes this script fail loudly instead of silently producing a stale decode).

    python tools/gen_replay_kernel.py            # regenerate
    python tools/gen_replay_kernel.py --check    # exit 1 if the committed file is stale (tests/test_replay_codegen.py)
"""
import argparse
import sys
from pathlib import Path

OUT = "csrc/replay/replay_kernel.cuh"


def lines(src, a, b):
    """1-based inclusive line range."""
    return src[a - 1:b]


def balanced(block, what):
    text = "\n".join(block)
    assert text.count("{") == text.count("}"), f"{what}: unbalanced braces ({text.count('{')} vs {text.count('}')})"


def edit(block, old, new, what):
    text = "\n".join(block)
    n = text.count(old)
    assert n == 1, f"{what}: expected 1 match, got {n}"
    return text.replace(old, new).split("\n")


def generate(tree: Path) -> str:
    k1 = (tree / "csrc/smxx/fwd_kernel1.cuh").read_text().split("\n")
    k2 = (tree / "csrc/smxx/fwd_kernel2.cuh").read_text().split("\n")

    # ---------------------------------------------------------------- K1 pieces
    k1_layouts = lines(k1, 5, 42)
    assert k1_layouts[0].startswith("template <int D, int CHUNK = 16>") and k1_layouts[1] == "struct K1Layouts {"
    k1_layouts[1] = ("struct ReplayK1Layouts {   // == K1Layouts (fwd_kernel1.cuh:5-42); "
                     "that header cannot be included in a second translation unit")

    k1_union = lines(k1, 54, 82)   # SharedStorageK1's union / beta / g_bf16-k_restored / dt_bias-g_total members
    assert k1_union[0].strip().startswith("// Phase A"), k1_union[0]
    assert "g_total" in "\n".join(k1_union[-3:]), k1_union[-3:]

    k1_math = lines(k1, 265, 511)
    balanced(k1_math, "K1 math")
    assert k1_math[0].strip() == "// --- QK L2 Normalization ---", k1_math[0]
    assert k1_math[-1].strip() == "inv_fwd_subst_fused_1warp(L_fp32, M_bf16, INV, compute_tid);", k1_math[-1]
    k1_math = edit(k1_math,
                   "int actual_len = min(CHUNK, seq_len - local_t * CHUNK);",
                   "int actual_len = r_end;  // [replay] the open chunk holds rows 0 .. r_end-1",
                   "actual_len")
    k1_math = edit(k1_math,
                   "int beta_smem_offset = (head_idx * T_total + int(bos) + local_t * CHUNK) & 7;",
                   "int beta_smem_offset = 0;  // [replay] beta rows are staged at offset 0 "
                   "(K1 used the TMA alignment shift)",
                   "beta_smem_offset")

    # ---------------------------------------------------------------- K2 piece
    k2_block = lines(k2, 463, 732)
    balanced(k2_block, "K2 block")
    assert k2_block[0].strip() == "{" and k2_block[-1].strip() == "}", (k2_block[0], k2_block[-1])
    k2_block = edit(k2_block,
                    "Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored."
                    "begin()), TransposedMMALayout{});",
                    "Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), "
                    "TransposedMMALayout{});  // [replay] single stage",
                    "k_restored_t")
    # Phase 6 (state update) only when the chunk is complete: wrap fwd_kernel2.cuh:659-731.
    p6_first, p6_last = 659 - 463, 731 - 463
    assert k2_block[p6_first].strip() == "// ======== Phase 6: s_acc update ========", k2_block[p6_first]
    assert k2_block[p6_last].strip() == "}", k2_block[p6_last]
    balanced(k2_block[p6_first:p6_last + 1], "K2 phase 6")
    assert k2_block[p6_last + 1].strip() == "}" and len(k2_block) == p6_last + 2, "phase 6 must end the K2 block"
    k2_block = (k2_block[:p6_first]
                + ["            if (close_chunk) {  // [replay] state update only when the chunk has all CHUNK rows"]
                + k2_block[p6_first:p6_last + 1]
                + ["            }  // [replay] close_chunk"]
                + k2_block[p6_last + 1:])

    template = (Path(__file__).resolve().parent / "replay_kernel.cuh.in").read_text()
    text = (template
            .replace("@@K1_LAYOUTS@@", "\n".join(k1_layouts))
            .replace("@@K1_UNION@@", "\n".join(k1_union))
            .replace("@@K1_MATH@@", "\n".join(k1_math))
            .replace("@@K2_BLOCK@@", "\n".join(k2_block)))
    assert "@@" not in text
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", nargs="?", default=str(Path(__file__).resolve().parent.parent), help="FlashKDA checkout")
    ap.add_argument("--check", action="store_true", help="fail if the committed file differs from the generated one")
    args = ap.parse_args()
    tree = Path(args.tree)
    text = generate(tree)
    out_path = tree / OUT
    if args.check:
        if not out_path.exists() or out_path.read_text() != text:
            print(f"{OUT} is out of date: run python tools/gen_replay_kernel.py", file=sys.stderr)
            sys.exit(1)
        print(f"{OUT} is up to date")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
