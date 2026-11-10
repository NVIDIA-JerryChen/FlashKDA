"""Device-only ordering of request segments; runtime metadata never enters keys."""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.utils import SmemAllocator

SEGMENT = 128

MAX_REQUESTS = 1023  # Maximum request count supported by shared metadata storage


@cute.kernel
def _schedule(
    cu: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    tasks: cute.Tensor,
    order: cute.Tensor,
    proportional: cutlass.Int32,
):
    # The first coefficient kernel (PDL) may launch now; it waits for this
    # grid's completion before reading any schedule output.
    cute.arch.griddepcontrol_launch_dependents()
    tid, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    B = cutlass.Int32(cu.shape[0] - 1)
    # Stage cu_seqlens once per CTA: the O(B) loops below then read shared memory
    # instead of issuing one dependent global load per request.
    smem = SmemAllocator()
    cus = smem.allocate_array(
        cutlass.Int32, num_elems=MAX_REQUESTS + 1, byte_alignment=16
    )
    for i in cutlass.range(tid, B + 1, 128):
        cus[i] = cu[i]
    cute.arch.sync_threads()
    begin = cus[b]
    length = cus[b + 1] - begin
    nb = (length + SEGMENT - 1) // SEGMENT
    offset = cutlass.Int32(0)
    total = cutlass.Int32(0)
    longer = cutlass.Int32(0)
    for i in cutlass.range(B):
        li = cus[i + 1] - cus[i]
        ni = (li + SEGMENT - 1) // SEGMENT
        if i < b:
            offset += ni
        total += ni
        # Longest-first CTA order: requests with more tokens (ties: lower index) first.
        if li > length:
            longer += 1
        if li == length:
            if i < b:
                longer += 1
    if tid == 0:
        prefix[b] = offset
        # Request record at its longest-first position: (index, first token,
        # length, first rank), so a scan CTA needs one 16-byte load.
        order[4 * longer] = b
        order[4 * longer + 1] = begin
        order[4 * longer + 2] = length
        order[4 * longer + 3] = offset
        if b == 0:
            prefix[B] = total
    for j in cutlass.range(tid, nb, 128):
        rank = cutlass.Int32(0)
        if proportional == 0:
            # Round-robin: round j holds segment j of every request with more
            # than j segments, in request order.
            for i in cutlass.range(B):
                ni = (cus[i + 1] - cus[i] + SEGMENT - 1) // SEGMENT
                rank += cute.min(ni, j)
                if i < b and ni > j:
                    rank += 1
        else:
            # Length-proportional: segments in order of (2j+1)/n (the middle of
            # segment j as a fraction of its request), ties by request index,
            # so every window advances each request by the same fraction and
            # the longest request never waits at a window boundary. Ranks stay
            # increasing in j within each request. Request i contributes its
            # segments j' with (2j'+1)*nb < (2j+1)*ni, i.e. the odd numbers
            # below ceil((2j+1)*ni/nb), plus an equal key when i < b.
            for i in cutlass.range(B):
                ni = (cus[i + 1] - cus[i] + SEGMENT - 1) // SEGMENT
                x = (2 * j + 1) * ni
                c = (x + nb - 1) // nb
                rank += cute.min(ni, c // 2)
                if i < b and x % nb == 0:
                    q = x // nb
                    if q % 2 == 1 and (q - 1) // 2 < ni:
                        rank += 1
        ranks[offset + j] = rank
        tasks[rank, 0] = b
        tasks[rank, 1] = j * SEGMENT
        tasks[rank, 2] = cute.min(SEGMENT, length - j * SEGMENT)


@cute.jit
def _launch(
    cu: cute.Tensor,
    prefix: cute.Tensor,
    ranks: cute.Tensor,
    tasks: cute.Tensor,
    order: cute.Tensor,
    proportional: cutlass.Int32,
    stream: cuda.CUstream,
):
    _schedule(cu, prefix, ranks, tasks, order, proportional).launch(
        grid=[cu.shape[0] - 1, 1, 1], block=[128, 1, 1], stream=stream
    )


def fake(shape):
    return cute.runtime.make_fake_compact_tensor(
        cute.Int32,
        shape,
        assumed_align=32,
        stride_order=tuple(reversed(range(len(shape)))),
    )


@functools.cache
def compiled_schedule(device, capability):
    b, n = (cute.sym_int() for _ in range(2))
    o = cute.sym_int()
    return cute.compile(
        _launch,
        fake((b,)),
        fake((b,)),
        fake((n,)),
        fake((n, 3)),
        fake((o,)),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cute.jit
def lower_bound(ranks, offset, n, target):
    lo = cutlass.Int32(0)
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if ranks[offset + mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cute.jit
def segment_range(ranks, offset, n, first_target, end_target):
    """(#ranks < first_target, #ranks < end_target) over one request's n
    increasing segment ranks, i.e. both lower_bound results. Up to 256
    segments: one warp ballot per 32 ranks (independent loads, no dependent
    chain); beyond: the two binary searches. Every lane gets the same pair."""
    lane = cute.arch.lane_idx()
    first = cutlass.Int32(0)
    end = cutlass.Int32(0)
    if n <= 256:
        for c in cutlass.range_constexpr(8):
            j = c * 32 + lane
            rank = cutlass.Int32(2147483647)
            if j < n:
                rank = ranks[offset + j]
            first += cute.arch.popc(cute.arch.vote_ballot_sync(rank < first_target))
            end += cute.arch.popc(cute.arch.vote_ballot_sync(rank < end_target))
    else:
        first = lower_bound(ranks, offset, n, first_target)
        end = lower_bound(ranks, offset, n, end_target)
    return first, end
