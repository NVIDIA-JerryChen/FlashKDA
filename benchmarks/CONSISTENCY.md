# Recurrent baseline timing

Run from the checkout on eight allocated B300/GB300 GPUs (eight per node, or four per node across two NVLink nodes):

```bash
torchrun --standalone --nproc-per-node=8 -m benchmarks.bench_consistency --heads 12 --length 129 --batch 4
```

Run acceptance tests first. Freeze the source/environment used for each record.
The harness verifies identical GPU models and all-pairs local NVLink topology and emits JSON with
software versions, workload, CP size and timing boundaries. It reports the maximum
rank time for local CUDA Graph computation, BF16 state relay, and eager CP execution.
Local computation includes API metadata checks. Relay includes rank-order waiting;
eager CP includes host dispatch, communication, waiting and computation. These
independent measurements need not add up. No speedup threshold is imposed.
Each rank records GPU UUID, driver, software versions, L2 capacity, source SHA256
and network settings. Matching source/software is required across ranks; local
NVLink is checked for the allocated GPUs by UUID. Keep the NCCL initialization
log to verify actual cross-node transport rather than inferring it from settings.
Clocks are left unchanged. `--artifact-dir PATH` saves the captured graph DOT and
the actual input-buffer inventory for each active rank and CP size.

Warm: 25 warmups followed by 100 calls inside one CUDA event pair. Cold: 20 samples,
each preceded by a 256 MiB read eviction and synchronization outside its event pair;
median latency is reported. Compilation, allocation, graph capture, initialization,
and eviction are excluded. Inputs and separate initial-state buffers stay fixed;
outputs are overwritten. CP receives reconstruct the same incoming state each run.
The transport-only measurement uses a separate buffer. Default workload is one
fixed configuration; change arguments explicitly for other measurements.
The cold protocol means read eviction before each sample; it is not a claim that
every memory access misses L2. Incoming CP state is produced by communication
within the timed operation. CP=1 relay is an empty-call timing with no transfer.

For two nodes, run on both nodes with the same unique shared store path (which must
not already exist), substituting node rank 0 or 1:

```bash
python -m tests.consistency.launch --nodes 2 --node-rank 0 --local-world-size 4 \
  --store /shared/run-unique.store --module benchmarks.bench_consistency
```

The shared filesystem must support file locking. Each process sees only its four
allocated GPUs. Use another unique store path for each new run. Record the
inter-node network alongside local NVLink topology; cross-node communication is
not assumed to use NVLink. Two-node timings are not interchangeable with single-node timings.
