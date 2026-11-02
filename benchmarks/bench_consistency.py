"""torchrun --standalone --nproc-per-node=8 -m benchmarks.bench_consistency."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import torch
import torch.distributed as dist
from cuda.bindings import runtime
from flash_kda import consistent_fwd
from flash_kda.continuation import cp_consistent_fwd
from tests.consistency.helpers import make_case, select


def provenance():
    """Collect every rank's identity before timing; never infer network transport."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    )
    devices = [tuple(x.strip() for x in line.split(",")) for line in inventory.splitlines()]
    uuid = "GPU-" + str(props.uuid).removeprefix("GPU-")
    identity = next((item for item in devices if item[1] == uuid), None)
    if identity is None:
        raise RuntimeError("cannot map the allocated CUDA GPU to its NVLink topology")
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(p for directory in ("flash_kda", "tests/consistency", "benchmarks")
                       for p in (root / directory).rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    versions = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                    tvm_ffi=importlib.metadata.version("apache-tvm-ffi"),
                    cuda_python=importlib.metadata.version("cuda-python"),
                    nccl=torch.cuda.nccl.version(), driver=identity[3])
    status, l2_bytes = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize, device)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError("cannot query GPU L2 capacity")
    return dict(rank=dist.get_rank(), host=socket.gethostname(), local_rank=device,
                gpu=props.name, uuid=uuid, physical_index=int(identity[0]),
                versions=versions, source_sha256=digest.hexdigest(),
                l2_bytes=l2_bytes, architecture=platform.machine(), user_site_disabled=bool(sys.flags.no_user_site),
                network_settings={key: os.getenv(key) for key in
                                  ("NCCL_NET", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA")})


def warm(fn):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000 / 100


def cold(fn, flush):
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(20):
        flush.sum()  # Read eviction, outside the timed region.
        torch.cuda.synchronize()
        dist.barrier()
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--length", type=int, default=129)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init = {}
    if os.getenv("FLASH_KDA_FILE_STORE"):
        init = dict(init_method="file://" + os.environ["FLASH_KDA_FILE_STORE"],
                    rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5), device_id=torch.device("cuda", torch.cuda.current_device()), **init)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        name, host = torch.cuda.get_device_name(), socket.gethostname()
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))
        if world != 8 or torch.cuda.device_count() != local_world or local_world not in (4, 8):
            raise RuntimeError("benchmark requires eight allocated GPUs, four or eight per NVLink node")
        if not any(x in name for x in ("B300", "GB300")) or "B300A" in name or host.startswith("galaxy-"):
            raise RuntimeError("benchmark requires an approved B300/GB300 node")
        topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        rows = {line.split()[0]: line.split()[1:] for line in topology.splitlines()
                if line.startswith("GPU") and "X" in line.split()}
        environment = provenance()
        environments = [None] * world
        dist.all_gather_object(environments, environment)
        indices = [item["physical_index"] for item in environments if item["host"] == host]
        def connected(i, j):
            row = rows.get(f"GPU{i}", [])
            return j < len(row) and (row[j] == "X" if i == j else row[j].startswith("NV"))
        if len(set(indices)) != local_world or not all(connected(i, j) for i in indices for j in indices):
            raise RuntimeError("all locally allocated GPUs must have verified NVLink connectivity")
        if len({item["gpu"] for item in environments}) != 1:
            raise RuntimeError("all ranks must use identical GPU models")
        if len({json.dumps(item["versions"], sort_keys=True) for item in environments}) != 1:
            raise RuntimeError("all ranks must use identical software and driver versions")
        if len({item["source_sha256"] for item in environments}) != 1:
            raise RuntimeError("all ranks must use identical frozen sources")
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(json.dumps(dict(node_environment=environment, topology=topology)), flush=True)
        if rank == 0:
            print(json.dumps(dict(local_world=local_world, nodes=world // local_world, gpu=name, torch=torch.__version__, cuda=torch.version.cuda,
                                  cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                                  tvm_ffi=importlib.metadata.version("apache-tvm-ffi"),
                                  ranks=environments, clock_control="unmodified", heads=args.heads,
                                  length=args.length, batch=args.batch)), flush=True)
        if args.artifact_dir:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
        case = make_case([args.length] * args.batch, args.heads)
        flush = torch.ones(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        for cp in (1, 2, 4, 8):
            group = dist.new_group(list(range(cp)), backend="nccl")
            active = rank < cp
            if active:
                spans = [(args.length * rank // cp, args.length * (rank+1) // cp)] * args.batch
                local, _ = select(case, [args.length] * args.batch, spans)
                out = torch.empty_like(local["q"])
                state = torch.empty_like(local["state_in"])
                def complete():
                    cp_consistent_fwd(**local, out=out, state_out=state, group=group)
                complete()
                # Capture only local compute; use separate input/output states.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    consistent_fwd(**local, out=out, state_out=state)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                if args.artifact_dir:
                    graph.enable_debug_mode()
                with torch.cuda.graph(graph):
                    consistent_fwd(**local, out=out, state_out=state)
                if args.artifact_dir:
                    graph.debug_dump(str(args.artifact_dir / f"cp{cp}-rank{rank}.dot"))
                    inputs = {key: dict(pointer=tensor.data_ptr(), bytes=tensor.numel()*tensor.element_size(),
                                        shape=list(tensor.shape), dtype=str(tensor.dtype))
                              for key, tensor in local.items() if isinstance(tensor, torch.Tensor)}
                    (args.artifact_dir / f"cp{cp}-rank{rank}-inputs.json").write_text(json.dumps(inputs, indent=2))
                transport = local["state_in"].clone()
                def relay():
                    if rank:
                        dist.recv(transport, src=rank-1, group=group)
                    if rank+1 < cp:
                        dist.send(transport, dst=rank+1, group=group)
                functions = dict(local_graph=graph.replay, state_relay=relay, cp_eager=complete)
            for metric in ("local_graph", "state_relay", "cp_eager"):
                for protocol in ("warm", "cold"):
                    dist.barrier()
                    # Inactive ranks participate in cold's per-sample barriers.
                    fn = functions[metric] if active else (lambda: None)
                    us = warm(fn) if protocol == "warm" else cold(fn, flush)
                    value = torch.tensor(us if active else 0., device="cuda")
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        print(json.dumps(dict(cp=cp, metric=metric, protocol=protocol,
                                              max_rank_us=value.item())), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
