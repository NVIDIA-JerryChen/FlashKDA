"""Multi-node launcher using an owned shared FileStore, with no port probing."""
import argparse
import os
import runpy
import sys
import torch.multiprocessing as mp


def worker(local_rank, args, extra):
    os.environ.update(LOCAL_RANK=str(local_rank), LOCAL_WORLD_SIZE=str(args.local_world_size),
                      RANK=str(args.node_rank * args.local_world_size + local_rank),
                      WORLD_SIZE=str(args.nodes * args.local_world_size),
                      FLASH_KDA_FILE_STORE=os.path.abspath(args.store))
    sys.argv = [args.module] + extra
    runpy.run_module(args.module, run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, required=True)
    parser.add_argument("--store", required=True,
                        help="Unique, initially absent path on a shared filesystem supporting locking")
    parser.add_argument("--module", default="tests.consistency.distributed")
    args, extra = parser.parse_known_args()
    if args.nodes < 1 or args.local_world_size < 1 or not 0 <= args.node_rank < args.nodes:
        raise ValueError("invalid node/rank configuration")
    mp.spawn(worker, args=(args, extra), nprocs=args.local_world_size, join=True)


if __name__ == "__main__":
    main()
