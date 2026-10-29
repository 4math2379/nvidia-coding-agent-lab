"""Validate the InfiniBand / RDMA fabric between two Modal nodes before serving across them.

Runs an NCCL all-reduce benchmark with torchrun over 2 nodes x 8 GPUs (16 GPUs), with RDMA on,
and prints the bus bandwidth plus the transport NCCL picked. Run it once, it lasts a few minutes:

  modal run coder/ib_bench.py                     # 2x H100:8, ~$63/h while it runs (~$3-5 total)
  modal run coder/ib_bench.py --gpu B200          # 2x B200:8

What to look for in the output:
  "NET/IB" + "GDRDMA"   InfiniBand with GPUDirect RDMA: GPU memory goes straight to the NIC.
  "NET/Socket"          TCP fallback: RDMA is not active, multi-node serving will crawl.
  busbw                 NCCL bus bandwidth in GB/s. 3,200 Gbps of RDMA per node is 400 GB/s raw;
                        expect a large fraction of that for 1 GiB messages.

The same script runs on-prem: torchrun --nnodes 2 --node_rank <r> --nproc_per_node 8
--master_addr <rank-0 ip> --master_port 29500 allreduce.py (see coder/README.md).
"""

import os
import subprocess

import modal

N_NODES = 2
GPUS_PER_NODE = 8  # Modal clusters take full nodes
MASTER_PORT = 29500

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.28.0-ubuntu2404")  # torch + NCCL, shared with serve.py
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Ubuntu 24.04: lets Modal find Python
    .apt_install("libibverbs1", "ibverbs-providers", "ibverbs-utils")  # libibverbs.so.1, libmlx5.so.1, ibv_devinfo
)

app = modal.App("nemotron-ib-bench")

ALLREDUCE = r'''
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)

for size_mb in (8, 64, 256, 1024):
    n = size_mb * 1024 * 1024 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    iters = 20
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    algbw = x.numel() * x.element_size() / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world  # nccl-tests convention for all-reduce
    if rank == 0:
        print(f"all_reduce {size_mb:5d} MiB  {dt * 1e3:8.2f} ms  algbw {algbw:7.1f} GB/s  busbw {busbw:7.1f} GB/s", flush=True)
dist.destroy_process_group()
'''


def run_bench():
    cluster = modal.Cluster.from_context()
    rank = cluster.container_rank()
    master = cluster.container_ips(family="ipv4")[0]

    if rank == 0:
        subprocess.run("ibv_devinfo -l || true; nvidia-smi topo -m || true", shell=True)
    with open("/tmp/allreduce.py", "w") as f:
        f.write(ALLREDUCE)

    env = {
        **os.environ,
        "NCCL_DEBUG": "INFO",  # logs "via NET/IB/GDRDMA" or "via NET/Socket"
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
    }
    subprocess.run(
        [
            "torchrun",
            f"--nnodes={N_NODES}",
            f"--node_rank={rank}",
            f"--nproc_per_node={GPUS_PER_NODE}",
            f"--master_addr={master}",
            f"--master_port={MASTER_PORT}",
            "/tmp/allreduce.py",
        ],
        env=env,
        check=True,
    )


# One function per GPU type: the GPU is part of the decorator, fixed at import time.
@app.function(image=image, gpu=f"H100!:{GPUS_PER_NODE}", timeout=30 * 60)
@modal.clustered(size=N_NODES, rdma=True)
def bench_h100():
    run_bench()


@app.function(image=image, gpu=f"B200:{GPUS_PER_NODE}", timeout=30 * 60)
@modal.clustered(size=N_NODES, rdma=True)
def bench_b200():
    run_bench()


@app.local_entrypoint()
def main(gpu: str = "H100"):
    fn = {"H100": bench_h100, "B200": bench_b200}[gpu.upper()]
    fn.remote()
