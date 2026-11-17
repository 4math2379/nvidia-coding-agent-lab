"""Nemotron 3 Ultra BF16 (full precision) served across 2 nodes x 8 H200 over RDMA, on Modal.

Why two nodes: the BF16 checkpoint is ~1.1 TB, more than one 8-GPU node can hold with room for
the KV cache. vLLM splits the model with tensor parallelism inside each node (NVLink) and pipeline
parallelism across nodes (InfiniBand/RoCE RDMA, 3,200 Gbps per node on Modal).

EXPERIMENTAL and EXPENSIVE: 16x H200 = ~$73/h while a replica runs. Validate the fabric first with
coder/ib_bench.py. For a personal coding agent, coder/serve.py (one GPU, ~$3/h) is the right tool;
this file is the multi-node / on-prem rehearsal.

  modal run coder/serve_cluster.py::download_weights  # once, CPU only: ~1.1 TB into the volume
  modal deploy coder/serve_cluster.py
  CODER_URL=<printed url> python coder/client.py
  modal app stop nemotron-coder-ultra-cluster        # stop billing

vLLM runs without Ray (multiprocessing backend): rank 0 serves the API, rank 1 runs --headless.
Only rank 0 receives HTTP traffic on Modal.

The weights are downloaded once, by download_weights, before any GPU starts: both nodes share
the huggingface-cache volume, and two vLLM processes downloading the same 1.1 TB into it at the
same time would race (vLLM's download lock only works within one host). The nodes then start
with HF_HUB_OFFLINE=1 and fail fast if the cache is missing.

Security: as in serve.py, coder/auth.py requires the key on every route (vLLM alone checks /v1/*
only) and CODER_PROXY_AUTH=1 adds Modal proxy auth, so anonymous hits cannot start 16 GPUs.
"""

import os
import subprocess
import time
from pathlib import Path

import modal

MINUTES = 60  # seconds

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_REVISION = "77df655d5e9f8362164ed14dd8b48f8bce657498"  # pinned Hugging Face commit (2026-11-17)
PROXY_AUTH = os.environ.get("CODER_PROXY_AUTH", "0") == "1"
N_NODES = 2
GPUS_PER_NODE = 8
GPU = f"H200:{GPUS_PER_NODE}"
PORT = 8000

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.28.0-ubuntu2404")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Ubuntu 24.04: lets Modal find Python
    .apt_install("libibverbs1", "ibverbs-providers")  # RDMA userspace: libibverbs.so.1, libmlx5.so.1
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "NCCL_DEBUG": "INFO",  # look for "via NET/IB/GDRDMA" in the logs
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
            "CODER_PROXY_AUTH": "1" if PROXY_AUTH else "0",
            "PYTHONPATH": "/opt/coder",  # vLLM imports auth.BearerEverywhere from here
        }
    )
    .add_local_file(Path(__file__).with_name("auth.py"), "/opt/coder/auth.py")
)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
API_KEY_ENV = "NEMOTRON_API_KEY"
api_key_secret = modal.Secret.from_name("nemotron-api-key", required_keys=[API_KEY_ENV])

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App("nemotron-coder-ultra-cluster")

with image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 40 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise RuntimeError(f"vLLM server exited with code {rc}")
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health", timeout=10).raise_for_status()
            return
        except requests.exceptions.RequestException:
            time.sleep(10)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


@app.server(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    min_containers=0,
    max_containers=N_NODES,  # counts nodes: one 2-node replica at most
    scaledown_window=10 * MINUTES,
    secrets=[hf_secret, api_key_secret],
    startup_timeout=45 * MINUTES,  # load only: the weights are already in the volume (download_weights)
    port=PORT,
    exit_grace_period=25,
    target_concurrency=32,
    unauthenticated=not PROXY_AUTH,  # coder/auth.py checks the key on every route
)
@modal.clustered(size=N_NODES, rdma=True)
class Server:
    @modal.enter()
    def startup(self):
        cluster = modal.Cluster.from_context()
        rank = cluster.container_rank()
        master = cluster.container_ips(family="ipv4")[0]

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--tokenizer-revision", MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--middleware", "auth.BearerEverywhere",  # coder/auth.py: the key on every route
            "--trust-remote-code",
            "--tensor-parallel-size", f"{GPUS_PER_NODE}",
            "--pipeline-parallel-size", f"{N_NODES}",
            "--nnodes", f"{N_NODES}",
            "--node-rank", f"{rank}",
            "--master-addr", master,
            "--kv-cache-dtype", "fp8",
            "--max-model-len", "262144",
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "64",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--mamba-ssm-cache-dtype", "float16",
            "--reasoning-parser", "nemotron_v3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--distributed-timeout-seconds", "3600",
        ]
        if rank == 0:
            cmd += ["--host", "0.0.0.0", "--port", f"{PORT}"]
        else:
            cmd += ["--headless"]  # contributes GPUs only; rank 0 owns the API server

        # Offline: the weights come from the volume filled by download_weights, never from the
        # Hub, so the two nodes cannot race on the same files.
        env = {**os.environ, "VLLM_API_KEY": os.environ[API_KEY_ENV], "HF_HUB_OFFLINE": "1"}
        self.process = subprocess.Popen(cmd, env=env)
        if rank == 0:
            wait_ready(self.process)
        else:
            time.sleep(30)  # fail fast on a broken worker instead of leaving rank 0 waiting
            if (rc := self.process.poll()) is not None:
                raise RuntimeError(f"vLLM headless worker exited with code {rc}")

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(image=image, volumes={HF_CACHE_PATH: HF_CACHE_VOL}, secrets=[hf_secret], timeout=6 * 60 * MINUTES)
def download_weights():
    """Download the pinned BF16 checkpoint (~1.1 TB) once, on a CPU container, before deploying:
    modal run coder/serve_cluster.py::download_weights"""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    HF_CACHE_VOL.commit()
    print(f"{MODEL_NAME}@{MODEL_REVISION[:7]} cached in {path}")
