"""Nemotron 3 coding agent backend on Modal: an OpenAI-compatible vLLM endpoint.

Pick a tier with CODER_TIER (read at deploy time, each tier is its own Modal app):

  super-rtx    Nemotron 3 Super 120B-A12B NVFP4 on 1x RTX PRO 6000   ~$3.03/h  (default, cheapest)
  super-b200   Nemotron 3 Super 120B-A12B NVFP4 on 1x B200           ~$6.25/h  (faster, more sessions)
  ultra-b200   Nemotron 3 Ultra 550B-A55B NVFP4 on 4x B200           ~$25/h    (SWE-bench Verified ~70%)

  modal deploy coder/serve.py                          # super-rtx
  CODER_TIER=ultra-b200 modal deploy coder/serve.py    # PowerShell: $env:CODER_TIER="ultra-b200"
  CODER_MTP=1 modal deploy coder/serve.py              # with speculative decoding (see MTP below)
  CODER_PROXY_AUTH=1 modal deploy coder/serve.py       # Modal proxy auth too (see Security below)
  modal run coder/serve.py                             # temporary copy + smoke test
  modal run coder/serve.py::download_weights           # optional: fill the weight cache first (CPU only)

Scale to zero: nothing is billed while idle; the first request after a pause pays the cold
start (weights from the huggingface-cache volume + CUDA graphs, a few minutes).

Secrets:
  huggingface-secret  HF_TOKEN=hf_xxx
  nemotron-api-key    NEMOTRON_API_KEY=<random key>

Security: vLLM itself checks the key on /v1/* only, so coder/auth.py is loaded as vLLM
middleware and requires it on every route except /health. With the default
unauthenticated=True, Modal's proxy still starts a GPU for any request that reaches the URL,
key or not; CODER_PROXY_AUTH=1 makes the proxy refuse requests without a Modal proxy token
(Modal-Key / Modal-Secret headers) before a container starts. Editors that cannot send extra
headers (Zed) then cannot reach the endpoint; client.py and OpenCode can.
"""

import os
import subprocess
import time
from pathlib import Path

import modal

MINUTES = 60  # seconds

# Flags from the vLLM recipes / model cards (recipes.vllm.ai, huggingface.co/nvidia).
COMMON_ARGS = [
    "--trust-remote-code",
    "--kv-cache-dtype", "fp8",
    "--enable-prefix-caching",  # agents resend the same long prefix every turn
    "--enable-chunked-prefill",
    "--reasoning-parser", "nemotron_v3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
]

# Each tier pins the model revision (Hugging Face commit, checked 2026-11-17): with
# --trust-remote-code, an unpinned start would run whatever code is on the repo's main branch.
# startup_minutes covers the first start (download + load + compile, see coder/README.md 3.3);
# run download_weights first to make every start a load-only start.
TIERS = {
    "super-rtx": {
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "revision": "ff433f5493e25d631c9f12b5d55c674229923d02",
        "gpu": "RTX-PRO-6000",
        "target_inputs": 8,
        "startup_minutes": 40,
        "args": [
            "--tensor-parallel-size", "1",
            "--max-model-len", "262144",
            "--gpu-memory-utilization", "0.92",
            "--max-num-seqs", "8",
            "--max-num-batched-tokens", "16384",
            "--mamba-cache-mode", "align",
            "--mamba-ssm-cache-dtype", "float32",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        ],
        "env": {"VLLM_FLOAT32_MATMUL_PRECISION": "high", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
    },
    "super-b200": {
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "revision": "ff433f5493e25d631c9f12b5d55c674229923d02",
        "gpu": "B200",
        "target_inputs": 32,
        "startup_minutes": 40,
        "args": [
            "--tensor-parallel-size", "1",
            "--max-model-len", "262144",
            "--gpu-memory-utilization", "0.92",
            "--max-num-seqs", "64",
            "--mamba-cache-mode", "align",
            "--mamba-ssm-cache-dtype", "float32",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        ],
        "env": {"VLLM_FLOAT32_MATMUL_PRECISION": "high", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
    },
    "ultra-b200": {
        "model": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",  # ~330 GB of weights
        "revision": "02462641f13d3af838b904f48195b9bb8a1e4ebc",
        "gpu": "B200:4",
        "target_inputs": 16,
        "startup_minutes": 75,  # first start: 15-40 min download + 5-10 min load, plus warmup
        "args": [
            "--tensor-parallel-size", "4",
            "--enable-expert-parallel",
            "--max-model-len", "262144",
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "16",
            "--max-num-batched-tokens", "32768",
            "--mamba-ssm-cache-dtype", "float16",
            "--mamba-backend", "flashinfer",
            "--enable-mamba-cache-stochastic-rounding",
            "--mamba-cache-philox-rounds", "5",
            "--speculative-config", '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}',
            "--model-loader-extra-config", '{"enable_multithread_load": true, "num_threads": 96}',
        ],
        "env": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    },
}

TIER_NAME = os.environ.get("CODER_TIER", "super-rtx")
if TIER_NAME not in TIERS:
    raise ValueError(f"CODER_TIER={TIER_NAME!r}: choose one of {', '.join(TIERS)}")
TIER = TIERS[TIER_NAME]
MODEL_NAME = TIER["model"]
MODEL_REVISION = TIER["revision"]
STARTUP_TIMEOUT = TIER["startup_minutes"] * MINUTES
PROXY_AUTH = os.environ.get("CODER_PROXY_AUTH", "0") == "1"

# MTP speculative decoding (the model drafts 3-5 tokens per step) is off by default: with
# vLLM 0.28 on RTX PRO 6000 it kills the engine with a GPU memory fault (Xid 31, "unspecified
# launch failure") at the first speculative step of a streamed chat request, reproduced with
# this file + client.py. Without it the same requests pass at ~88 tokens/s.
# CODER_MTP=1 turns it back on, e.g. to test a newer vLLM image.
MTP = os.environ.get("CODER_MTP", "0") == "1"
SERVE_ARGS = list(TIER["args"])
if not MTP:
    i = SERVE_ARGS.index("--speculative-config")
    del SERVE_ARGS[i : i + 2]

# The tier is fixed at deploy time: inside the container the image env carries it.
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.28.0-ubuntu2404")
    .entrypoint([])
    # Ubuntu 24.04 ships only python3; without `python` Modal cannot detect the interpreter.
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .env(
        {
            "CODER_TIER": TIER_NAME,
            "CODER_MTP": "1" if MTP else "0",
            "CODER_PROXY_AUTH": "1" if PROXY_AUTH else "0",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "NVIDIA_TF32_OVERRIDE": "1",
            "PYTHONPATH": "/opt/coder",  # vLLM imports auth.BearerEverywhere from here
            **TIER["env"],
        }
    )
    .add_local_file(Path(__file__).with_name("auth.py"), "/opt/coder/auth.py")
)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
API_KEY_ENV = "NEMOTRON_API_KEY"
api_key_secret = modal.Secret.from_name("nemotron-api-key", required_keys=[API_KEY_ENV])

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)
VLLM_CACHE_PATH = "/root/.cache/vllm"  # torch.compile artifacts: faster restarts

REGION = "us"
ROUTING_REGION = "us-west"
PORT = 8000

# 0: nothing billed when idle, the URL answers 503 "no upstreams available" until a container runs.
# 1: always on (billed 24/7). Stop it with: modal app stop nemotron-coder-<tier>
MIN_CONTAINERS = 0
# How long the GPU stays warm after the last request (billed). An agent turn sends requests
# seconds apart, so 5 min covers a session; a pause longer than that costs a ~2-3 min cold start.
SCALEDOWN_WINDOW = 5 * MINUTES

app = modal.App(f"nemotron-coder-{TIER_NAME}")

with vllm_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = STARTUP_TIMEOUT - 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise RuntimeError(f"vLLM server exited with code {rc}")
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health", timeout=10).raise_for_status()
            return
        except requests.exceptions.RequestException:
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def warmup():
    # Failures only slow down the first user request: log them, do not kill the container.
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "def fib(n):"}],
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {os.environ[API_KEY_ENV]}"}
    for _ in range(3):
        try:
            requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=90,  # warmup stays well inside the 5 min left after wait_ready
            ).raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"warmup request failed: {e}")


def vllm_cmd() -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",
        "--middleware",
        "auth.BearerEverywhere",  # coder/auth.py: the key on every route, not only /v1/*
        *COMMON_ARGS,
        *SERVE_ARGS,
    ]


@app.server(
    image=vllm_image,
    gpu=TIER["gpu"],
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, VLLM_CACHE_PATH: VLLM_CACHE_VOL},
    compute_region=REGION,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=1,  # caps the bill: one replica is plenty for a personal coding agent
    secrets=[hf_secret, api_key_secret],
    startup_timeout=STARTUP_TIMEOUT,  # first start: download (70-330 GB) + torch.compile + CUDA graphs
    port=PORT,
    routing_region=ROUTING_REGION,
    exit_grace_period=25,
    target_concurrency=TIER["target_inputs"],
    # Default: the proxy lets every request through (editors can only send the bearer key) and
    # coder/auth.py checks the key on every route. CODER_PROXY_AUTH=1: the proxy also wants a
    # Modal proxy token, so requests without one never start a GPU.
    unauthenticated=not PROXY_AUTH,
)
class Server:
    @modal.enter()
    def startup(self):
        # vLLM reads its key from VLLM_API_KEY: it stays out of the command line and the logs.
        env = {**os.environ, "VLLM_API_KEY": os.environ[API_KEY_ENV]}
        self.process = subprocess.Popen(vllm_cmd(), env=env)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(
    image=vllm_image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[hf_secret],
    timeout=120 * MINUTES,
)
def download_weights():
    """Fill the huggingface-cache volume with the tier's pinned weights on a CPU container, so
    the GPU never pays for the download: modal run coder/serve.py::download_weights"""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    HF_CACHE_VOL.commit()
    print(f"{MODEL_NAME}@{MODEL_REVISION[:7]} cached in {path}")


@app.local_entrypoint()
def test(prompt: str = "Write a CUDA kernel that adds two float arrays, with its launch code."):
    """Smoke test: waits for the server, then runs coder/client.py checks against it."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import client

    url = Server.get_url()
    if not url:
        raise RuntimeError("No URL for the server: deploy it first (modal deploy coder/serve.py).")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"export {API_KEY_ENV}=<key stored in the nemotron-api-key secret> first.")
    client.run_checks(url, key, MODEL_NAME, prompt)
