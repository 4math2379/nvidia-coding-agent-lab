# Lab: NVIDIA on-prem coding agent, powered by Nemotron on Modal

Goal: an OpenAI-compatible endpoint serving an NVIDIA model (Nemotron 3) that is good enough
to drive a coding agent specialized in deploying AI on on-prem NVIDIA hardware
(CUDA, TensorRT-LLM, Dynamo, NIM, GPU/Network Operator, InfiniBand/NCCL). The same container
runs on Modal for the lab, then on your own servers.

```
 OpenCode / Aider / Continue          NVIDIA docs MCP (CUDA, Dynamo, AIPerf, Guardrails)
            |  OpenAI API + tool calls          ^
            v                                   |
   Modal  ── vLLM 0.28 ── Nemotron 3 Super / Ultra (NVFP4, prefix cache)
            |
   on-prem ── same vLLM image (docker compose) ── then Kubernetes + Dynamo + InfiniBand
```

## Contents

0. [The files](#0-the-files)
1. [Choosing where to run the model](#1-choosing-where-to-run-the-model)
2. [Prerequisites (one-time)](#2-prerequisites-one-time)
3. [Deploying the endpoint](#3-deploying-the-endpoint)
4. [Testing the endpoint](#4-testing-the-endpoint)
5. [Stopping, monitoring, and controlling cost](#5-stopping-monitoring-and-controlling-cost)
6. [Connecting the coding agent](#6-connecting-the-coding-agent)
7. [NVIDIA MCP servers and skills](#7-nvidia-mcp-servers-and-skills)
8. [Multi-node InfiniBand / RDMA](#8-multi-node-infiniband--rdma)
9. [Moving on-prem](#9-moving-on-prem)
10. [Troubleshooting](#10-troubleshooting)
11. [Advanced directions](#11-advanced-directions)

---

## 0. The files

| File | Type | How to run it | Role |
|---|---|---|---|
| [serve.py](serve.py) | Modal app | `modal deploy coder/serve.py` | vLLM endpoint, 3 tiers (`CODER_TIER`) |
| [client.py](client.py) | **local Python script** | `python coder/client.py` (**not** `modal run`) | Test: health, tool call, code answer, throughput |
| [ib_bench.py](ib_bench.py) | Modal app | `modal run coder/ib_bench.py` | 2-node NCCL all-reduce benchmark over RDMA/InfiniBand |
| [serve_cluster.py](serve_cluster.py) | Modal app | `modal deploy coder/serve_cluster.py` | Ultra BF16 on 2 nodes × 8 H200 over RDMA (experimental) |
| [zed/settings.json](zed/settings.json) | Zed config | merged into your Zed settings | Free NVIDIA API + Modal endpoint + NVIDIA MCP (section 6.0) |
| [opencode.json](opencode.json) + [AGENTS.md](AGENTS.md) | agent config | copied into your project | Modal model + NVIDIA MCP + on-prem specialist prompt |
| [onprem/docker-compose.yml](onprem/docker-compose.yml) | Docker | `docker compose -f ... up -d` | The same server on your machine |
| [../.mcp.json](../.mcp.json) | Claude Code config | automatic in this repo | NVIDIA MCP for Claude Code |

Simple rule: a file that contains `app = modal.App(...)` is run with `modal ...`;
`client.py` does not contain one, so it is run with `python`.

## 1. Choosing where to run the model

Modal prices as of October 2026, billed per second, nothing when no container is running
(Starter plan: $30 of free credit per month).

| Option | Model | GPU | Active cost | When |
|---|---|---|---|---|
| **build.nvidia.com (recommended for daily use)** | `nvidia/nemotron-3.5-super-vl-120b-a12b`, `nvidia/nemotron-3.5-lightning-30b-a3b` | none | free (quota, rate-limited) | Day-to-day coding in the IDE. Tested with tool calls: 0.4 s / 64 tok/s and 0.2 s / 98 tok/s. Ultra is also available there, but with ~200 s of waiting before the first token |
| `super-rtx` (default) | Nemotron 3 Super 120B-A12B NVFP4 | 1× RTX PRO 6000 96 GB | ~$3.03/h | Personal agent, 1 to 8 sessions, 256k context |
| `super-b200` | same | 1× B200 180 GB | ~$6.25/h | Faster, several agents in parallel |
| `ultra-b200` | Nemotron 3 Ultra 550B-A55B NVFP4 | 4× B200 | ~$25/h | Hard tasks: SWE-bench Verified ~70% versus ~60% for Super |
| `serve_cluster.py` | Ultra BF16 | 2 × 8 H200 over RDMA | ~$73/h | Multi-node InfiniBand rehearsal, not for daily use |

What a session costs: the container stays warm for 5 min after the last request
(`SCALEDOWN_WINDOW`), then shuts down. 2 h of work on `super-rtx` ≈ 2 h 05 min billed ≈ $6.3,
plus ~$0.15 per cold start. Keep Modal for the lab (on-prem, InfiniBand, model testing)
and the free API for day-to-day coding.

Consumer pre-Blackwell cards such as a GTX 1080 Ti (11 GB, Pascal) cannot serve these models:
no NVFP4/FP8 support and far too little memory. Such a card is still useful for learning CUDA,
Nsight and compute-sanitizer locally.

## 2. Prerequisites (one-time)

Commands are given for bash (WSL, Linux, macOS). The PowerShell equivalent is shown below
when the syntax differs.

### 2.1 Local tools

```bash
cd nvidia-coding-agent-lab          # or the path to your clone
python3 -m venv venv && source venv/bin/activate
pip install -U -r requirement.txt   # modal>=1.6.2, openai, requests, aiohttp
modal --version                     # must print 1.6.2 or later
```

`modal>=1.6.2` is required for `ib_bench.py` and `serve_cluster.py` (`modal.clustered`,
`modal.Cluster`).

### 2.2 Modal account

```bash
modal setup            # opens the browser, saves the token in ~/.modal.toml
modal profile current  # shows your workspace
```

### 2.3 Modal secrets

Two secrets, shared with `memo.py` and `omni.py`. The key names must be exact.

**Hugging Face token** (to download the weights; create a "read" token at
huggingface.co/settings/tokens):

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

**Endpoint API key**: every client will have to send it as `Authorization: Bearer ...`.
Generate it **and keep it**: Modal does not let you read back a secret's value.

```bash
export NEMOTRON_API_KEY=$(openssl rand -hex 32)
echo "$NEMOTRON_API_KEY"                      # keep it (password manager, .env)
modal secret create nemotron-api-key NEMOTRON_API_KEY=$NEMOTRON_API_KEY
```

PowerShell:

```powershell
$env:NEMOTRON_API_KEY = -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Max 256) })
$env:NEMOTRON_API_KEY
modal secret create nemotron-api-key NEMOTRON_API_KEY=$env:NEMOTRON_API_KEY
```

Lost the key or need to rotate it: recreate it with `--force`, then redeploy (containers read the
secret at startup):

```bash
modal secret create --force nemotron-api-key NEMOTRON_API_KEY=$(openssl rand -hex 32)
```

Check: `modal secret list` must show `huggingface-secret` and `nemotron-api-key`.

### 2.4 Local .env file (optional but handy)

Add to `.env` (ignored by git):

```bash
NEMOTRON_API_KEY=<your key>
CODER_URL=<the URL printed at deployment, step 3>
```

then, in every new terminal: `set -a; source .env; set +a`.

## 3. Deploying the endpoint

### 3.1 Default tier (super-rtx)

```bash
modal deploy coder/serve.py
```

What happens:

1. Modal builds the `vllm/vllm-openai:v0.28.0-ubuntu2404` image (≈ 2 min the first time,
   instant afterwards thanks to the cache).
2. The `nemotron-coder-super-rtx` app is created, **no GPU starts yet**
   (`MIN_CONTAINERS = 0`).
3. The output prints the server URL, of the form:

   ```
   https://<workspace>--nemotron-coder-super-rtx-server.us-west.modal.direct
   ```

   Copy it: this is your `CODER_URL` (without `/v1` at the end). It is also visible at
   https://modal.com/apps, in the app.

### 3.2 Other tiers

The tier is read **at deploy time**; each tier is a separate Modal app
(`nemotron-coder-<tier>`) with its own URL, so they can coexist.

```bash
CODER_TIER=super-b200 modal deploy coder/serve.py
CODER_TIER=ultra-b200 modal deploy coder/serve.py
```

PowerShell (the variable persists in the session: clear it afterwards):

```powershell
$env:CODER_TIER = "super-b200"; modal deploy coder/serve.py
Remove-Item Env:CODER_TIER
```

An unknown name stops immediately with `CODER_TIER='x': choose one of super-rtx, super-b200, ultra-b200`.

### 3.3 The first start (cold start)

The GPU only starts on the first request. The first time:

| Step | super-rtx / super-b200 | ultra-b200 |
|---|---|---|
| Downloading the weights into the `huggingface-cache` volume | ~70 GB, 5 to 15 min | ~330 GB, 15 to 40 min |
| Loading + torch compilation + CUDA graphs | 2 to 5 min | 5 to 10 min |
| Subsequent starts (weights and compilation cached) | 2 to 4 min | 5 to 10 min |

During this time the URL answers `503 {"error":"no upstreams available"}`: this is normal.
`startup_timeout` is 40 min. To follow the startup: `modal app logs -f nemotron-coder-super-rtx`.

### 3.4 Useful settings in serve.py

| Constant | Default | Effect |
|---|---|---|
| `MIN_CONTAINERS` | `0` | `1` = GPU on 24/7 (billed continuously), no more cold starts |
| `SCALEDOWN_WINDOW` | `5 * MINUTES` | How long the GPU stays warm after the last request |
| `max_containers` | `1` | Caps the bill: a single replica |
| `CODER_MTP` (deploy-time variable) | `0` | MTP speculative decoding, off by default because it crashes vLLM 0.28 on RTX PRO 6000 (section 10.1). `CODER_MTP=1 modal deploy coder/serve.py` to try it again with a newer vLLM image |
| `TIERS[...]["args"]` | official recipes | vLLM flags (context, `--max-num-seqs`, MTP...) |

After any change: `modal deploy coder/serve.py` again (same URL).

## 4. Testing the endpoint

### 4.1 With the local script (normal method)

`client.py` is a Python script that runs **on your machine** and calls the URL. It is **not**
run with `modal run` (otherwise: `module 'client' has no attribute 'app'`).

```bash
export CODER_URL=https://<workspace>--nemotron-coder-super-rtx-server.us-west.modal.direct
export NEMOTRON_API_KEY=<your key>
python coder/client.py
```

PowerShell:

```powershell
$env:CODER_URL = "https://<workspace>--nemotron-coder-super-rtx-server.us-west.modal.direct"
$env:NEMOTRON_API_KEY = "<your key>"
python coder/client.py
```

Flow and expected output:

```
server starting (503), retrying in 15 s...      <- repeated during the cold start
OK   server up, model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
OK   tool calling: run_shell({'command': 'nvidia-smi'})
__global__ void add(const float* a, ...)        <- the streamed answer
TTFT 0.7 s, 88 chunks/s
```

The answer goes to standard output, everything else (waiting, tests, reasoning, throughput,
warnings) goes to standard error. Redirecting with `>` therefore gives a file that contains
only the answer, while the messages stay visible in the terminal.

Options:

```bash
python coder/client.py "Write a NIM Dockerfile for an air-gapped cluster."     # your prompt
python coder/client.py "Explain NCCL_IB_HCA." --think                          # with reasoning
python coder/client.py "..." --max-tokens 32768                                 # longer answer
python coder/client.py --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  # force the id
python coder/client.py "Docs for an InfiniBand cluster for a SOC, diagrams in Mermaid." > cyber.md
```

Answer length: 16,384 tokens by default (32,768 with `--think`, because reasoning counts
against the budget). If the answer is cut off, the client prints
`WARN answer cut at max_tokens=...`: run it again with a larger `--max-tokens`. The model
does not produce images: ask for diagrams in **Mermaid** (rendered by GitHub, VS Code,
draw.io) or in ASCII.

`CODER_MTP` only affects deployment (`modal deploy`), not the client.

The same script also tests the free NVIDIA API or an on-prem server:

```bash
CODER_URL=https://integrate.api.nvidia.com NEMOTRON_API_KEY=$NVIDIA_API_KEY \
  python coder/client.py --model nvidia/nemotron-3.5-super-vl-120b-a12b
CODER_URL=http://localhost:8000 python coder/client.py
```

### 4.2 With the Modal entrypoint (alternative)

```bash
modal run coder/serve.py
```

This starts a **temporary copy** of the server (`-dev` suffix), independent of the deployed
app, with its own cold start, then runs the same tests. Use it to test a change before
deploying it; to test the already deployed app, use 4.1.

### 4.3 With curl

```bash
curl -i $CODER_URL/health                                   # 200 without a key
curl $CODER_URL/v1/models -H "Authorization: Bearer $NEMOTRON_API_KEY"
curl $CODER_URL/v1/chat/completions \
  -H "Authorization: Bearer $NEMOTRON_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
curl -i $CODER_URL/v1/models                                # must answer 401 (no key)
```

With reasoning enabled (`"enable_thinking": true`, the model's default), the reasoning arrives
in `reasoning_content` and the answer in `content`.

## 5. Stopping, monitoring, and controlling cost

```bash
modal app list                                   # app status (deployed / stopped, tasks)
modal container list                             # running containers (hence GPUs) = billed
modal app logs nemotron-coder-super-rtx          # last 100 lines
modal app logs -f nemotron-coder-super-rtx       # live follow
modal app stop nemotron-coder-super-rtx          # stops everything, the URL no longer answers
```

- **Billing**: only while a container is running. With `MIN_CONTAINERS = 0`, it shuts
  down by itself 5 min after the last request; `modal app stop` cuts it off immediately.
- **After a `stop`**: `modal deploy coder/serve.py` brings the app back into service (same URL).
- **The volumes** `huggingface-cache` and `vllm-cache` persist: the weights are downloaded
  only once. Storage cost is low; delete with `modal volume delete <name>` if needed.
- **Cap**: set a spending limit in the Modal workspace settings
  (Settings → Usage / Billing).
- **Where the money goes**: `modal billing summary` (or `--for "this month"`). Measured on
  2026-10-27 for this lab: ~$20 of deployed apps (the `serve.py` GPU, including the 7
  restarts caused by the MTP crash), $3.34 of ephemeral apps (reproduction tests),
  volume storage covered by the free storage allowance.
- **vLLM metrics** (tokens/s, cache): `$CODER_URL/metrics`, Prometheus format.

## 6. Connecting the coding agent

### 6.0 Zed

Zed talks to any OpenAI-compatible server. The `coder/zed/settings.json` file contains the
blocks to merge into your Zed settings (Windows: `%APPDATA%\Zed\settings.json`,
Linux/macOS: `~/.config/zed/settings.json`):

- `language_models.openai_compatible.nvidia`: the free build.nvidia.com API
  (Nemotron 3.5 Super-VL, Lightning, Ultra).
- `language_models.openai_compatible.nemotron-modal`: your Modal endpoint (replace
  `<workspace>`, keep `/v1` at the end).
- `context_servers`: the NVIDIA docs MCP servers (CUDA, Dynamo, AIPerf).
- `agent.default_model`: the agent's default model (optional).

API keys, never in `settings.json`: Agent panel → settings → `nvidia` provider →
paste the `nvapi-...` key; `nemotron-modal` provider → paste your `NEMOTRON_API_KEY`.
Alternative: environment variables `NVIDIA_API_KEY` and `NEMOTRON_MODAL_API_KEY` (the name
comes from the provider identifier), set on the **Windows** side if Zed runs on Windows:
variables from the WSL terminal do not reach it.

Usage:

1. Open the Agent panel and pick the model in the selector: "Nemotron 3.5 Super-VL
   (NVIDIA API, free)" for coding, "Lightning" for quick answers.
2. The Modal model only answers once the GPU has started: the first message after a pause
   waits 2 to 3 min (503 during the cold start). Reserve it for lab tests, then
   `modal app stop nemotron-coder-super-rtx`.
3. The MCP tools appear in the Agent panel; the CUDA MCP asks for an NVIDIA Developer
   sign-in the first time.
4. Project rules: Zed reads `AGENTS.md` at the project root. Copying `coder/AGENTS.md` into
   a project gives the agent the role of an NVIDIA on-prem engineer.

### 6.1 OpenCode (recommended)

This is the harness NVIDIA uses to measure Nemotron on SWE-bench.

```bash
npm i -g opencode-ai
cp coder/opencode.json coder/AGENTS.md /path/to/your/project/
cd /path/to/your/project
export CODER_URL=... NEMOTRON_API_KEY=...       # OpenCode reads {env:...} at startup
opencode
```

- `opencode.json` declares two providers: `nemotron` (your Modal endpoint) and
  `nvidia-build` (free API, key `NVIDIA_API_KEY`). To switch models: `/models` in
  OpenCode.
- For Ultra, select `Nemotron 3 Ultra (Modal)` **and** point `CODER_URL` at the URL of the
  `nemotron-coder-ultra-b200` app.
- The NVIDIA MCP servers are declared in the `mcp` section; the CUDA MCP asks for an
  NVIDIA Developer sign-in on the first call.
- `AGENTS.md` gives the agent its role as an NVIDIA on-prem engineer and forces it to check flags
  and versions in the MCP servers rather than from memory.

### 6.2 Other OpenAI-compatible clients

| Parameter | Value |
|---|---|
| Base URL | `$CODER_URL/v1` |
| Key | `$NEMOTRON_API_KEY` |
| Model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (or the Ultra id) |

Aider:

```bash
aider --openai-api-base $CODER_URL/v1 --openai-api-key $NEMOTRON_API_KEY \
      --model openai/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

Continue / Cline / Roo: "OpenAI compatible" provider with the three values above.

### 6.3 Claude Code

Claude Code speaks the Anthropic Messages API, not the OpenAI API: it needs a translation proxy
(LiteLLM) in front of vLLM. Possible, but OpenCode is the direct path.

## 7. NVIDIA MCP servers and skills

### 7.1 Public MCP servers (verified on 2026-10-24)

| Server | URL | Content |
|---|---|---|
| CUDA docs (official Nsight) | `https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs` | CUDA docs + samples. NVIDIA Developer account sign-in on the first call |
| Dynamo | `https://docs.nvidia.com/dynamo/_mcp/server` | Distributed serving, KV routing, disaggregated prefill/decode, recipes |
| AIPerf | `https://docs.nvidia.com/aiperf/_mcp/server` | LLM endpoint benchmarking |
| NeMo Guardrails | `https://docs.nvidia.com/nemo/guardrails/_mcp/server` | Guardrails |
| Brev | `https://docs.nvidia.com/brev/_mcp/server` | NVIDIA GPU cloud (disabled by default in `opencode.json`) |
| NemoClaw, Infra Controller | `https://docs.nvidia.com/{nemoclaw,infra-controller}/_mcp/server` | As needed |

- **Claude Code**: `.mcp.json` at the repo root declares them; Claude Code offers to
  enable them when you open the project. Global install:
  `claude mcp add --scope user --transport http nvidia-cuda-docs https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs`
- **Without MCP**: all NVIDIA docs on Fern expose `/llms.txt` and a Markdown version
  of each page (append `.md` to the URL), for example https://docs.nvidia.com/dynamo/llms.txt.
- **On-prem / sensitive data**: self-hosted version of the CUDA MCP,
  [Nsight Copilot Blueprint](https://github.com/NVIDIA-AI-Blueprints/nsight-copilot)
  (Ubuntu 22.04+, Docker Compose v2, NVIDIA Container Toolkit, 200 GB of disk).

### 7.2 Official NVIDIA skills

Full catalog: `npx skills add nvidia/skills --list` (400+ skills). The most useful here:

| Skill | For |
|---|---|
| `dynamo-recipe-runner` | Deploying a Dynamo recipe (model × backend × GPU) on Kubernetes |
| `dynamo-router-starter` | Round-robin / KV-aware / least-loaded routing |
| `dynamo-interconnect-check` | Checking NIXL/UCX/NCCL over RDMA/NVLink after deployment |
| `dynamo-troubleshoot` | Diagnosing a broken Dynamo deployment |
| `nvshmem-select-remote-transport` | Choosing the NVSHMEM inter-node transport |
| `doca-gpunetio-ib-write-bw` | Measuring GPU-driven RDMA WRITE bandwidth |
| `compute-sanitizer-diagnose-cuda-bugs` | Memory bugs / races in a CUDA kernel |
| `tilegym-cutile-python` | Writing cuTile kernels |
| `rag-blueprint` | Deploying the NVIDIA RAG Blueprint |
| `nemotron-customize` | Fine-tuning / RL of Nemotron |

```bash
npx skills add nvidia/skills --skill dynamo-recipe-runner --agent claude-code --global --yes
```

Restart the agent after installation.

## 8. Multi-node InfiniBand / RDMA

Modal Clusters (GA since October 2026):

- `@modal.clustered(size=N, rdma=True)` under `@app.function` or `@app.server`.
- Up to 32 nodes / 256 GPUs per cluster; 3,200 Gbps of RDMA per node (6,400 on B300).
- Full nodes only: `H100:8` is valid, `H100:4` is not.
- All nodes start together or not at all; if one node fails, the whole cluster is
  stopped.
- Modal sets the NCCL variables by itself. `experimental_options={"efa_disabled": True}` forces an
  InfiniBand Verbs host if a tool does not support EFA (at the cost of wait times).
- The image must provide `libibverbs.so.1` and `libmlx5.so.1`: the scripts install
  `libibverbs1` and `ibverbs-providers`.

### 8.1 Validate the network (do this first)

```bash
modal run coder/ib_bench.py              # 2 × H100:8, ~$63/h, a few minutes (~$3-5)
modal run coder/ib_bench.py --gpu B200   # 2 × B200:8
```

What you should see:

| In the output | Meaning |
|---|---|
| `ibv_devinfo -l` lists `mlx5_*` devices | The InfiniBand/RoCE cards are visible |
| `nvidia-smi topo -m` | GPU ↔ NIC topology (NVLink `NV18`, `PIX/PXB` to the NICs) |
| `NET/IB` ... `GDRDMA` | InfiniBand with GPUDirect RDMA: the GPU writes directly to the network card |
| `NET/Socket` | TCP fallback: RDMA inactive, multi-node will be very slow |
| `busbw` at 1024 MiB | NCCL bus bandwidth in GB/s; 3,200 Gbps ≈ 400 GB/s raw per node |

The script also works on-prem: copy the `ALLREDUCE` block into `allreduce.py`, then, on
each node:

```bash
torchrun --nnodes 2 --node_rank <0|1> --nproc_per_node 8 \
         --master_addr <rank 0 ip> --master_port 29500 allreduce.py
```

### 8.2 Serving across several nodes (experimental)

```bash
modal deploy coder/serve_cluster.py
CODER_URL=<printed url> python coder/client.py
modal app stop nemotron-coder-ultra-cluster        # ~$73/h: do not forget
```

- Ultra in BF16 (~1.1 TB) split as TP 8 within each node (NVLink) × PP 2 across nodes (RDMA).
- vLLM without Ray: rank 0 serves the API, rank 1 runs in `--headless` mode; Modal only sends
  HTTP traffic to rank 0.
- First start: download of ~1.1 TB (up to 60 min).
- Untested so far: run it only after 3, 4 and 8.1 are validated. In the logs, look for
  `NET/IB/GDRDMA`.

## 9. Moving on-prem

### 9.1 A single server

Prerequisites: NVIDIA driver ≥ 575, Docker + NVIDIA Container Toolkit, a Blackwell GPU with ≥ 96 GB
(RTX PRO 6000, B200, DGX Spark). On Hopper, use the FP8 checkpoint with TP 4.

```bash
export HF_TOKEN=hf_xxx NEMOTRON_API_KEY=$(openssl rand -hex 32)
docker compose -f coder/onprem/docker-compose.yml up -d
docker compose -f coder/onprem/docker-compose.yml logs -f      # wait for "Application startup complete"
CODER_URL=http://localhost:8000 python coder/client.py
```

Offline (air-gapped):

1. On a connected machine: `docker pull vllm/vllm-openai:v0.28.0-ubuntu2404` then
   `docker save` (or push it to your local registry).
2. `hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` then copy the cache.
3. On the server: `docker load`, restore the `hf-cache` volume, `HF_HUB_OFFLINE=1`.

### 9.2 Kubernetes cluster

1. **GPU Operator** (drivers, Container Toolkit, DCGM exporter).
2. **Network Operator** (OFED drivers, GPUDirect RDMA, InfiniBand / RoCE SR-IOV).
3. **KAI Scheduler** to start all the pods of a job together.
4. Validate the network with nccl-tests (`all_reduce_perf`): same `NET/IB/GDRDMA` criterion as in 8.1.

### 9.3 Distributed serving with Dynamo

- NVIDIA Dynamo v1.6. Official Nemotron 3 Super recipe: 2 TP4 + EP workers on 8× B200,
  KV-aware routing, MTP; 1,388 tok/s/GPU measured under agentic load on B200.
- Next step: disaggregated prefill and decode, with the KV cache moving over NIXL on RDMA.
  This is where InfiniBand really pays off.
- Skills: `dynamo-recipe-runner` then `dynamo-interconnect-check`.
- Measure before/after each tuning change with AIPerf (TTFT, ITL, tok/s/GPU).

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `module 'client' has no attribute 'app'` | `client.py` run with `modal run` | `python coder/client.py` (section 4.1) |
| `We were unable to determine the version of Python installed in the Image` | vLLM 0.28 images are based on Ubuntu 24.04: `python3` exists, `python` does not | Already fixed: the image runs `ln -sf /usr/bin/python3 /usr/bin/python`. Keep this line if you change the image |
| `503 {"error":"no upstreams available"}` | No container is running (cold start or idle app) | Wait: the client retries by itself. Follow with `modal app logs -f ...` |
| 503 lasting more than 40 min | Startup failed | `modal app logs nemotron-coder-super-rtx`: look for `Error`, `CUDA out of memory`, Hugging Face `401` |
| `401 Unauthorized` on `/v1/*` | Key missing or different from the one in the secret | Check `echo $NEMOTRON_API_KEY`; if lost, recreate the secret (2.3) and redeploy |
| `Set CODER_URL and NEMOTRON_API_KEY first.` | Variables not exported in this terminal | `export ...` or `set -a; source .env; set +a` |
| `404` on the URL | `/v1` added twice, or wrong app | `CODER_URL` **without** `/v1`; check the URL in `modal app list` / the dashboard |
| `openai.APIError: EngineCore encountered an issue` mid-answer | The vLLM engine crashed on the GPU: logs show `CUDA error: unspecified launch failure` + `Xid 31 ... MMU Fault` (see 10.1) | The deployment runs with MTP: redeploy without it (`modal deploy coder/serve.py`, MTP off by default). The client retries twice, but with MTP every retry crashes again |
| `FAIL tool calling: no tool call` | Tool parser missing or different model | Keep `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| Answer cut mid-sentence, `WARN answer cut at max_tokens=...` | Token budget reached | `--max-tokens 32768` (or more) |
| The container is still running after the answer | Normal: it stays warm for 5 min (`SCALEDOWN_WINDOW`) for the next question | Wait, or `modal app stop nemotron-coder-super-rtx` to cut it off right away |
| `CODER_TIER='x': choose one of ...` | Typo in the tier | `super-rtx`, `super-b200` or `ultra-b200` |
| `Secret 'huggingface-secret' not found` | Secret missing in this workspace | Section 2.3 |
| `AttributeError: module 'modal' has no attribute 'clustered'` | Modal < 1.6.2 | `pip install -U "modal>=1.6.2"` |
| The GPU stays on | `MIN_CONTAINERS = 1` or regular requests | `modal container list`, then `modal app stop ...` |
| Ultra: no GPU available | 4× B200 unavailable in the region | Retry later or remove `compute_region` in `serve.py` |
| `charmap codec can't encode` (Windows) | Windows console not in UTF-8 | `$env:PYTHONUTF8 = "1"` before the `modal` command |

Diagnostic commands:

```bash
modal profile current
modal secret list
modal volume ls huggingface-cache
modal app list
modal container list
modal app logs nemotron-coder-super-rtx
```

### 10.1 GPU crash "EngineCore encountered an issue" (analyzed on 2026-10-27)

Symptoms on `super-rtx` with MTP enabled:

1. The client prints `503`s (cold start), the tool-call test passes, then the
   streamed answer stops with `openai.APIError: EngineCore encountered an issue`. The
   client's retries crash again on each new container.
2. `modal app logs nemotron-coder-super-rtx --search Error` shows
   `torch.AcceleratorError: CUDA error: unspecified launch failure` in `EngineCore`.
3. The system logs (`modal app logs ... -s system`) show at the same moment
   `[gpu-health] Xid 31 ... MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_WRITE`.

Diagnosis:

- **Xid 31** = a CUDA kernel wrote to an unmapped GPU memory address. 7 crashes on
  7 different physical GPUs: this is not a faulty card (a hardware failure tends to produce
  Xid 48/79/94 instead), it is a software bug in vLLM 0.28.
- The vLLM dump (`dump_input.py`) places every crash at the **first MTP speculative decoding
  step** (`scheduled_spec_decode_tokens: [-1, -1, -1]`), right after the prompt prefill.
- Reproduced on Modal with the exact code of `serve.py` + `client.py` (RTX PRO 6000, `us` region,
  driver 610.57.04):

  | Config | Result |
  |---|---|
  | MTP enabled (`CODER_MTP=1`) | crashes on the very first streamed answer |
  | MTP disabled (`CODER_MTP=0`) | 3/3 succeeded, TTFT 0.1 s, ~88 tokens/s |

- Requests sent another way (`requests` library, same apparent parameters) did not
  crash with MTP: the trigger also depends on the exact shape of the request.
  An agent like OpenCode sends its own requests, so you should not count on that.

What is in place:

- **MTP is off by default** in `serve.py`. Redeploying is enough:
  `modal deploy coder/serve.py`.
- `client.py` retries the answer up to 2 times if the engine dies anyway; in an output file,
  the break is marked `--- answer restarted ---`.
- To retest MTP with a newer vLLM image: change the image tag in `serve.py`
  (keep the `ln -sf /usr/bin/python3 /usr/bin/python` line), deploy with `CODER_MTP=1`,
  then run `python coder/client.py`.

Check whether it is still this problem:

```bash
modal app logs nemotron-coder-super-rtx --since 2h --search "unspecified launch failure"
modal app logs nemotron-coder-super-rtx --since 2h -s system --search Xid
modal app logs nemotron-coder-super-rtx --since 2h --search "speculative_config"   # MTP active?
```

## 11. Advanced directions

- **Nemotron 3.5 Super-VL 120B-A12B** (released October 2026, NVFP4 available): compare it
  with Nemotron 3 Super on your tasks; its vLLM recipes were still young at release.
- **Specialize the model** for your on-prem stack: `nemotron-customize` skill (SFT/PEFT with
  NeMo AutoModel or Megatron-Bridge, then RL with NeMo RL / NeMo Gym), `-Base-BF16` checkpoint.
- **RAG over your internal docs**: NeMo Retriever + Nemotron 3.5 Embed/Rerank, exposed as an MCP
  server (`nemo-retriever-mcp`) so the agent can search your runbooks.
- **Guardrails**: NeMo Guardrails in front of the endpoint if other teams use it.

## Sources

- [Modal pricing](https://modal.com/pricing), [GPU](https://modal.com/docs/guide/gpu), [Multi-node clusters](https://modal.com/docs/guide/multi-node-clusters)
- [vLLM recipe for Nemotron 3 Super](https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), [Nemotron 3 Ultra NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
- [Nemotron 3 Super technical report](https://arxiv.org/abs/2604.12374)
- [Dynamo recipe for Nemotron 3 Super](https://docs.nvidia.com/dynamo/latest/recipes/nemotron-3-super)
- [vLLM multi-node](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
- [NVIDIA Nsight AI / CUDA MCP](https://developer.nvidia.com/nsight-ai), [NVIDIA skills](https://github.com/NVIDIA/skills)
