"""Check a deployed coder endpoint: health, models, tool calling, coding answer, throughput.

  export CODER_URL=https://<workspace>--nemotron-coder-super-rtx-server.us-west.modal.direct
  export NEMOTRON_API_KEY=...
  python coder/client.py
  python coder/client.py "Write a TensorRT-LLM build command for a FP8 Llama checkpoint." --think
  python coder/client.py "Document an InfiniBand cluster, diagrams in Mermaid." > doc.md   # answer only

stdout carries the answer only; checks, reasoning and stats go to stderr.

Works against any OpenAI-compatible server: Modal, an on-prem vLLM (onprem/docker-compose.yml),
or build.nvidia.com (CODER_URL=https://integrate.api.nvidia.com, key = NVIDIA_API_KEY).
Endpoint deployed with CODER_PROXY_AUTH=1: also export MODAL_KEY and MODAL_SECRET (a Modal proxy
token); they are sent as Modal-Key / Modal-Secret headers.
"""

import argparse
import json
import os
import sys
import time

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

MINUTES = 60  # seconds
# If the vLLM engine dies mid-answer (GPU fault), Modal replaces the container: wait and retry.
STREAM_RETRIES = 2

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command on the developer's machine and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def wait_for_server(client: OpenAI, timeout: int = 40 * MINUTES) -> str:
    """Return the served model id; retry while Modal answers 503 (cold start)."""
    deadline = time.time() + timeout
    while True:
        try:
            return client.models.list().data[0].id
        except (APIConnectionError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status not in (None, 502, 503, 504) or time.time() > deadline:
                raise
            print(f"server starting ({status or 'no connection'}), retrying in 15 s...", file=sys.stderr)
            time.sleep(15)


def check_tool_call(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Which NVIDIA GPUs are on this machine? Use the tool."}],
        tools=TOOLS,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    calls = r.choices[0].message.tool_calls or []
    if not calls:
        sys.exit(f"FAIL tool calling: no tool call, got {r.choices[0].message.content!r}")
    try:
        args = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        sys.exit(f"FAIL tool calling: arguments are not valid JSON: {calls[0].function.arguments!r}")
    print(f"OK   tool calling: {calls[0].function.name}({args})", file=sys.stderr)


def stream_answer(client: OpenAI, model: str, prompt: str, think: bool, max_tokens: int | None = None,
                  progress: dict | None = None):
    """Stream the answer to stdout; reasoning and stats go to stderr, so `> file.md` keeps the answer only.

    progress["printed"] becomes True once part of the answer reached stdout."""
    # The reasoning trace counts against max_tokens: leave it room.
    max_tokens = max_tokens or (32768 if think else 16384)
    start = time.time()
    first = None
    n_chunks = 0
    finish = None
    # Model card sampling: reasoning 1.0 / 0.95 for agentic coding.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        reasoning = getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None)
        if reasoning:
            print(reasoning, end="", flush=True, file=sys.stderr)
        if reasoning or choice.delta.content:
            first = first or time.time()
            n_chunks += 1
        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
            if progress is not None:
                progress["printed"] = True
    end = time.time()
    print()
    if first:
        # one chunk ~ one token without speculative decoding; MTP can pack several per chunk
        print(f"\nTTFT {first - start:.1f} s, {n_chunks / max(end - first, 1e-6):.0f} chunks/s", file=sys.stderr)
    if finish == "length":
        print(f"WARN answer cut at max_tokens={max_tokens}: rerun with --max-tokens {max_tokens * 2}", file=sys.stderr)


def retriable(e: APIError) -> bool:
    """A dying engine or a cold start is worth a retry; a client error (wrong model, key, size) is not."""
    if isinstance(e, APIStatusError):
        return e.status_code >= 500
    return True  # connection lost, or an error event sent mid-stream by a dying engine


def with_retries(client: OpenAI, step, progress: dict | None = None):
    """Run step(); after a server-side failure wait for a new container and run it again."""
    for attempt in range(1, STREAM_RETRIES + 2):
        try:
            return step()
        except APIError as e:
            if attempt > STREAM_RETRIES or not retriable(e):
                raise
            print(f"\nserver error ({e.message}), retry {attempt}/{STREAM_RETRIES} "
                  "once a new container is up...", file=sys.stderr)
            time.sleep(15)  # let the dead container leave the pool before polling again
            wait_for_server(client)
            if progress and progress.get("printed"):
                print("\n\n--- answer restarted ---\n")  # stdout may be a file: mark the cut
                progress["printed"] = False


def proxy_headers() -> dict[str, str]:
    """Modal proxy token headers, for an endpoint deployed with CODER_PROXY_AUTH=1."""
    key, secret = os.environ.get("MODAL_KEY"), os.environ.get("MODAL_SECRET")
    return {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}


def run_checks(base_url: str, api_key: str, model: str | None, prompt: str, think: bool = False,
               max_tokens: int | None = None):
    client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=api_key, timeout=30 * MINUTES,
                    default_headers=proxy_headers() or None)
    served = wait_for_server(client)
    print(f"OK   server up, model {served}", file=sys.stderr)
    model = model or served
    with_retries(client, lambda: check_tool_call(client, model))
    progress = {"printed": False}
    with_retries(client, lambda: stream_answer(client, model, prompt, think, max_tokens, progress), progress)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", default="Write a CUDA kernel that adds two float arrays, with its launch code.")
    p.add_argument("--think", action="store_true", help="enable reasoning (slower, better on hard tasks)")
    p.add_argument("--model", help="model id (default: the first one the server lists)")
    p.add_argument("--max-tokens", type=int, help="answer budget (default 16384, 32768 with --think)")
    a = p.parse_args()

    base_url = os.environ.get("CODER_URL", "")
    api_key = os.environ.get("NEMOTRON_API_KEY")
    if not base_url or not api_key:
        sys.exit("Set CODER_URL and NEMOTRON_API_KEY first.")
    run_checks(base_url, api_key, a.model, a.prompt, a.think, a.max_tokens)


if __name__ == "__main__":
    main()
