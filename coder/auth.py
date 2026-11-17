"""Bearer key on every route of the vLLM server, not only /v1/*.

vLLM's own key check (--api-key / VLLM_API_KEY) covers paths under /v1 only: routes such as
/invocations (a chat route), /tokenize or /metrics answer anyone (docs.vllm.ai, "Security").
This ASGI middleware closes that gap. vLLM loads it with

    vllm serve ... --middleware auth.BearerEverywhere      # auth.py on PYTHONPATH

and it then requires `Authorization: Bearer <VLLM_API_KEY>` on every HTTP and WebSocket
request, except `/health` (readiness probes) and CORS preflight (OPTIONS). Anything else is
refused by default, so a new vLLM route cannot be exposed by accident.
"""

import hmac
import json
import os

OPEN_PATHS = frozenset({"/health"})


class BearerEverywhere:
    def __init__(self, app):
        self.app = app
        key = os.environ.get("VLLM_API_KEY", "")
        if not key:
            raise RuntimeError("VLLM_API_KEY is not set: refusing to start a server without a key.")
        self.expected = f"Bearer {key}".encode()

    def _allowed(self, scope) -> bool:
        if scope.get("method") == "OPTIONS" or scope.get("path") in OPEN_PATHS:
            return True
        sent = next((v for k, v in scope.get("headers", []) if k == b"authorization"), b"")
        return hmac.compare_digest(sent, self.expected)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self._allowed(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # policy violation
            return
        body = json.dumps({"error": {"message": "Missing or invalid API key", "code": 401}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
