#!/usr/bin/env python3
"""OpenAI-compatible Pi bridge with gated Kimi layer evidence and Hydra routing."""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import asyncio
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

KIMI_DIR = Path(__file__).resolve().parent
HERMES_ROOT = KIMI_DIR.parent.parent
REPORT_DIR = HERMES_ROOT / "reports" / "tuios" / "kimi-k3"
HYDRA_BASE_URL = os.environ.get("HYDRA_BASE_URL", "http://127.0.0.1:8090/v1").rstrip("/")
HYDRA_API_KEY = os.environ.get("HYDRA_API_KEY", "dummy")
HYDRA_HEAVY_BACKEND = os.environ.get("HYDRA_HEAVY_BACKEND", "").strip()
HYDRA_HEAVY_MODEL = os.environ.get("HYDRA_HEAVY_MODEL", "hydra-auto").strip()
FIRST_LAYER_EVIDENCE = REPORT_DIR / "first_layer_verification.json"

app = FastAPI(title="Kimi K3 First-Layer / Hydra Bridge", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "kimi-k3-moe"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-kimi-k3-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "kimi-k3-moe"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    router_metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Kimi K3 Inference & Real Reasoning Engine
# ---------------------------------------------------------------------------
def _extract_text_content(content: Any) -> str:
    """Safely extracts plain text from string or structured OpenAI/Pi payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content is not None else ""

def _configured_shard_dir() -> Optional[Path]:
    configured = os.environ.get("SHARD_DIR") or os.environ.get("KIMI_K3_CHECKPOINT")
    if not configured:
        return None
    candidate = Path(configured).expanduser().resolve()
    return candidate.parent if candidate.is_file() else candidate


def _first_layer_binary() -> Optional[Path]:
    names = (
        KIMI_DIR / "bin" / "test_real_layer.exe",
        KIMI_DIR / "bin" / "test_real_layer",
        KIMI_DIR / "build" / "bin" / "test_real_layer.exe",
        KIMI_DIR / "build" / "bin" / "test_real_layer",
    )
    return next((item for item in names if item.is_file()), None)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def local_first_layer_status() -> dict:
    shard_dir = _configured_shard_dir()
    binary = _first_layer_binary()
    evidence = _read_json(FIRST_LAYER_EVIDENCE)
    verified = bool(
        evidence
        and evidence.get("status") == "verified"
        and shard_dir
        and evidence.get("shard_dir") == str(shard_dir)
    )
    return {
        "configured": shard_dir is not None,
        "shard_dir": str(shard_dir) if shard_dir else None,
        "shard_dir_exists": bool(shard_dir and shard_dir.is_dir()),
        "runner": str(binary) if binary else None,
        "runner_available": binary is not None,
        "verified": verified,
        "evidence": str(FIRST_LAYER_EVIDENCE) if evidence else None,
        "mode": "released_kimi_layer_0" if verified else "not_active",
    }


def hydra_status(timeout: float = 2.5) -> dict:
    # This endpoint reports router capability without probing this bridge as a
    # local model, avoiding a circular Hydra -> bridge -> Hydra health check.
    url = f"{HYDRA_BASE_URL}/models/hydra-auto"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"online": 200 <= response.status < 300, "url": url, "http_status": response.status, "body": body}
    except Exception as error:
        return {"online": False, "url": url, "error": str(error)}

def route_heavy_work_to_hydra(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> dict:
    """Route inference to Hydra only; never impersonate unavailable local Kimi inference."""
    standard_messages = []
    for message in messages:
        content = _extract_text_content(message.content)
        item = {"role": message.role, "content": content}
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if content or message.tool_calls:
            standard_messages.append(item)
    if not standard_messages:
        raise RuntimeError("at least one non-empty message is required")

    headers = {
        "Authorization": f"Bearer {HYDRA_API_KEY}",
        "Content-Type": "application/json",
        "X-Hermes-Origin": "pi-kimi-compat-bridge",
        "X-Use-Workflow": "false",
        # The bridge itself is registered as Hydra's local backend. Excluding
        # it prevents a completion from recursively routing back into itself.
        "X-Exclude-Backends": "local",
    }
    if HYDRA_HEAVY_BACKEND:
        headers["X-Force-Backend"] = HYDRA_HEAVY_BACKEND
    request = urllib.request.Request(
        f"{HYDRA_BASE_URL}/chat/completions",
        data=json.dumps({
            "model": HYDRA_HEAVY_MODEL,
            "messages": standard_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **({"tools": tools} if tools else {}),
            **({"tool_choice": tool_choice} if tool_choice is not None else {}),
        }).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Hydra HTTP {error.code}: {detail}") from error
    except Exception as error:
        raise RuntimeError(f"Hydra unavailable: {error}") from error

    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if (not isinstance(content, str) or not content.strip()) and not tool_calls:
        raise RuntimeError("Hydra returned neither assistant content nor tool calls")
    return {
        "content": content.strip() if isinstance(content, str) else "",
        "message": {
            "role": "assistant",
            "content": content,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        },
        "finish_reason": choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop"),
        "upstream": payload,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    hydra = hydra_status()
    layer = local_first_layer_status()
    return {
        "status": "ok" if hydra["online"] else "degraded",
        "service": "pi-kimi-compat-hydra-bridge",
        "port": int(os.environ.get("KIMI_K3_PORT", 8095)),
        "inference_backend": "hydra-router" if hydra["online"] else None,
        "hydra_heavy_backend": HYDRA_HEAVY_BACKEND,
        "hydra_heavy_model": HYDRA_HEAVY_MODEL,
        "hydra": hydra,
        "local_first_layer": layer,
        "truthful_mode": True,
    }


@app.get("/v1/local-layer/status")
async def first_layer_status():
    return local_first_layer_status()


@app.post("/v1/local-layer/verify")
async def verify_first_layer():
    status = local_first_layer_status()
    shard_dir = _configured_shard_dir()
    runner = _first_layer_binary()
    if not shard_dir or not shard_dir.is_dir():
        raise HTTPException(status_code=412, detail="SHARD_DIR/KIMI_K3_CHECKPOINT does not resolve to a checkpoint directory")
    if not runner:
        raise HTTPException(status_code=412, detail="test_real_layer binary is not built")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    c_run = subprocess.run(
        [str(runner), str(shard_dir), "0", "4", "8"],
        cwd=REPORT_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    verifier = KIMI_DIR / "tools" / "verify_real_layer.py"
    dump = REPORT_DIR / "real_layer.json"
    py_run = None
    if c_run.returncode == 0 and dump.is_file() and verifier.is_file():
        py_run = subprocess.run(
            [sys.executable, str(verifier), str(shard_dir), str(dump)],
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    verified = bool(c_run.returncode == 0 and py_run and py_run.returncode == 0)
    evidence = {
        "status": "verified" if verified else "failed",
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "layer": 0,
        "shard_dir": str(shard_dir),
        "runner": str(runner),
        "c_exit_code": c_run.returncode,
        "reference_exit_code": py_run.returncode if py_run else None,
        "c_stdout_tail": c_run.stdout[-4000:],
        "c_stderr_tail": c_run.stderr[-4000:],
        "reference_stdout_tail": py_run.stdout[-4000:] if py_run else "",
        "reference_stderr_tail": py_run.stderr[-4000:] if py_run else "",
    }
    FIRST_LAYER_EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if not verified:
        raise HTTPException(status_code=422, detail=evidence)
    return evidence

@app.get("/v1/models")
async def list_models():
    layer = local_first_layer_status()
    return {
        "object": "list",
        "data": [
            {
                "id": "kimi-k3-moe",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pi-kimi-compat-hydra-bridge",
                "display_name": "Pi Kimi-compatible alias (Hydra-routed; local layer separately gated)",
                "local_kimi_layer_0_verified": layer["verified"],
                "upstream_model": HYDRA_HEAVY_MODEL,
                "upstream_backend": HYDRA_HEAVY_BACKEND
            }
        ]
    }

from fastapi.responses import StreamingResponse

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    loop = asyncio.get_running_loop()
    try:
        routed = await loop.run_in_executor(
            None,
            route_heavy_work_to_hydra,
            req.messages,
            req.max_tokens if req.max_tokens is not None else 2048,
            req.temperature if req.temperature is not None else 0.2,
            req.tools,
            req.tool_choice,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    content = routed["content"]
    tool_calls = routed["message"].get("tool_calls")
    prompt_len = sum(len(str(m.content)) for m in req.messages) // 4
    compl_len = len(content) // 4
    req_id = f"chatcmpl-kimi-k3-{uuid.uuid4().hex[:8]}"
    created_ts = int(time.time())

    if req.stream:
        async def event_generator():
            if tool_calls:
                tool_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [dict(call, index=index) for index, call in enumerate(tool_calls)],
                        },
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(tool_chunk)}\n\n"

            chunks = content.split(" ") if content else []
            for i, chunk in enumerate(chunks):
                text_piece = chunk if i == len(chunks) - 1 else chunk + " "
                chunk_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_piece, "role": "assistant" if i == 0 else None},
                            "finish_reason": None
                        }
                    ]
                }
                # Remove None fields from delta
                if chunk_data["choices"][0]["delta"]["role"] is None:
                    del chunk_data["choices"][0]["delta"]["role"]
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)

            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": routed["finish_reason"]
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return ChatCompletionResponse(
        id=req_id,
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=routed["message"],
                finish_reason=routed["finish_reason"]
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_len,
            completion_tokens=compl_len,
            total_tokens=prompt_len + compl_len
        ),
        router_metadata={
            "bridge": "pi-kimi-compat-hydra",
            "local_first_layer_verified": local_first_layer_status()["verified"],
            "upstream_backend": routed["upstream"].get("router_metadata", {}).get("backend", HYDRA_HEAVY_BACKEND),
            "upstream_model": routed["upstream"].get("model", HYDRA_HEAVY_MODEL),
            "hydra_fallback_used": routed["upstream"].get("router_metadata", {}).get("fallback_used", False),
        }
    )

if __name__ == "__main__":
    port = int(os.environ.get("KIMI_K3_PORT", 8095))
    layer = local_first_layer_status()
    print(f"[KIMI/HYDRA] Bridge su http://127.0.0.1:{port}; Hydra={HYDRA_BASE_URL}; heavy={HYDRA_HEAVY_BACKEND}/{HYDRA_HEAVY_MODEL}; local layer 0 verified={layer['verified']}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
