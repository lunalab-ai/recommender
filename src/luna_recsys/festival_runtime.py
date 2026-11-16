"""Free CPU runtime for W12A, with independent model preparation and no API key."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .llm_runtime import (
    CHAT_MODEL, EMBED_MODEL, MODEL_DIGESTS, OLLAMA_VERSION, LocalOllama, prepare_binary,
)


def start_service(cache: Path) -> LocalOllama:
    """Reuse the lesson's local service or start the checked portable runtime."""
    host = os.environ.get("LUNA_OLLAMA_HOST")
    marker = cache / "service.json"
    if not host and marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))["host"]
        try:
            LocalOllama(previous).request("version", timeout=2)
            host = previous
        except RuntimeError:
            pass
    if not host:
        binary = prepare_binary(cache)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        host = f"http://127.0.0.1:{port}"
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=str(cache / "models"),
                   OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NUM_PARALLEL="1",
                   CUDA_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
        with (cache / "server.log").open("ab") as log:
            process = subprocess.Popen([str(binary), "serve"], env=env, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        marker.write_text(json.dumps({"host": host, "pid": process.pid}), encoding="utf-8")
    client = LocalOllama(host)
    for attempt in range(45):
        try:
            version = client.request("version", timeout=2)["version"]
            break
        except RuntimeError:
            if attempt == 44:
                raise
            time.sleep(1)
    if version.lstrip("v") != OLLAMA_VERSION:
        raise RuntimeError(f"확인한 Ollama {OLLAMA_VERSION}이 필요합니다. 현재 {version}")
    return client


def ensure_model(client: LocalOllama, model: str, *, pull: bool = True) -> str:
    """Check one exact model digest. A live failure is returned to the notebook."""
    expected = MODEL_DIGESTS[model]
    tags = {row["name"]: row["digest"] for row in client.request("tags")["models"]}
    if model not in tags and pull:
        print("CPU 모형을 준비합니다:", model, flush=True)
        client.request("pull", {"name": model, "stream": False}, timeout=660)
        tags = {row["name"]: row["digest"] for row in client.request("tags")["models"]}
    if tags.get(model) != expected:
        raise RuntimeError(f"준비된 모형의 파일을 확인하세요: {model}")
    return expected


def prepare_festival(cache: Path) -> LocalOllama:
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    client = start_service(cache)
    ensure_model(client, CHAT_MODEL)
    return client
