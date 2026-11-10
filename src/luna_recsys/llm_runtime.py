"""Verified, local-only Ollama preparation for the chapter-9 CPU exercises.

Downloads only official artifacts into the caller's cache. No service installation,
administrator changes, credentials, hosted API, or simulated model responses.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import socket
import subprocess
import tarfile
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

OLLAMA_VERSION = '0.40.15'
CHAT_MODEL = 'qwen2.5:0.5b'
EMBED_MODEL = 'nomic-embed-text:v1.5'
MODEL_DIGESTS = {
    CHAT_MODEL: 'a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67',
    EMBED_MODEL: '0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f',
}
ASSETS = {
    'Windows': ('ollama-windows-amd64.zip', '9ca74a7ab66ec86870b1eecac7df160df08dc4e3a291ab5d2d59bcf9310221c6'),
    'Linux': ('ollama-linux-amd64.tar.zst', '628c018404a4206493efd38e46251c154ede5141b8b8dc85c659b3b027c47379'),
}


def file_sha256(path: str | Path) -> str:
    """Stream a local file's SHA-256 without reading the whole archive into RAM."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_checked(url: str, path: Path, expected: str) -> Path:
    """HTTPS download/cache; reject a corrupt file, never fabricate replacement data."""
    if urlparse(url).scheme != 'https':
        raise ValueError('A public HTTPS source is required.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != expected:
            raise ValueError('Cached file checksum differs; preserve and inspect that file.')
        return path
    partial = path.with_name(path.name + '.part')
    with urlopen(url, timeout=120) as source, partial.open('wb') as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if file_sha256(partial) != expected:
        raise ValueError('Downloaded bytes do not match the verified source checksum.')
    partial.replace(path)
    return path


def _cpu_asset(name: str) -> bool:
    return not any(s in name.lower() for s in ('cuda', 'rocm', 'vulkan', 'cublas', 'cufft'))


def _safe_target(folder: Path, name: str) -> Path:
    parts = PurePosixPath(name.replace('\\', '/')).parts
    if not parts or '..' in parts or ':' in parts[0] or name.startswith(('/', '\\')):
        raise ValueError('Unsafe archive entry.')
    target = (folder / Path(*parts)).resolve()
    target.relative_to(folder.resolve())
    return target


def _extract_cpu(archive: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir() or not _cpu_asset(item.filename):
                    continue
                target = _safe_target(folder, item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
    else:
        import zstandard
        with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                for item in tar:
                    if not _cpu_asset(item.name) or item.isdir():
                        continue
                    target = _safe_target(folder, item.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if item.isfile():
                        with tar.extractfile(item) as contents, target.open('wb') as out:
                            shutil.copyfileobj(contents, out)
                        target.chmod(item.mode & 0o777)
                    elif item.issym():
                        (target.parent / item.linkname).resolve().relative_to(folder.resolve())
                        if target.is_symlink() and os.readlink(target) == item.linkname:
                            continue
                        if target.exists() or target.is_symlink():
                            raise ValueError('Existing archive link target differs; preserve and inspect it.')
                        target.symlink_to(item.linkname)
                    else:
                        raise ValueError('Unexpected archive entry type.')


def prepare_binary(cache: str | Path) -> Path:
    """Prepare versioned CPU files in a user cache; retain licenses and verify reuse."""
    system = platform.system()
    if system not in ASSETS or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('This lesson supports x86-64 Colab Linux and Windows.')
    cache = Path(cache).resolve()
    folder = cache / ('ollama-' + OLLAMA_VERSION + '-' + system.lower())
    stamp = folder / 'verified-files.json'
    if stamp.exists():
        entries = json.loads(stamp.read_text(encoding='utf-8'))
        if not entries or any(file_sha256(folder / n) != h for n, h in entries.items()):
            raise ValueError('Prepared runtime files changed; inspect the cache before reuse.')
    else:
        name, checksum = ASSETS[system]
        archive = cache / name
        url = f'https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/{name}'
        print('최초 실행기 준비: 공식 파일 다운로드와 무결성 검사 중입니다.', flush=True)
        download_checked(url, archive, checksum)
        _extract_cpu(archive, folder)
        entries = {p.relative_to(folder).as_posix(): file_sha256(p)
                   for p in folder.rglob('*') if p.is_file()}
        stamp.write_text(json.dumps(entries, sort_keys=True), encoding='utf-8')
        archive.unlink()  # Only this newly verified, reproducible temporary archive.
    binary = folder / ('ollama.exe' if system == 'Windows' else 'bin/ollama')
    if not binary.is_file():
        raise FileNotFoundError('Official runtime executable was not found after extraction.')
    return binary


class LocalOllama:
    """Actual HTTP chat/embed calls to a loopback CPU server, with observable metadata."""

    def __init__(self, host: str):
        parsed = urlparse(host)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1') or parsed.username or parsed.password:
            raise ValueError('Only a local, credential-free Ollama endpoint is allowed.')
        self.host = host.rstrip('/')
        self.calls: list[dict] = []

    def request(self, endpoint: str, payload: dict | None = None, *, timeout: float = 300) -> dict:
        started = time.monotonic()
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = Request(self.host + '/api/' + endpoint, data=data,
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except (OSError, ValueError, HTTPError) as exc:
            raise RuntimeError(f'실제 Ollama {endpoint} 요청 실패: {exc}') from exc
        if 'error' in result:
            raise RuntimeError('Ollama returned an error: ' + str(result['error']))
        metadata = {k: result[k] for k in ('model', 'prompt_eval_count', 'eval_count', 'total_duration') if k in result}
        metadata.update(endpoint=endpoint, wall_seconds=time.monotonic() - started)
        if endpoint in ('chat', 'embed'):
            self.calls.append(metadata)
        return result

    def chat(self, messages: list[dict], schema: dict) -> tuple[str, dict]:
        result = self.request('chat', {'model': CHAT_MODEL, 'messages': messages,
            'format': schema, 'stream': False, 'keep_alive': '5m',
            'options': {'temperature': 0, 'seed': 20261111, 'num_ctx': 2048,
                        'num_predict': 128, 'num_thread': 2, 'num_gpu': 0}})
        content = result.get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('Actual model response was empty.')
        return content, self.calls[-1].copy()

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Embedding inputs must contain nonempty text.')
        result = self.request('embed', {'model': EMBED_MODEL, 'input': texts,
            'truncate': False, 'keep_alive': '5m', 'options': {'num_thread': 2, 'num_gpu': 0}})
        vectors = result.get('embeddings', [])
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise RuntimeError('Actual embedding output has an invalid shape.')
        return vectors, self.calls[-1].copy()


def prepare_runtime(cache: str | Path | None = None) -> LocalOllama:
    """Prepare/reuse this lesson's local service and exact official model digests.

    LUNA_OLLAMA_HOST explicitly reuses a controlled loopback server for local QA.
    Colab needs no key. Failures propagate; there is no cached-answer fallback.
    """
    cache = Path(cache or Path.home() / '.cache/luna-recsys/w11b-runtime').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    host = os.environ.get('LUNA_OLLAMA_HOST')
    service = cache / 'service.json'
    if not host and service.exists():
        candidate = json.loads(service.read_text(encoding='utf-8'))['host']
        try:
            LocalOllama(candidate).request('version', timeout=2)
            host = candidate
        except RuntimeError:
            pass
    if not host:
        binary = prepare_binary(cache)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        host = f'http://127.0.0.1:{port}'
        env = os.environ.copy()
        env.update(OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=str(cache / 'models'),
                   OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_NUM_PARALLEL='1',
                   CUDA_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')
        with (cache / 'server.log').open('ab') as log:
            proc = subprocess.Popen([str(binary), 'serve'], env=env, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        service.write_text(json.dumps({'host': host, 'pid': proc.pid}), encoding='utf-8')
    client = LocalOllama(host)
    for attempt in range(60):
        try:
            version = client.request('version', timeout=2)['version']
            break
        except RuntimeError:
            if attempt == 59:
                raise
            time.sleep(1)
    if version.lstrip('v') != OLLAMA_VERSION:
        raise RuntimeError('The local runtime version differs from this verified lesson.')
    tags = {m['name']: m['digest'] for m in client.request('tags')['models']}
    for model, digest in MODEL_DIGESTS.items():
        if model not in tags:
            print('무료 로컬 모형 준비:', model, flush=True)
            client.request('pull', {'name': model, 'stream': False}, timeout=600)
            tags = {m['name']: m['digest'] for m in client.request('tags')['models']}
        if tags.get(model) != digest:
            raise RuntimeError('Model bytes differ from the verified lesson: ' + model)
    print('실제 무료 CPU 추론 준비 완료 · Ollama', version, flush=True)
    return client
