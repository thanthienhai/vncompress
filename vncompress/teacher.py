"""Teacher-LLM client for the dataset-distillation stage (docs/dataset_pipeline.md §4, §14).

This is the layer §4 always described but nothing implemented: a strong
instruction-following model that turns `(context, query, target_ratio)` into
supervision the student compressor can learn from.

Design constraints taken straight from §14:

- **Not locked to one model.** Any OpenAI-compatible `/chat/completions`
  endpoint works; model name, prompt version and generation parameters travel
  with every row, so an experiment stays reproducible after the teacher is
  swapped.
- **Cached.** Teacher calls cost money and are the slowest part of the
  pipeline. Every response is cached on disk keyed by
  (model, prompt version, messages, parameters), so a re-run after a crash or
  a prompt tweak to a *different* stage costs nothing.
- **Retried on a fixed delay.** A failed call waits `retry_delay` seconds (30
  by default) and is sent again, up to `max_attempts` (3) in total. A fixed
  wait rather than exponential backoff is deliberate: the dominant failure on a
  shared endpoint is a transient rate limit or restart, and a long, predictable
  pause is both gentler on the endpoint and easier to reason about when
  hundreds of workers are in flight.
- **Every exhausted call is recorded, never silently dropped.** After the last
  attempt the caller writes the failure to `data/teacher/failures_<stage>.jsonl`
  with the record id, the error and the attempt count, so a run can be traced
  and the failures replayed later instead of leaving a hole nobody notices.
- **Runnable offline.** `DryRunTeacherClient` walks the entire flow --
  prompts, parsing, verification, merge, split -- without spending a token, so
  the pipeline can be developed and tested before it is ever pointed at a
  billed endpoint.

Deliberately stdlib-only (`urllib`): one POST does not justify a new
dependency, and it keeps the flow testable in CI with nothing installed. To
switch to the `openai` SDK, reimplement `HTTPTeacherClient._post` -- nothing
else touches the transport.

Credentials come from `.env` (gitignored) or the real environment; see
`.env.example`. Nothing in this module ever logs the key.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_ENV_FILE = '.env'
DEFAULT_CACHE_DIR = os.path.join('data', 'teacher', '.cache')

#: Aliases accepted in .env so the credential blob a provider hands you can be
#: pasted in as-is (`baseURL` / `apiKey` / `model_name`).
_ENV_ALIASES = {
    'VNCOMPRESS_TEACHER_BASE_URL': ('TEACHER_BASE_URL', 'OPENAI_BASE_URL', 'baseURL', 'base_url'),
    'VNCOMPRESS_TEACHER_API_KEY': ('TEACHER_API_KEY', 'OPENAI_API_KEY', 'apiKey', 'api_key'),
    'VNCOMPRESS_TEACHER_MODEL': ('TEACHER_MODEL', 'model_name', 'model'),
}

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """Minimal `.env` reader (no python-dotenv dependency).

    Real environment variables win by default, so CI/secret managers override
    a developer's local file rather than the other way round.
    """
    path = path or os.path.join(_repo_root(), DEFAULT_ENV_FILE)
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    if os.environ.get(name):
        return os.environ[name]
    for alias in _ENV_ALIASES.get(name, ()):
        if os.environ.get(alias):
            return os.environ[alias]
    return default


class TeacherConfigError(RuntimeError):
    """Raised when the teacher endpoint is not configured. Carries the fix."""


@dataclass
class TeacherConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False, default='')
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    #: Total attempts per call, including the first. 3 = initial + 2 retries.
    max_attempts: int = 3
    #: Seconds to wait before resending a failed call. Fixed, not exponential.
    retry_delay: float = 30.0
    cache_dir: Optional[str] = None

    @classmethod
    def from_env(cls, env_file: Optional[str] = None, require_key: bool = True) -> 'TeacherConfig':
        load_dotenv(env_file)
        base_url = (_env('VNCOMPRESS_TEACHER_BASE_URL') or '').rstrip('/')
        model = _env('VNCOMPRESS_TEACHER_MODEL') or ''
        api_key = _env('VNCOMPRESS_TEACHER_API_KEY') or ''
        missing = [n for n, v in (('VNCOMPRESS_TEACHER_BASE_URL', base_url),
                                  ('VNCOMPRESS_TEACHER_MODEL', model),
                                  ('VNCOMPRESS_TEACHER_API_KEY', api_key if require_key else 'n/a'))
                   if not v]
        if missing:
            raise TeacherConfigError(
                'Teacher endpoint is not configured; missing: ' + ', '.join(missing) + '.\n'
                'Fix: cp .env.example .env and fill it in (.env is gitignored), '
                'or export the variables. Use --dry-run to exercise the flow without an endpoint.')
        return cls(
            base_url=base_url, model=model, api_key=api_key,
            temperature=float(_env('VNCOMPRESS_TEACHER_TEMPERATURE', '0.1')),
            max_tokens=int(_env('VNCOMPRESS_TEACHER_MAX_TOKENS', '4096')),
            timeout=int(_env('VNCOMPRESS_TEACHER_TIMEOUT', '120')),
            max_attempts=int(_env('VNCOMPRESS_TEACHER_MAX_ATTEMPTS', '3')),
            retry_delay=float(_env('VNCOMPRESS_TEACHER_RETRY_DELAY', '30')),
            cache_dir=_env('VNCOMPRESS_TEACHER_CACHE_DIR', os.path.join(_repo_root(), DEFAULT_CACHE_DIR)),
        )

    def provenance(self, prompt_version: str) -> Dict[str, Any]:
        """The §14 metadata every generated row must carry. Never includes the key."""
        return {
            'model': self.model,
            'base_url': self.base_url,
            'prompt_version': prompt_version,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


# ============================================================================
# Clients
# ============================================================================


class TeacherCallError(RuntimeError):
    """A call that exhausted every attempt. Carries what a failure log needs."""

    def __init__(self, message: str, attempts: int, last_error: str, status: Optional[int] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {'error': str(self), 'attempts': self.attempts,
                'last_error': self.last_error, 'status': self.status}


class TeacherClient:
    """Interface: turn a chat message list into raw assistant text."""

    name = 'abstract'

    def complete(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        raise NotImplementedError


class HTTPTeacherClient(TeacherClient):
    """OpenAI-compatible `/chat/completions` over stdlib urllib, with retries."""

    name = 'http'

    def __init__(self, config: TeacherConfig, sleep=time.sleep):
        self.config = config
        self._sleep = sleep
        self.n_calls = 0
        self.n_retries = 0

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            f'{self.config.base_url}/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.config.api_key}'},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def complete(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature if temperature is None else temperature,
            'max_tokens': self.config.max_tokens,
        }
        last_error: Optional[Exception] = None
        status: Optional[int] = None
        attempts = max(1, self.config.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                self.n_calls += 1
                body = self._post(payload)
                return (body['choices'][0]['message'].get('content') or '').strip()
            except urllib.error.HTTPError as exc:
                # A 4xx that is not rate limiting is a bug in the request:
                # retrying burns quota and hides the real error.
                if exc.code not in _RETRYABLE_STATUS:
                    detail = exc.read().decode('utf-8', 'replace')[:400] if exc.fp else ''
                    raise TeacherCallError(f'Teacher endpoint returned HTTP {exc.code}: {detail}',
                                           attempts=attempt, last_error=repr(exc), status=exc.code) from exc
                last_error, status = exc, exc.code
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = exc
            if attempt < attempts:
                self.n_retries += 1
                self._sleep(self.config.retry_delay)
        raise TeacherCallError(
            f'Teacher call failed after {attempts} attempt(s)',
            attempts=attempts, last_error=repr(last_error), status=status)


class DryRunTeacherClient(TeacherClient):
    """Deterministic offline stand-in.

    Returns a structurally valid response derived from the prompt itself, so
    every downstream stage -- JSON parsing, §6 verification, the merge, the
    document-level split -- is exercised end to end without an endpoint or a
    single billed token. It is a plumbing check, not a quality check: the
    "compression" is a naive lead-sentence cut.
    """

    name = 'dry-run'

    def __init__(self, responder=None):
        self.responder = responder
        self.n_calls = 0
        self.seen: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        self.n_calls += 1
        self.seen.append(messages)
        if self.responder is not None:
            return self.responder(messages)
        from .teacher_prompts import dry_run_response

        return dry_run_response(messages)


class CachedTeacherClient(TeacherClient):
    """Disk cache in front of any client (§14 'Cache teacher output').

    Keyed by (model, prompt version, messages, temperature, max_tokens), so
    editing one stage's prompt does not invalidate another's, and a re-run
    after a crash resumes for free.
    """

    name = 'cached'

    def __init__(self, inner: TeacherClient, cache_dir: str, model: str, prompt_version: str,
                 max_tokens: int, enabled: bool = True):
        self.inner = inner
        self.cache_dir = cache_dir
        self.model = model
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.enabled = enabled
        self.n_hits = 0
        self.n_misses = 0
        # Counters are read from the main thread while workers update them.
        self._lock = threading.Lock()

    def _key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        blob = json.dumps({
            'model': self.model, 'prompt_version': self.prompt_version,
            'messages': messages, 'temperature': temperature, 'max_tokens': self.max_tokens,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], f'{key}.json')

    def complete(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        temp = 0.0 if temperature is None else temperature
        if not self.enabled:
            return self.inner.complete(messages, temperature)
        path = self._path(self._key(messages, temp))
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = json.load(f)['content']
                with self._lock:
                    self.n_hits += 1
                return content
            except (OSError, ValueError, KeyError):
                pass  # corrupt cache entry: fall through and regenerate
        content = self.inner.complete(messages, temperature)
        with self._lock:
            self.n_misses += 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write via a per-thread temp file then rename: two workers racing on
        # the same prompt must never leave a half-written entry behind.
        tmp = f'{path}.{threading.get_ident()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'content': content, 'model': self.model,
                       'prompt_version': self.prompt_version}, f, ensure_ascii=False)
        os.replace(tmp, path)
        return content


def build_client(config: Optional[TeacherConfig], prompt_version: str, dry_run: bool = False,
                 use_cache: bool = True) -> TeacherClient:
    """The one place that decides which transport a stage talks to."""
    inner: TeacherClient = DryRunTeacherClient() if dry_run else HTTPTeacherClient(config)
    if not use_cache or config is None:
        return inner
    return CachedTeacherClient(
        inner, cache_dir=config.cache_dir or os.path.join(_repo_root(), DEFAULT_CACHE_DIR),
        model=config.model if not dry_run else f'dry-run:{config.model}',
        prompt_version=prompt_version, max_tokens=config.max_tokens)


# ============================================================================
# Response parsing
# ============================================================================


class TeacherOutputError(ValueError):
    """The teacher returned something that is not usable JSON."""


def extract_json(text: str) -> Any:
    """Parse a JSON object out of a chat response.

    Instruction-tuned models wrap JSON in ```json fences or prepend a sentence
    even when told not to, and §14 says to retry on invalid output rather than
    silently dropping the sample -- so callers need to tell "model was chatty"
    (recoverable) apart from "model produced nothing parseable" (retry).
    """
    if not text or not text.strip():
        raise TeacherOutputError('empty response')
    candidate = text.strip()

    if candidate.startswith('```'):
        lines = candidate.splitlines()
        lines = lines[1:]                      # drop ```json
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        candidate = '\n'.join(lines).strip()

    try:
        return json.loads(candidate)
    except ValueError:
        pass

    # Fall back to the outermost {...} / [...] span.
    for opener, closer in (('{', '}'), ('[', ']')):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except ValueError:
                continue
    raise TeacherOutputError(f'no parseable JSON in response (first 200 chars: {text[:200]!r})')
