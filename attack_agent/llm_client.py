"""Thin wrapper around litellm for the attack agent.

We keep one entry point so we can swap the backend later without touching
callsites. Failure modes are surfaced as exceptions rather than silent empty
strings.

Reliability:
- Transient network, rate-limit, and service errors are retried with backoff.
- Permanent request/auth/path errors fail immediately with structured
  diagnostics. Provider-side model routing failures are retried with backoff.
- Model requests have no client-side deadline and wait for the provider to
  finish or close the connection.
- Configuration follows OpenAI-compatible environment names:
  OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL. OPENAI_API_BASE is accepted
  as a legacy alias.
- Retry count can be tuned via ATTACK_AGENT_LLM_NUM_RETRIES. Set
  ATTACK_AGENT_FORCE_HTTP_CLIENT=1 to use the explicit status-code policy even
  when litellm is installed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit

_DEFAULT_NUM_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 2.0
_DEFAULT_RETRY_MAX_DELAY = 60.0
_DEFAULT_TEMPERATURE = 0.7
_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_HTTP_ERRORS = {(403, "model_not_available")}
_DEFAULT_THINKING_BODY: dict[str, Any] = {
    "enable_thinking": True,
    "chat_template_kwargs": {
        "enable_thinking": True,
    },
}
_DISABLE_THINKING_BODY: dict[str, Any] = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any]
    raw_model: str = ""


class LLMRequestError(RuntimeError):
    """Actionable provider error with retry metadata and no request secrets."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str = "",
        error_code: str = "",
        request_id: str = "",
        retryable: bool = False,
        attempts: int = 1,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.request_id = request_id
        self.retryable = retryable
        self.attempts = attempts
        details = []
        if status_code is not None:
            details.append(f"status={status_code}")
        if error_type:
            details.append(f"type={error_type}")
        if error_code:
            details.append(f"code={error_code}")
        if request_id:
            details.append(f"request_id={request_id}")
        details.append(f"attempts={attempts}")
        super().__init__(f"{message} ({', '.join(details)})")


def _ensure_litellm():
    try:
        from litellm import completion  # type: ignore
        return completion
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "litellm is required for the attack agent. "
            "Install with `uv pip install litellm`."
        ) from exc


def _try_litellm():
    try:
        from litellm import completion  # type: ignore
        return completion
    except ImportError:
        return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_retryable_http_error(status_code: int, error_code: str) -> bool:
    return (
        status_code in _RETRYABLE_HTTP_STATUS
        or (status_code, error_code) in _RETRYABLE_HTTP_ERRORS
    )


def _is_timeout_error(exc: BaseException) -> bool:
    """Recognize direct and urllib-wrapped socket timeouts."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, error.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(reason, TimeoutError) or "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def _auth_disabled_for_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    configured = {
        value.strip().lower()
        for value in os.environ.get("ATTACK_AGENT_NO_AUTH_HOSTS", "").split(",")
        if value.strip()
    }
    return host in configured


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _split_model_spec(model_id: str) -> tuple[str, str | None]:
    """Accept `model@base_url` specs for voters served by different endpoints."""
    model = (model_id or "").strip()
    base_url = None
    if "@" in model:
        model, base_url = model.rsplit("@", 1)
        model = model.strip()
        base_url = base_url.strip() or None
    if model.startswith("openai/"):
        model = model.removeprefix("openai/")
    return model, base_url


def chat(
    model_id: str,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    num_retries: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Run a single-turn chat completion and return the assistant content.

    Sets `num_retries` on the underlying litellm call so transient
    LLM-service errors (APIConnectionError, RateLimitError, ...) are retried
    with exponential backoff at the lowest layer. ``timeout_seconds`` is kept
    only for call-site compatibility and is intentionally ignored."""
    return chat_with_usage(
        model_id=model_id,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        num_retries=num_retries,
        timeout_seconds=timeout_seconds,
    ).content


def chat_with_usage(
    model_id: str,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    num_retries: int | None = None,
    timeout_seconds: float | None = None,
) -> ChatResult:
    """Run a single-turn chat completion and return content plus token usage.

    This mirrors `chat()` but preserves provider usage metadata for phase-2
    judge accounting. Requests have no client-side deadline."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    openai_model_id, base_url = _split_model_spec(model_id)
    completion = None if _env_bool("ATTACK_AGENT_FORCE_HTTP_CLIENT", False) else _try_litellm()
    if completion is None:
        return _chat_openai_compatible_http(
            model_id=openai_model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            base_url=base_url,
            num_retries=num_retries,
            timeout_seconds=timeout_seconds,
        )
    kwargs: dict[str, Any] = {
        "model": openai_model_id,
        "custom_llm_provider": "openai",
        "messages": messages,
        "temperature": temperature,
        "num_retries": (
            max(0, num_retries)
            if num_retries is not None
            else _env_int("ATTACK_AGENT_LLM_NUM_RETRIES", _DEFAULT_NUM_RETRIES)
        ),
    }
    _ = timeout_seconds  # Deprecated compatibility argument; deadlines are disabled.
    # LiteLLM applies its own 600-second fallback when timeout is omitted or
    # None. Infinity is accepted by its HTTP client and represents no deadline.
    kwargs["timeout"] = float("inf")
    thinking_mode = os.environ.get("ATTACK_AGENT_THINKING_MODE", "explicit").strip().lower()
    if thinking_mode in {"omit", "provider_default", "default"}:
        pass
    elif _env_bool("ATTACK_AGENT_ENABLE_THINKING", True):
        kwargs["extra_body"] = _DEFAULT_THINKING_BODY
    resolved_base_url = base_url or _env_first("OPENAI_BASE_URL", "OPENAI_API_BASE")
    if resolved_base_url is not None:
        kwargs["api_base"] = resolved_base_url
    api_key = _env_first("OPENAI_API_KEY")
    if api_key is not None and not _auth_disabled_for_url(resolved_base_url):
        kwargs["api_key"] = api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        sanitized_extra = dict(extra_body)
        sanitized_extra.pop("timeout", None)
        sanitized_extra.pop("request_timeout", None)
        kwargs.update(sanitized_extra)  # extra_body wins for non-timeout overrides

    resp = completion(**kwargs)
    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError(f"Unexpected litellm response shape: {resp!r}") from exc
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if hasattr(usage, "model_dump"):
        usage_dict = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage_dict = usage.dict()
    elif isinstance(usage, dict):
        usage_dict = dict(usage)
    else:
        usage_dict = {}
    raw_model = str(getattr(resp, "model", "") or (resp.get("model", "") if isinstance(resp, dict) else ""))
    return ChatResult(content=content, usage=usage_dict, raw_model=raw_model)


def _chat_openai_compatible_http(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
    base_url: str | None,
    num_retries: int | None = None,
    timeout_seconds: float | None = None,
) -> ChatResult:
    resolved_base_url = (base_url or _env_first("OPENAI_BASE_URL", "OPENAI_API_BASE") or "").rstrip("/")
    if not resolved_base_url:
        raise RuntimeError("OPENAI_BASE_URL or OPENAI_API_BASE is required when litellm is unavailable")
    if not model_id.strip():
        raise LLMRequestError(
            "LLM request is missing the required model ID",
            status_code=422,
            error_type="invalid_request_error",
            error_code="model_required",
        )
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "stream": _env_bool("ATTACK_AGENT_STREAM_RESPONSES", False),
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    thinking_mode = os.environ.get("ATTACK_AGENT_THINKING_MODE", "explicit").strip().lower()
    if thinking_mode in {"omit", "provider_default", "default"}:
        pass
    elif _env_bool("ATTACK_AGENT_ENABLE_THINKING", True):
        if "deepseek" in model_id.lower():
            body["reasoning_effort"] = os.environ.get("ATTACK_AGENT_REASONING_EFFORT", "max")
            body["thinking"] = {"type": "enabled"}
        else:
            body.update(_DEFAULT_THINKING_BODY)
    else:
        # These fields are accepted by the configured DeepSeek, GLM, Kimi, and
        # Qwen OpenAI-compatible endpoints and prevent reasoning-only replies.
        body.update(_DISABLE_THINKING_BODY)
    if extra_body:
        sanitized_extra = dict(extra_body)
        sanitized_extra.pop("timeout", None)
        sanitized_extra.pop("request_timeout", None)
        body.update(sanitized_extra)
    headers = {"Content-Type": "application/json"}
    api_key = _env_first("OPENAI_API_KEY")
    if api_key is not None and not _auth_disabled_for_url(resolved_base_url):
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LLMRequestError(
            f"LLM request body cannot be serialized as strict JSON: {exc}",
            status_code=400,
            error_type="invalid_request_error",
            error_code="invalid_json",
        ) from exc

    retries = max(
        0,
        num_retries
        if num_retries is not None
        else _env_int("ATTACK_AGENT_LLM_NUM_RETRIES", _DEFAULT_NUM_RETRIES),
    )
    max_attempts = retries + 1
    _ = timeout_seconds  # Deprecated compatibility argument; deadlines are disabled.
    request_timeout = None
    endpoint = f"{resolved_base_url}/chat/completions"
    for attempt in range(1, max_attempts + 1):
        response_headers = None
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        started_at = time.monotonic()
        request_finished = threading.Event()
        _log_request_start(model_id, endpoint=endpoint, attempt=attempt, max_attempts=max_attempts)
        _start_stall_warning(
            request_finished,
            model_id=model_id,
            endpoint=endpoint,
            attempt=attempt,
            started_at=started_at,
            request_timeout=request_timeout,
        )
        try:
            open_kwargs = {"timeout": request_timeout} if request_timeout is not None else {}
            with request.urlopen(req, **open_kwargs) as resp:
                response_headers = resp.headers
                if body.get("stream"):
                    data = _read_openai_sse_response(resp)
                    failure = None
                else:
                    raw = resp.read().decode("utf-8", errors="replace")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        failure = LLMRequestError(
                            f"Provider returned invalid JSON: {exc}",
                            status_code=200,
                            error_type="service_error",
                            error_code="invalid_response_json",
                            request_id=_request_id(response_headers),
                            retryable=True,
                            attempts=attempt,
                        )
                    else:
                        failure = None
            if failure is None:
                try:
                    content = data["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError, TypeError, AttributeError) as exc:
                    failure = LLMRequestError(
                        f"Provider returned an unexpected response shape: {exc}",
                        status_code=200,
                        error_type="service_error",
                        error_code="invalid_response_shape",
                        request_id=_request_id(response_headers),
                        retryable=True,
                        attempts=attempt,
                    )
                else:
                    if not isinstance(content, str) or not content.strip():
                        failure = LLMRequestError(
                            "Provider returned an empty message content",
                            status_code=200,
                            error_type="service_error",
                            error_code="empty_response_content",
                            request_id=_request_id(response_headers),
                            retryable=True,
                            attempts=attempt,
                        )
                        content = ""
                    else:
                        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                        _log_request_done(
                            model_id,
                            endpoint=endpoint,
                            attempt=attempt,
                            elapsed=time.monotonic() - started_at,
                        )
                        return ChatResult(
                            content=content,
                            usage=usage,
                            raw_model=str(data.get("model", model_id)),
                        )
        except error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            error_type, error_code, error_message = _parse_provider_error(raw_error)
            failure = LLMRequestError(
                _http_error_guidance(exc.code, error_code, error_message),
                status_code=exc.code,
                error_type=error_type,
                error_code=error_code,
                request_id=_request_id(exc.headers),
                retryable=_is_retryable_http_error(exc.code, error_code),
                attempts=attempt,
            )
            response_headers = exc.headers
        except LLMRequestError as exc:
            failure = exc
        except (error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            timed_out = _is_timeout_error(exc)
            failure = LLMRequestError(
                f"LLM connection failed: {exc}",
                error_type="connection_error",
                error_code="request_timeout" if timed_out else "connection_error",
                retryable=True,
                attempts=attempt,
            )
            response_headers = None
        finally:
            request_finished.set()

        _log_request_done(
            model_id,
            endpoint=endpoint,
            attempt=attempt,
            elapsed=time.monotonic() - started_at,
        )

        failure_max_attempts = max_attempts
        if not failure.retryable or attempt >= failure_max_attempts:
            raise failure
        delay = _retry_delay(
            attempt=attempt,
            headers=response_headers,
            error_code=failure.error_code,
        )
        _log_retry(
            failure,
            model_id=model_id,
            next_attempt=attempt + 1,
            max_attempts=failure_max_attempts,
            delay=delay,
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


def _read_openai_sse_response(resp: Any) -> dict[str, Any]:
    """Collect an OpenAI-compatible SSE stream into a normal response shape."""
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    raw_model = ""
    saw_event = False
    event_count = 0
    reasoning_chars = 0
    last_progress = time.monotonic()
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        saw_event = True
        event_count += 1
        if isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])
        if event.get("model"):
            raw_model = str(event["model"])
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        reasoning_piece = delta.get("reasoning_content")
        if isinstance(reasoning_piece, str):
            reasoning_chars += len(reasoning_piece)
        piece = delta.get("content")
        if piece is None:
            piece = message.get("content")
        if isinstance(piece, str):
            content_parts.append(piece)
        now = time.monotonic()
        if event_count == 1 or now - last_progress >= 15.0:
            print(
                f"[llm-stream] events={event_count} "
                f"content_chars={sum(len(item) for item in content_parts)} "
                f"reasoning_chars={reasoning_chars}",
                file=sys.stderr,
                flush=True,
            )
            last_progress = now
    if not saw_event:
        raise LLMRequestError(
            "Provider returned an empty SSE stream",
            status_code=200,
            error_type="service_error",
            error_code="empty_stream",
            retryable=True,
        )
    return {
        "choices": [{"message": {"content": "".join(content_parts)}}],
        "usage": usage,
        "model": raw_model,
    }


def _parse_provider_error(raw: str) -> tuple[str, str, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "service_error", "", raw.strip()[:500]
    if not isinstance(payload, dict):
        return "service_error", "", str(payload)[:500]
    details = payload.get("error", payload)
    if not isinstance(details, dict):
        return "service_error", "", str(details)[:500]
    return (
        str(details.get("type") or payload.get("type") or ""),
        str(details.get("code") or payload.get("code") or ""),
        str(details.get("message") or payload.get("message") or "")[:500],
    )


def _request_id(headers: Any) -> str:
    if headers is None:
        return ""
    for name in ("x-request-id", "request-id"):
        value = headers.get(name)
        if value:
            return str(value)
    return ""


def _http_error_guidance(status: int, code: str, provider_message: str) -> str:
    guidance = {
        400: "Request was rejected as invalid JSON or invalid parameters; inspect request fields and escaping",
        401: "API authentication failed; check OPENAI_API_KEY and whether the key is active",
        403: "Model is unavailable to this API key; check the model ID and project permission",
        404: "API path was not found; OPENAI_BASE_URL should normally end in /v1",
        422: "Request is missing a required field; check the model ID and request parameters",
        429: "API rate limit was reached; the request will be retried with backoff",
        500: "Provider failed while processing the request; the request will be retried",
        502: "Provider gateway or upstream failed; the request will be retried",
        503: "Provider is temporarily unavailable; the request will be retried",
        504: "Provider upstream wait expired; the request will be retried",
    }.get(status, "LLM request failed")
    if code:
        guidance = f"{guidance} [{code}]"
    if provider_message:
        guidance = f"{guidance}: {provider_message}"
    return guidance


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay(*, attempt: int, headers: Any, error_code: str) -> float:
    retry_after = _retry_after_seconds(headers)
    max_delay = max(0.0, _env_float("ATTACK_AGENT_LLM_RETRY_MAX_DELAY", _DEFAULT_RETRY_MAX_DELAY))
    if retry_after is not None:
        return min(max_delay, retry_after)
    base_delay = max(0.0, _env_float("ATTACK_AGENT_LLM_RETRY_BASE_DELAY", _DEFAULT_RETRY_BASE_DELAY))
    lowered_code = error_code.lower()
    if "tpm" in lowered_code:
        base_delay = max(base_delay, 30.0)
    elif "rpm" in lowered_code:
        base_delay = max(base_delay, 5.0)
    exponential = base_delay * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, min(1.0, base_delay * 0.25))
    return min(max_delay, exponential + jitter)


def _log_retry(
    failure: LLMRequestError,
    *,
    model_id: str,
    next_attempt: int,
    max_attempts: int,
    delay: float,
) -> None:
    fields = [
        f"model={model_id}",
        f"status={failure.status_code or 'network'}",
        f"code={failure.error_code or failure.error_type or 'unknown'}",
        f"attempt={next_attempt}/{max_attempts}",
        f"sleep={delay:.1f}s",
    ]
    if failure.request_id:
        fields.append(f"request_id={failure.request_id}")
    print(f"[llm-retry] {' '.join(fields)}", file=sys.stderr, flush=True)


def _endpoint_host(endpoint: str) -> str:
    return urlsplit(endpoint).netloc or endpoint


def _log_request_start(model_id: str, *, endpoint: str, attempt: int, max_attempts: int) -> None:
    print(
        f"[llm-request] START model={model_id} host={_endpoint_host(endpoint)} "
        f"attempt={attempt}/{max_attempts}",
        file=sys.stderr,
        flush=True,
    )


def _log_request_done(model_id: str, *, endpoint: str, attempt: int, elapsed: float) -> None:
    print(
        f"[llm-request] DONE model={model_id} host={_endpoint_host(endpoint)} "
        f"attempt={attempt} elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _start_stall_warning(
    finished: threading.Event,
    *,
    model_id: str,
    endpoint: str,
    attempt: int,
    started_at: float,
    request_timeout: float | None,
) -> None:
    interval = _env_float("ATTACK_AGENT_LLM_STALL_WARNING_SECONDS", 300.0)
    if interval <= 0:
        return

    def warn_until_finished() -> None:
        while not finished.wait(interval):
            elapsed = time.monotonic() - started_at
            timeout_status = (
                f"timeout={request_timeout:.1f}s"
                if request_timeout is not None
                else "no timeout configured"
            )
            print(
                f"[llm-request] WAITING model={model_id} host={_endpoint_host(endpoint)} "
                f"attempt={attempt} elapsed={elapsed:.1f}s; request is still open ({timeout_status})",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(target=warn_until_finished, daemon=True).start()


def default_model() -> str:
    model = _env_first("ATTACK_AGENT_MODEL", "OPENAI_MODEL", "MODEL") or "deepseek-v4-pro"
    model, _ = _split_model_spec(model)
    return model
