from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Sequence
from urllib import error, parse, request

from .utils import utc_now


class OpenRouterError(RuntimeError):
    pass


@dataclass(slots=True)
class HTTPResult:
    status: int
    headers: dict[str, str]
    payload: Any


RequestFunction = Callable[[str, str, Mapping[str, str], bytes | None, float], HTTPResult]


class OpenRouterClient:
    """Small dependency-free client for the OpenRouter endpoints used by Model Radar."""

    def __init__(
        self,
        api_key: str = "",
        *,
        timeout_seconds: float = 90,
        retries: int = 3,
        http_referer: str = "",
        app_title: str = "Model Radar",
        request_fn: RequestFunction | None = None,
    ):
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(0, int(retries))
        self.http_referer = http_referer
        self.app_title = app_title
        self.request_fn = request_fn or _urllib_request

    def _headers(self, *, require_auth: bool = False, metadata: bool = False) -> dict[str, str]:
        if require_auth and not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is required for this operation")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "model-radar/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        if metadata:
            headers["X-OpenRouter-Metadata"] = "true"
        return headers

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Any = None,
        require_auth: bool = False,
        metadata: bool = False,
    ) -> HTTPResult:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self.request_fn(
                    method,
                    url,
                    self._headers(require_auth=require_auth, metadata=metadata),
                    body,
                    self.timeout_seconds,
                )
                if 200 <= result.status < 300:
                    return result
                message = _error_message(result.payload)
                if result.status not in {429, 500, 502, 503, 504, 529} or attempt >= self.retries:
                    raise OpenRouterError(f"OpenRouter HTTP {result.status}: {message}")
                retry_after = _retry_after(result.headers)
                time.sleep(retry_after if retry_after is not None else min(2**attempt, 30))
            except (error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2**attempt, 30))
        raise OpenRouterError(f"OpenRouter request failed: {last_error}")

    def list_models(self) -> list[dict[str, Any]]:
        url = "https://openrouter.ai/api/v1/models"
        models: list[dict[str, Any]] = []
        visited: set[str] = set()
        while url and url not in visited:
            visited.add(url)
            result = self._request("GET", url)
            payload = result.payload
            if isinstance(payload, list):
                page = payload
                next_url = None
            elif isinstance(payload, dict):
                page = payload.get("data", [])
                links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
                next_url = links.get("next")
            else:
                raise OpenRouterError("Unexpected Models API response")
            models.extend(item for item in page if isinstance(item, dict))
            url = parse.urljoin(url, str(next_url)) if next_url else ""
        return models

    def get_model_endpoints(self, model_id: str) -> dict[str, Any]:
        encoded = "/".join(parse.quote(part, safe="") for part in model_id.split("/"))
        url = f"https://openrouter.ai/api/v1/models/{encoded}/endpoints"
        result = self._request("GET", url)
        return result.payload if isinstance(result.payload, dict) else {"data": result.payload}

    def rankings_daily(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "day",
        category: str = "",
        language_type: str = "",
        modality: str = "",
        context_bucket: str = "",
    ) -> dict[str, Any]:
        if end_date is None:
            end_date = (date.today() - timedelta(days=1)).isoformat()
        if start_date is None:
            start_date = (date.fromisoformat(end_date) - timedelta(days=29)).isoformat()
        params: dict[str, str] = {
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
        }
        for key, value in {
            "category": category,
            "language_type": language_type,
            "modality": modality,
            "context_bucket": context_bucket,
        }.items():
            if value:
                params[key] = value
        url = "https://openrouter.ai/api/v1/datasets/rankings-daily?" + parse.urlencode(params)
        result = self._request("GET", url, require_auth=True)
        if not isinstance(result.payload, dict):
            raise OpenRouterError("Unexpected rankings response")
        return result.payload

    def chat_completion(self, model_id: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if options.get("system_prompt"):
            messages.append({"role": "system", "content": str(options["system_prompt"])})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": float(options.get("temperature", 0.0)),
            "top_p": float(options.get("top_p", 1.0)),
            "max_tokens": int(options.get("max_tokens", 256)),
        }
        if options.get("seed") is not None:
            body["seed"] = int(options["seed"])
        provider = _provider_options(options)
        if provider:
            body["provider"] = provider
        result = self._request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            payload=body,
            require_auth=True,
            metadata=True,
        )
        parsed = parse_completion(result.payload)
        operational_headers = {
            key: value
            for key, value in result.headers.items()
            if key.lower().startswith("x-openrouter") or key.lower() == "x-generation-id"
        }
        parsed["http_headers"] = operational_headers
        if isinstance(parsed.get("raw"), dict) and operational_headers:
            parsed["raw"] = {**parsed["raw"], "_http_headers": operational_headers}
        return parsed

    def submit_batch(
        self,
        model_id: str,
        requests_to_submit: Sequence[tuple[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        requests_payload = []
        for custom_id, prompt in requests_to_submit:
            messages: list[dict[str, str]] = []
            if options.get("system_prompt"):
                messages.append({"role": "system", "content": str(options["system_prompt"])})
            messages.append({"role": "user", "content": prompt})
            body: dict[str, Any] = {
                "messages": messages,
                "temperature": float(options.get("temperature", 0.0)),
                "top_p": float(options.get("top_p", 1.0)),
                "max_tokens": int(options.get("max_tokens", 256)),
            }
            if options.get("seed") is not None:
                body["seed"] = int(options["seed"])
            provider = _provider_options(options)
            if provider:
                body["provider"] = provider
            requests_payload.append({"custom_id": custom_id, "body": body})
        # Key order is deliberate: OpenRouter stream-parses these top-level fields.
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": model_id,
            "requests": requests_payload,
        }
        result = self._request(
            "POST",
            "https://openrouter.ai/api/beta/batches",
            payload=payload,
            require_auth=True,
        )
        if not isinstance(result.payload, dict) or not result.payload.get("id"):
            raise OpenRouterError("Unexpected batch submission response")
        return result.payload

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        encoded = parse.quote(batch_id, safe="")
        result = self._request(
            "GET",
            f"https://openrouter.ai/api/beta/batches/{encoded}",
            require_auth=True,
        )
        if not isinstance(result.payload, dict):
            raise OpenRouterError("Unexpected batch response")
        return result.payload

    def run_batch(
        self,
        model_id: str,
        requests_to_submit: Sequence[tuple[str, str]],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        batch = self.submit_batch(model_id, requests_to_submit, options)
        batch_id = str(batch["id"])
        terminal = {"completed", "failed", "expired", "cancelled"}
        timeout_seconds = float(options.get("batch_timeout_seconds", 86400))
        poll_interval = max(1.0, float(options.get("poll_interval_seconds", 30)))
        started = time.monotonic()
        while str(batch.get("status")) not in terminal:
            if time.monotonic() - started > timeout_seconds:
                raise OpenRouterError(f"Batch {batch_id} timed out after {timeout_seconds:g}s")
            time.sleep(poll_interval)
            batch = self.get_batch(batch_id)
        if batch.get("status") != "completed":
            raise OpenRouterError(f"Batch {batch_id} ended with status={batch.get('status')}")
        output: list[dict[str, Any]] = []
        for item in batch.get("results") or []:
            custom_id = item.get("custom_id")
            if item.get("error"):
                output.append(
                    {
                        "custom_id": custom_id,
                        "response": "",
                        "error": _error_message(item["error"]),
                        "raw": item,
                        "created_at": utc_now(),
                    }
                )
                continue
            response = item.get("response") if isinstance(item.get("response"), dict) else {}
            body = response.get("body") if isinstance(response.get("body"), dict) else {}
            parsed = parse_completion(body)
            parsed.update({"custom_id": custom_id, "raw": item})
            output.append(parsed)
        return output


def parse_completion(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"response": "", "error": "Non-object completion response", "raw": payload}
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")
        )
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning")
        content = reasoning if isinstance(reasoning, str) else ""

    metadata = payload.get("openrouter_metadata")
    provider = payload.get("provider")
    if not provider and isinstance(metadata, dict):
        provider = metadata.get("provider") or metadata.get("provider_name")

    return {
        "response": content.strip(),
        "generation_id": payload.get("id"),
        "returned_model": payload.get("model"),
        "provider": provider,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "raw": payload,
        "error": None,
        "created_at": utc_now(),
    }


def _provider_options(options: dict[str, Any]) -> dict[str, Any]:
    provider: dict[str, Any] = {}
    only = options.get("provider_only") or []
    if isinstance(only, str):
        only = [item.strip() for item in only.split(",") if item.strip()]
    if only:
        provider["only"] = list(only)
    if "allow_fallbacks" in options:
        provider["allow_fallbacks"] = bool(options["allow_fallbacks"])
    if options.get("require_parameters"):
        provider["require_parameters"] = True
    return provider


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return min(float(value), 60.0)
            except (TypeError, ValueError):
                return None
    return None


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        nested = payload.get("error")
        if isinstance(nested, dict):
            return str(nested.get("message") or nested)
        return str(payload.get("message") or payload)
    return str(payload)


def _urllib_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HTTPResult:
    req = request.Request(url, data=body, headers=dict(headers), method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw.strip() else {}
            return HTTPResult(response.status, dict(response.headers.items()), payload)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {"message": raw[:1000]}
        return HTTPResult(exc.code, dict(exc.headers.items()), payload)
