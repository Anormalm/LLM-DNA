from __future__ import annotations

import json
from typing import Any
from urllib import request


def send_webhook(url: str, message: str, *, format_name: str = "discord", timeout: float = 20) -> None:
    if not url:
        return
    if format_name == "discord":
        payload: dict[str, Any] = {"content": message}
    elif format_name == "slack":
        payload = {"text": message}
    else:
        payload = {"message": message}
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "model-radar/0.1.0"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Webhook returned HTTP {response.status}")


def drift_message(event: dict[str, Any]) -> str:
    return (
        f"Model Radar drift alert: {event['model_id']}\n"
        f"distance={event['distance']:.4f} threshold={event['threshold_used']:.4f}\n"
        f"previous={event['previous_run_id']} current={event['current_run_id']}"
    )

