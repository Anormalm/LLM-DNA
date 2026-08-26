from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import RadarDB
from .reports import render_model_report


STYLE = """
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#0b0d10;color:#e8eaed}
main{max-width:1100px;margin:40px auto;padding:0 22px}a{color:#8ab4f8}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:10px;border-bottom:1px solid #2a2f36}.pill{padding:3px 8px;border-radius:99px;background:#252a31}
pre{white-space:pre-wrap;background:#15191e;padding:20px;border-radius:12px}.score{font-variant-numeric:tabular-nums}
"""


def serve(db: RadarDB, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/models":
                self._json(db.list_models(active_only=False))
                return
            if parsed.path == "/api/model":
                model_id = parse_qs(parsed.query).get("id", [""])[0]
                model = db.get_model(model_id)
                if not model:
                    self.send_error(404, "Unknown model")
                    return
                self._json(model)
                return
            if parsed.path == "/model":
                model_id = parse_qs(parsed.query).get("id", [""])[0]
                try:
                    report = render_model_report(db, model_id)
                except KeyError:
                    self.send_error(404, "Unknown model")
                    return
                body = f"<a href='/'>Back</a><pre>{html.escape(report)}</pre>"
                self._html(_page(model_id, body))
                return
            if parsed.path != "/":
                self.send_error(404)
                return
            rows = []
            for model in db.list_models(active_only=False):
                tags = " ".join(
                    f"<span class='pill'>{tag}</span>"
                    for tag, value in (
                        ("stealth", model.get("is_stealth")),
                        ("preview", model.get("is_preview")),
                        ("free", model.get("is_free")),
                    )
                    if value
                )
                model_id = html.escape(model["model_id"])
                rows.append(
                    f"<tr><td><a href='/model?id={model_id}'>{model_id}</a></td>"
                    f"<td>{'active' if model.get('active') else 'inactive'}</td>"
                    f"<td>{tags or '—'}</td><td class='score'>{float(model.get('priority_score') or 0):.2f}</td>"
                    f"<td>{html.escape(model.get('first_seen_at') or '')}</td></tr>"
                )
            body = (
                "<h1>Model Radar</h1><p>Emerging models and behavioral drift.</p>"
                "<table><thead><tr><th>Model</th><th>Status</th><th>Tags</th>"
                "<th>Score</th><th>First seen</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )
            self._html(_page("Model Radar", body))

        def _json(self, payload: Any) -> None:
            raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _html(self, payload: str) -> None:
            raw = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Model Radar dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width'><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )

