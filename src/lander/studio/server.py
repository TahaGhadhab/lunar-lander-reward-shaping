"""Local HTTP server for the studio: the page, a JSON API, and a server-sent-event stream.

Standard library only. Binds to 127.0.0.1 by default: it can start training and run tests on this machine,
so it must not be reachable from the network.
"""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lander.studio.engine import Studio, list_models
from lander.studio.params import PARAMS, PRESETS, defaults

STATIC = Path(__file__).with_name("static")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the terminal quiet
            pass

        # ---------- helpers
        def send_json(self, data, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else {}

        def send_static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = studio.bus.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = ": ping\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
            except OSError:  # tab closed
                pass
            finally:
                studio.bus.unsubscribe(q)

        # ---------- routes
        def do_GET(self):  # noqa: N802
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                return self.send_static("index.html")
            if route in ("/app.js", "/style.css"):
                return self.send_static(route[1:])
            if route == "/api/events":
                return self.stream()
            try:
                if route == "/api/params":
                    return self.send_json(
                        {
                            "params": PARAMS,
                            "presets": PRESETS,
                            "defaults": defaults(),
                            "models": list_models(studio.root),
                        }
                    )
                if route == "/api/state":
                    return self.send_json(studio.state())
                if route == "/api/experiments":
                    from lander.dashboard import DEFAULT_REWARDS, collect_curves, collect_results

                    runs = studio.root / "runs"
                    return self.send_json(
                        {
                            "results": collect_results(runs, DEFAULT_REWARDS),
                            "curves": collect_curves(runs, DEFAULT_REWARDS),
                        }
                    )
                if route == "/api/experiments/progression":
                    return self.send_json(studio.progression or {"computing": True})

            except Exception as exc:
                return self.send_json({"error": str(exc)}, 500)
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            route = self.path.split("?", 1)[0]
            try:
                body = self.read_json()
                if route == "/api/train/start":
                    return self.send_json({"ok": True, "cfg": studio.start_training(body)})
                if route == "/api/train/stop":
                    studio.stop_training()
                    return self.send_json({"ok": True})
                if route == "/api/view":
                    if "enabled" in body:
                        studio.view["enabled"] = bool(body["enabled"])
                    if "speed" in body:
                        studio.view["speed"] = max(0.25, min(16.0, float(body["speed"])))
                    label = studio.load_pilot(body["model"]) if body.get("model") else None
                    return self.send_json({"ok": True, "view": studio.view, "label": label})
                if route == "/api/eval/start":
                    return self.send_json({"ok": True, **studio.start_eval(body)})
                if route == "/api/eval/stop":
                    studio.stop_eval.set()
                    return self.send_json({"ok": True})
                if route == "/api/tests/collect":
                    return self.send_json({"items": studio.collect_tests()})
                if route == "/api/tests/run":
                    return self.send_json({"ok": True, "ids": studio.run_tests(body.get("ids"))})
                if route == "/api/models":
                    return self.send_json({"models": list_models(studio.root)})
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            self.send_error(404)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    print("starting the studio (loading models)...")
    studio = Studio()
    threading.Thread(target=studio.collect_tests, daemon=True, name="collect").start()
    threading.Thread(target=studio.load_progression, daemon=True, name="progression").start()
    server = ThreadingHTTPServer((host, port), make_handler(studio))
    server.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"Lander Studio running at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
