"""Simple static file server with API proxy for tunnel use."""
import http.server
import urllib.request
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5174
DIST = os.path.join(os.path.dirname(__file__), "dist")
API_TARGET = "http://localhost:8003"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIST, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            # SPA fallback: serve index.html for non-file paths
            full = os.path.join(DIST, self.path.lstrip("/"))
            if not os.path.exists(full) and not "." in self.path.split("/")[-1]:
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def _proxy(self):
        url = API_TARGET + self.path
        body = None
        if self.headers.get("Content-Length"):
            body = self.rfile.read(int(self.headers["Content-Length"]))

        req = urllib.request.Request(url, data=body, method=self.command)
        for key, val in self.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding",):
                        self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())


if __name__ == "__main__":
    os.chdir(DIST)
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving {DIST} on http://0.0.0.0:{PORT}")
    print(f"Proxying /api/* -> {API_TARGET}")
    server.serve_forever()
