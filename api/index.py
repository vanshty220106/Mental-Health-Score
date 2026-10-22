import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from main import app as base_app


class VercelPathMiddleware:
    """
    Restores the original request path from Vercel headers if rewritten to /api/index.py
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            raw_matched = headers.get(b"x-matched-path") or headers.get(b"x-forwarded-uri")
            if raw_matched:
                matched_path = raw_matched.decode("latin1").split("?")[0]
                if matched_path and not matched_path.startswith("/api/index"):
                    scope["path"] = matched_path
                    scope["raw_path"] = matched_path.encode("latin1")
            elif scope.get("path", "").startswith("/api/index.py"):
                # Strip /api/index.py prefix if appended
                trimmed = scope["path"][len("/api/index.py"):]
                scope["path"] = trimmed if trimmed else "/"
                scope["raw_path"] = scope["path"].encode("latin1")
            elif scope.get("path", "").startswith("/api/index"):
                trimmed = scope["path"][len("/api/index"):]
                scope["path"] = trimmed if trimmed else "/"
                scope["raw_path"] = scope["path"].encode("latin1")

        await self.app(scope, receive, send)


app = VercelPathMiddleware(base_app)
