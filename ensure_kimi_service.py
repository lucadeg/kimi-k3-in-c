#!/usr/bin/env python3
"""
Ensures the Kimi K3 local inference service is running on port 8095.
If not, launches it in the background and waits for the /health endpoint.
"""

import sys
import time
import subprocess
from pathlib import Path
import urllib.request

PORT = 8095
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"
CURRENT_DIR = Path(__file__).resolve().parent
SERVICE_SCRIPT = CURRENT_DIR / "kimi_k3_service.py"

def is_service_ready() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "Kimi-HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def ensure_kimi_running():
    if is_service_ready():
        print(f"[KIMI K3] Local Engine is active on http://127.0.0.1:{PORT}")
        return True

    print(f"[KIMI K3] Starting Local Inference Engine on http://127.0.0.1:{PORT}...")
    if sys.platform == "win32":
        # Launch in background via pythonw or subprocess detached
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [sys.executable, str(SERVICE_SCRIPT)],
            cwd=str(CURRENT_DIR),
            creationflags=CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True
        )
    else:
        subprocess.Popen(
            [sys.executable, str(SERVICE_SCRIPT)],
            cwd=str(CURRENT_DIR),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )

    # Wait up to 10 seconds for service to come online
    for _ in range(20):
        time.sleep(0.5)
        if is_service_ready():
            print(f"[KIMI K3] Service successfully started and verified on http://127.0.0.1:{PORT}")
            return True

    print("[KIMI K3] Warning: Service start timeout. Port 8095 not yet responding.")
    return False

if __name__ == "__main__":
    ok = ensure_kimi_running()
    sys.exit(0 if ok else 1)
