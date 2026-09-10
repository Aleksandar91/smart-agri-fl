import os
import signal
import subprocess
import time
from typing import Optional

import requests


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


_STOP = False


def _handle_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _start_camera_subprocess() -> Optional[subprocess.Popen]:
    """
    Starts USB camera capture loop as a child process.

    Controlled by env:
      - ENABLE_USB_CAMERA=true/false (default: false)
    All CAM_* env vars are forwarded implicitly (same process env).
    """
    enabled = env_bool("ENABLE_USB_CAMERA", False)
    if not enabled:
        return None

    cmd = ["python", "-m", "app.capture_usb_cam"]
    try:
        proc = subprocess.Popen(cmd)
        print("[fl-client] usb camera enabled; started", {"cmd": cmd, "pid": proc.pid})
        return proc
    except Exception as e:
        print("[fl-client] usb camera failed to start", repr(e))
        return None


def main() -> None:
    base_url = os.getenv("FL_SERVER_BASE_URL", "http://fl-server:8080").rstrip("/")
    interval_s = env_int("INTERVAL_SECONDS", 5)
    timeout_s = env_int("HTTP_TIMEOUT_SECONDS", 3)

    print(f"[fl-client] starting. base_url={base_url} interval_s={interval_s}")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    cam_proc = _start_camera_subprocess()

    print(
        "[fl-client] health/ping polling disabled "
        f"(base_url={base_url}, timeout_s={timeout_s}); idle loop only"
    )
    while not _STOP:
        # Health/ping polling disabled — it flooded FL experiment logs.
        # Re-enable when testing connectivity against fl-server:8080:
        # try:
        #     health = requests.get(f"{base_url}/health", timeout=timeout_s)
        #     ping = requests.get(f"{base_url}/ping", timeout=timeout_s)
        #     print(
        #         "[fl-client] ok",
        #         {"health": health.status_code, "ping": ping.status_code},
        #         {"health_body": health.json(), "ping_body": ping.json()},
        #     )
        # except Exception as e:
        #     print("[fl-client] error", repr(e))
        time.sleep(interval_s)

    if cam_proc is not None:
        try:
            print("[fl-client] stopping usb camera process", {"pid": cam_proc.pid})
            cam_proc.terminate()
            cam_proc.wait(timeout=10)
        except Exception:
            try:
                cam_proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()

