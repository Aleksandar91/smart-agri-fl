import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    interval_s: float
    output_dir: Path
    device: str
    resolution: Tuple[int, int]
    jpeg_quality: int
    keep_last: int
    prefix: str
    no_banner: bool
    warmup_s: float
    rotate: int


_STOP = False


def _handle_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _timestamp_name(prefix: str) -> str:
    # Example: capture_20260216_235959_123.jpg
    now = datetime.now()
    base = now.strftime("%Y%m%d_%H%M%S")
    ms = int(now.microsecond / 1000)
    return f"{prefix}_{base}_{ms:03d}.jpg"


def _prune_old_images(output_dir: Path, keep_last: int, prefix: str) -> None:
    if keep_last <= 0:
        return
    images = sorted(
        output_dir.glob(f"{prefix}_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in images[keep_last:]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            # best-effort cleanup
            pass


def _parse_device(dev: str) -> Union[int, str]:
    # Accept "0" as index, or "/dev/video0" as path
    s = dev.strip()
    if s.isdigit():
        return int(s)
    return s


def _capture_with_fswebcam(cfg: Config, out_path: Path) -> None:
    exe = shutil.which("fswebcam")
    if not exe:
        raise RuntimeError("fswebcam not found in PATH")

    w, h = cfg.resolution
    cmd = [
        exe,
        "-d",
        cfg.device,
        "-r",
        f"{w}x{h}",
        "--jpeg",
        str(cfg.jpeg_quality),
    ]
    if cfg.no_banner:
        cmd.append("--no-banner")
    if cfg.rotate in {90, 180, 270}:
        cmd += ["--rotate", str(cfg.rotate)]
    # Give camera a moment to settle (some USB cams need it)
    if cfg.warmup_s > 0:
        cmd += ["-S", str(max(1, int(cfg.warmup_s)))]
    cmd.append(str(out_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "fswebcam failed "
            + repr(
                {
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                }
            )
        )


def _capture_with_opencv(cfg: Config, out_path: Path) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenCV (cv2) not available: {e!r}") from e

    dev = _parse_device(cfg.device)
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open device={cfg.device!r}")

    w, h = cfg.resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))

    # Warmup: read frames for a short duration
    warmup_until = time.monotonic() + max(0.0, cfg.warmup_s)
    ok = False
    frame = None
    while time.monotonic() < warmup_until:
        ok, frame = cap.read()
        if ok:
            break

    if not ok:
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("cv2 failed to read a frame")

    if cfg.rotate in {90, 180, 270}:
        if cfg.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif cfg.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
    ok2 = cv2.imwrite(str(out_path), frame, params)
    if not ok2:
        raise RuntimeError("cv2.imwrite failed")


def capture_once(cfg: Config) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    out_path = cfg.output_dir / _timestamp_name(cfg.prefix)
    tmp_path = out_path.with_suffix(".jpg.tmp")

    # Write to temp then rename to avoid partially-written files
    if shutil.which("fswebcam"):
        _capture_with_fswebcam(cfg, tmp_path)
    else:
        _capture_with_opencv(cfg, tmp_path)

    tmp_path.replace(out_path)
    _prune_old_images(cfg.output_dir, cfg.keep_last, cfg.prefix)
    return out_path


def build_config_from_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Periodically capture images from a USB camera and save locally."
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=env_float("CAM_INTERVAL_SECONDS", 5.0),
        help="Capture interval in seconds (env: CAM_INTERVAL_SECONDS).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=env_str("CAM_OUTPUT_DIR", "./captures"),
        help="Directory where images are saved (env: CAM_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=env_str("CAM_DEVICE", "/dev/video0"),
        help='Camera device ("/dev/video0" or "0") (env: CAM_DEVICE).',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=env_int("CAM_WIDTH", 1280),
        help="Capture width (env: CAM_WIDTH).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=env_int("CAM_HEIGHT", 720),
        help="Capture height (env: CAM_HEIGHT).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=env_int("CAM_JPEG_QUALITY", 90),
        help="JPEG quality 1-100 (env: CAM_JPEG_QUALITY).",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=env_int("CAM_KEEP_LAST", 0),
        help="If >0, keep only last N images (env: CAM_KEEP_LAST).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=env_str("CAM_PREFIX", "capture"),
        help="Filename prefix (env: CAM_PREFIX).",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        default=env_bool("CAM_NO_BANNER", True),
        help="Disable fswebcam banner (env: CAM_NO_BANNER).",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=env_float("CAM_WARMUP_SECONDS", 0.5),
        help="Warmup time before capture (env: CAM_WARMUP_SECONDS).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=env_int("CAM_ROTATE", 0),
        help="Rotate image (0/90/180/270) (env: CAM_ROTATE).",
    )

    args = parser.parse_args()

    interval_s = max(0.1, float(args.interval_seconds))
    output_dir = Path(args.output_dir).expanduser()
    device = str(args.device)
    width = max(1, int(args.width))
    height = max(1, int(args.height))
    jpeg_quality = int(args.jpeg_quality)
    jpeg_quality = 95 if jpeg_quality > 100 else (1 if jpeg_quality < 1 else jpeg_quality)
    keep_last = int(args.keep_last)
    keep_last = 0 if keep_last < 0 else keep_last
    prefix = str(args.prefix).strip() or "capture"
    warmup_s = max(0.0, float(args.warmup_seconds))
    rotate = int(args.rotate)
    rotate = rotate if rotate in {0, 90, 180, 270} else 0

    return Config(
        interval_s=interval_s,
        output_dir=output_dir,
        device=device,
        resolution=(width, height),
        jpeg_quality=jpeg_quality,
        keep_last=keep_last,
        prefix=prefix,
        no_banner=bool(args.no_banner),
        warmup_s=warmup_s,
        rotate=rotate,
    )


def main() -> int:
    cfg = build_config_from_args()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print(
        "[capture-usb-cam] starting",
        {
            "interval_s": cfg.interval_s,
            "output_dir": str(cfg.output_dir),
            "device": cfg.device,
            "resolution": f"{cfg.resolution[0]}x{cfg.resolution[1]}",
            "jpeg_quality": cfg.jpeg_quality,
            "keep_last": cfg.keep_last,
            "prefix": cfg.prefix,
            "backend": "fswebcam" if shutil.which("fswebcam") else "opencv(if available)",
        },
    )

    next_deadline = time.monotonic()
    while not _STOP:
        next_deadline += cfg.interval_s
        try:
            out = capture_once(cfg)
            print("[capture-usb-cam] saved", str(out))
        except Exception as e:
            print("[capture-usb-cam] error", repr(e))
            if not shutil.which("fswebcam"):
                print(
                    "[capture-usb-cam] hint: install fswebcam (recommended) or add OpenCV. "
                    "On Raspberry Pi OS: sudo apt-get update && sudo apt-get install -y fswebcam"
                )

        sleep_s = max(0.0, next_deadline - time.monotonic())
        if sleep_s > 0:
            time.sleep(sleep_s)

    print("[capture-usb-cam] stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

