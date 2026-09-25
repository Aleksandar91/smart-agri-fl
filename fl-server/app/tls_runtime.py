"""Optional gRPC mTLS for the Flower server.

Flower 1.32 already accepts a (CA, server cert, server key) tuple, but
``generic_create_grpc_server`` hard-codes ``require_client_auth=False``.
When ``FL_TLS_CERTS_DIR`` is set we patch that flag so a client without a
CA-signed certificate cannot complete the handshake.

This is transport security only. It does not stop a provisioned farm from
sending a poisoned update.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _must_read(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit(f"TLS enabled but missing certificate file: {path}")
    data = path.read_bytes()
    if not data:
        raise SystemExit(f"TLS certificate file is empty: {path}")
    return data


def install_require_client_auth() -> None:
    """Force Flower's gRPC server to request and verify client certificates."""
    import grpc

    original = grpc.ssl_server_credentials

    def patched(
        private_key_certificate_chain_pairs,
        root_certificates=None,
        require_client_auth=False,
        *args,
        **kwargs,
    ):
        kwargs.pop("require_client_auth", None)
        return original(
            private_key_certificate_chain_pairs,
            root_certificates=root_certificates,
            require_client_auth=root_certificates is not None,
            *args,
            **kwargs,
        )

    grpc.ssl_server_credentials = patched  # type: ignore[method-assign]


def load_server_certificates() -> tuple[bytes, bytes, bytes] | None:
    """Return Flower ``certificates=`` tuple, or None to keep plaintext gRPC."""
    certs_dir = _env_str("FL_TLS_CERTS_DIR")
    if not certs_dir:
        return None
    root = Path(certs_dir)
    ca_path = Path(_env_str("FL_TLS_CA_CERT") or str(root / "ca.crt"))
    cert_path = Path(_env_str("FL_TLS_SERVER_CERT") or str(root / "server.crt"))
    key_path = Path(_env_str("FL_TLS_SERVER_KEY") or str(root / "server.key"))
    ca = _must_read(ca_path)
    cert = _must_read(cert_path)
    key = _must_read(key_path)
    require_client = _env_bool("FL_TLS_REQUIRE_CLIENT_AUTH", True)
    if require_client:
        install_require_client_auth()
        mode = "mtls"
    else:
        mode = "tls-server-only"
    print(
        "[fl-server] tls",
        {
            "mode": mode,
            "ca": str(ca_path),
            "cert": str(cert_path),
        },
    )
    return ca, cert, key
