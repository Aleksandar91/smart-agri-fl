"""Optional gRPC mTLS for the Flower client.

Flower 1.32 ``start_client`` verifies the server with ``root_certificates``
but never presents a client certificate. When ``FL_TLS_CERTS_DIR`` is set we
patch ``grpc.ssl_channel_credentials`` so the handshake sends this farm's
cert/key. Plaintext remains the default (unset ``FL_TLS_CERTS_DIR``).
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


def _must_read(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit(f"TLS enabled but missing certificate file: {path}")
    data = path.read_bytes()
    if not data:
        raise SystemExit(f"TLS certificate file is empty: {path}")
    return data


def install_client_certificate(cert_pem: bytes, key_pem: bytes) -> None:
    """Inject this farm's cert into every secure gRPC channel Flower opens."""
    import grpc

    original = grpc.ssl_channel_credentials

    def patched(root_certificates=None, private_key=None, certificate_chain=None):
        return original(
            root_certificates=root_certificates,
            private_key=key_pem if private_key is None else private_key,
            certificate_chain=cert_pem if certificate_chain is None else certificate_chain,
        )

    grpc.ssl_channel_credentials = patched  # type: ignore[method-assign]


def prepare_client_tls(client_id: str) -> bytes | None:
    """Return CA bytes for ``start_client``, or None for plaintext gRPC."""
    certs_dir = _env_str("FL_TLS_CERTS_DIR")
    if not certs_dir:
        return None
    root = Path(certs_dir)
    ca_path = Path(_env_str("FL_TLS_CA_CERT") or str(root / "ca.crt"))
    cert_path = Path(_env_str("FL_TLS_CLIENT_CERT") or str(root / f"{client_id}.crt"))
    key_path = Path(_env_str("FL_TLS_CLIENT_KEY") or str(root / f"{client_id}.key"))
    ca = _must_read(ca_path)
    cert = _must_read(cert_path)
    key = _must_read(key_path)
    install_client_certificate(cert, key)
    print(
        "[fl-client] tls",
        {
            "mode": "mtls",
            "ca": str(ca_path),
            "cert": str(cert_path),
            "client_id": client_id,
        },
    )
    return ca
