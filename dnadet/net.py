"""Shared network setup.

Exists for one reason: a stock macOS python.org interpreter ships with no root
CA bundle (`ssl.get_default_verify_paths().cafile` is None), so every HTTPS call
fails with CERTIFICATE_VERIFY_FAILED even though curl on the same machine
succeeds - curl uses the system keychain, Python does not. Relying on the
operator to export SSL_CERT_FILE means the pipeline works in the shell where
someone remembered and fails in every other one, which is exactly how a demo
breaks in front of an audience.

Verification is never disabled. An unverified TLS connection would make every
"retrieved live from ClinVar" claim in the evidence log unfalsifiable.
"""

from __future__ import annotations

import os
import ssl

_WARNED = {"done": False}

UA = "DNA-Detective/1.0 (SZU Summer Camp 2026)"

JSON_HEADERS = {
    "Content-Type": "application/json",
    # Cloudflare fronts several APIs used here and rejects urllib's default
    # "Python-urllib/3.12" with 403 error code 1010 (banned browser signature).
    "User-Agent": UA,
    "Accept": "application/json",
}


def ssl_context() -> ssl.SSLContext:
    """A verifying context that works without SSL_CERT_FILE being exported."""
    try:
        import certifi  # noqa: PLC0415 - optional, probed at runtime

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        if (ssl.get_default_verify_paths().cafile is None
                and not os.environ.get("SSL_CERT_FILE")
                and not _WARNED["done"]):
            _WARNED["done"] = True
            print("  ! no CA bundle found - HTTPS will fail. Fix with:\n"
                  "    python3 -m pip install certifi")
        return ctx
