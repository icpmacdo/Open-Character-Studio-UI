"""
Lightweight Tinker smoke test.

Usage:
    python tools/check_tinker.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        import tinker  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"Tinker import failed: {exc}")
        return 1

    if not os.getenv("TINKER_API_KEY"):
        print("TINKER_API_KEY is not set; export it and rerun.")
        return 1

    try:
        service_client = tinker.ServiceClient()
        capabilities = service_client.get_server_capabilities()
        supported_raw = getattr(capabilities, "supported_models", []) or []
        supported = [
            getattr(item, "model_name", getattr(item, "name", str(item))) for item in supported_raw
        ]
        print(f"Tinker OK. {len(supported)} supported models detected.")
        if supported:
            print("Examples:", ", ".join(supported[:5]) + (" ..." if len(supported) > 5 else ""))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Tinker capability probe failed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
