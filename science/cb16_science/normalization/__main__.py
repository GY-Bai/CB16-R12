"""``python -m cb16_science.normalization`` entrypoint (allowlisted)."""

from __future__ import annotations

from .experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
