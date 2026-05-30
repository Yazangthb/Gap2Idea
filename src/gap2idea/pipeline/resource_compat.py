"""Tiny cross-platform shim around `resource` for sandbox memory/CPU limits.

The POSIX `resource` module does not exist on Windows. The sanity-stage
sandbox uses `preexec_fn` to call `setrlimit` for memory and CPU caps; on
Windows those caps are best-effort (wall-clock timeout is the only hard
backstop). This shim makes the same call site work on both platforms by
exposing:

  • `AVAILABLE` — True iff real `resource` is importable.
  • `RLIMIT_AS`, `RLIMIT_CPU` — POSIX constants, or `None` on Windows.
  • `setrlimit(which, limits)` — calls the real function on POSIX, no-op
    on Windows.

Importers can guard the `preexec_fn` setup with `if AVAILABLE:`.
"""
from __future__ import annotations

try:
    import resource as _r
    AVAILABLE = True
    RLIMIT_AS  = _r.RLIMIT_AS
    RLIMIT_CPU = _r.RLIMIT_CPU
    setrlimit  = _r.setrlimit  # type: ignore[assignment]
except ImportError:  # pragma: no cover - Windows path
    AVAILABLE = False
    RLIMIT_AS  = None
    RLIMIT_CPU = None

    def setrlimit(*args, **kwargs) -> None:  # type: ignore[misc]
        # No-op on Windows. The sandbox falls back to wall-clock timeout.
        return None
