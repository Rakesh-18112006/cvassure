"""Air-gap enforcement.

The problem statement requires the audit to run with no internet access. A
promise is not evidence, so ``--offline-assert`` replaces the socket layer with
one that raises. If any code path — ours or a library's — reaches for the
network, the run fails loudly instead of quietly succeeding on a laptop that
happened to have wifi.
"""

from __future__ import annotations

import socket
from typing import Any


class NetworkAccessAttempted(RuntimeError):
    """Raised when something tries to open a network connection."""


_original_socket = socket.socket
_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo
_armed = False


class _BlockedSocket(socket.socket):
    def __init__(self, *args: Any, **kwargs: Any):
        raise NetworkAccessAttempted(
            "cvassure was started with --offline-assert and something just tried "
            "to open a network connection. The audit is required to run air-gapped, "
            "so this is a failure, not a warning."
        )


def _blocked(*args: Any, **kwargs: Any) -> Any:
    raise NetworkAccessAttempted(
        "cvassure was started with --offline-assert and something just tried to "
        "reach the network. The audit is required to run air-gapped."
    )


def assert_offline() -> None:
    """Arm the guard. Idempotent."""
    global _armed
    if _armed:
        return
    socket.socket = _BlockedSocket  # type: ignore[misc,assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    _armed = True


def allow_network() -> None:
    """Undo the guard. Only used by the test suite."""
    global _armed
    socket.socket = _original_socket  # type: ignore[misc,assignment]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _original_getaddrinfo  # type: ignore[assignment]
    _armed = False


def is_armed() -> bool:
    return _armed
