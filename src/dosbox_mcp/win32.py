"""Backwards-compatible re-export of the platform backend.

This module was the Win32 layer before the backends were split apart.  It
stays because scripts outside this package import ``dosbox_mcp.win32.Process``
and ``dosbox_mcp.win32.find_windows`` directly, and a port that does not
concern them should not make them change.

New code should import :mod:`dosbox_mcp.platform` instead, which resolves to
the right backend for the host.
"""
from __future__ import annotations

from .platform import (  # noqa: F401
    BaseProcess,
    Capabilities,
    Process,
    ProcessError,
    Window,
    capabilities,
    client_size,
    find_windows,
    grab_client,
    menu_command,
    menu_items,
    menu_window,
    resize_client,
    resolve_command,
    terminate,
)

__all__ = [
    "BaseProcess",
    "Capabilities",
    "Process",
    "ProcessError",
    "Window",
    "capabilities",
    "client_size",
    "find_windows",
    "grab_client",
    "menu_command",
    "menu_items",
    "menu_window",
    "resize_client",
    "resolve_command",
    "terminate",
]
