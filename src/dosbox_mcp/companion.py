"""Authenticated protocol-v2 client for the DOSBox-X companion channel.

Messages are UTF-8 JSON records behind a four-byte little-endian length.  The
framing is deliberately transport independent: Windows uses a named pipe and
POSIX uses a same-user Unix-domain socket.  Every response is correlated by
request id and every emitted event carries the emulator session id, monotonic
event sequence, and emulator-owned counters.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT = 15.0


class CompanionError(RuntimeError):
    """The emulator companion refused or failed a request."""


def endpoint_for(pid: int) -> str:
    if os.name == "nt":
        return rf"\\.\pipe\dosbox-x-mcp-{pid}"
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/dosbox-x-mcp-{os.getuid()}"))
    return str(runtime / f"dosbox-x-mcp-{pid}.sock")


def encode_message(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise CompanionError(f"companion message is {len(body)} bytes; limit is {MAX_MESSAGE_BYTES}")
    return struct.pack("<I", len(body)) + body


def decode_message(data: bytes) -> dict[str, Any]:
    if len(data) < 4:
        raise CompanionError("truncated companion frame header")
    size = struct.unpack_from("<I", data)[0]
    if size > MAX_MESSAGE_BYTES:
        raise CompanionError(f"companion frame declares {size} bytes; limit is {MAX_MESSAGE_BYTES}")
    if len(data) != size + 4:
        raise CompanionError(f"companion frame declares {size} bytes but contains {len(data) - 4}")
    try:
        value = json.loads(data[4:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise CompanionError(f"malformed companion JSON: {failure}") from None
    if not isinstance(value, dict):
        raise CompanionError("companion message must be a JSON object")
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise CompanionError("companion disconnected mid-message")
        chunks += part
    return bytes(chunks)


def read_message(stream: BinaryIO) -> dict[str, Any]:
    header = _read_exact(stream, 4)
    size = struct.unpack("<I", header)[0]
    if size > MAX_MESSAGE_BYTES:
        raise CompanionError(f"companion frame declares {size} bytes; limit is {MAX_MESSAGE_BYTES}")
    return decode_message(header + _read_exact(stream, size))


@dataclass(frozen=True)
class Counters:
    pic_us: int
    video_frame: int
    cpu_cycles: int
    memory_tick: int | None = None

    @classmethod
    def from_message(cls, value: object) -> "Counters":
        if not isinstance(value, dict):
            raise CompanionError("event has no counter object")
        try:
            return cls(
                pic_us=int(value["pic_us"]),
                video_frame=int(value["video_frame"]),
                cpu_cycles=int(value["cpu_cycles"]),
                memory_tick=(None if value.get("memory_tick") is None else int(value["memory_tick"])),
            )
        except (KeyError, TypeError, ValueError):
            raise CompanionError("event counters are incomplete or non-numeric") from None


class CompanionClient:
    """One request-at-a-time client with strict identity and sequence checks."""

    def __init__(self, pid: int, token: str, endpoint: str | None = None) -> None:
        if not token or len(token) > 256:
            raise CompanionError("a non-empty launch token of at most 256 characters is required")
        self.pid = int(pid)
        self.token = token
        self.endpoint = endpoint or endpoint_for(pid)
        self.session_id: str | None = None
        self.last_sequence = -1
        self.capabilities: dict[str, Any] = {}

    def _connect(self, timeout: float = DEFAULT_TIMEOUT) -> BinaryIO:
        if os.name == "nt":
            deadline = time.monotonic() + timeout
            failure: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    return open(self.endpoint, "r+b", buffering=0)
                except OSError as error:
                    # The server disconnects and relistens on the same pipe
                    # instance after each atomic request.  Opening during that
                    # tiny handover reports EINVAL or PIPE_BUSY on Windows.
                    failure = error
                    time.sleep(0.01)
            raise CompanionError(f"cannot open companion named pipe {self.endpoint}: {failure}") from None
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout)
        try:
            connection.connect(self.endpoint)
        except OSError as failure:
            connection.close()
            raise CompanionError(f"cannot connect to companion socket {self.endpoint}: {failure}") from None
        return connection.makefile("rwb", buffering=0)

    def request(self, operation: str, payload: dict[str, Any] | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        message = {
            "protocol": PROTOCOL_VERSION,
            "type": "request",
            "request_id": request_id,
            "token": self.token,
            "operation": operation,
            "payload": payload or {},
        }
        with self._connect(timeout) as stream:
            stream.write(encode_message(message))
            response = read_message(stream)
        if response.get("protocol") != PROTOCOL_VERSION:
            raise CompanionError(f"companion protocol {response.get('protocol')!r} is not supported; expected 2")
        if response.get("request_id") != request_id:
            raise CompanionError("stale or mismatched companion response")
        if response.get("status") != "ok":
            raise CompanionError(str(response.get("error") or "companion request failed"))
        session_id = response.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise CompanionError("companion response has no session identity")
        if self.session_id is not None and session_id != self.session_id:
            raise CompanionError("companion session identity changed; pending state is invalid")
        self.session_id = session_id
        events = response.get("events", [])
        if not isinstance(events, list):
            raise CompanionError("companion events must be a list")
        for event in events:
            if not isinstance(event, dict):
                raise CompanionError("companion event must be an object")
            sequence = int(event.get("sequence", -1))
            if sequence <= self.last_sequence:
                raise CompanionError("companion event sequence is stale or out of order")
            if event.get("session_id") != session_id:
                raise CompanionError("companion event belongs to another session")
            Counters.from_message(event.get("counters"))
            self.last_sequence = sequence
        result = response.get("result")
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise CompanionError("companion result must be an object")
        return {**result, "request_id": request_id, "session_id": session_id, "events": events}

    def negotiate(self) -> dict[str, Any]:
        result = self.request("negotiate", {"versions": [PROTOCOL_VERSION]})
        capabilities = result.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise CompanionError("capability negotiation returned a non-object")
        self.capabilities = capabilities
        return result

    def run_timeline(self, clock: str, events: list[dict[str, Any]], timeout: float, expire_after_counter: int | None = None) -> dict[str, Any]:
        if clock not in {"video_frame", "pic_us", "cpu_cycles", "wall_clock"}:
            raise CompanionError(f"unsupported timeline clock {clock!r}")
        if not 1 <= len(events) <= 4096:
            raise CompanionError("a timeline needs 1..4096 events")
        payload: dict[str, Any] = {"clock": clock, "events": events}
        # A second emulator-owned time domain prevents a stopped VGA counter
        # from turning a bounded frame experiment into an unbounded pipe read.
        if clock != "pic_us":
            payload["expire_after_pic_us"] = max(1, int(timeout * 1_000_000))
        if expire_after_counter is not None:
            if expire_after_counter < 1:
                raise CompanionError("timeline expiry must be positive")
            payload["expire_after_counter"] = expire_after_counter
        result = self.request("run_timeline", payload, timeout=timeout)
        actual = result.get("timeline_events")
        if not isinstance(actual, list) or len(actual) != len(events):
            raise CompanionError("companion did not acknowledge every timeline event")
        if clock != "wall_clock" and any(item.get("actual_counter") is None for item in actual):
            raise CompanionError("an emulator-time event lacks its actual counter")
        return result


__all__ = ["CompanionClient", "CompanionError", "Counters", "PROTOCOL_VERSION", "decode_message", "encode_message", "endpoint_for", "read_message"]
