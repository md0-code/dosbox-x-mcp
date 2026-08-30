"""Bounded writer-event aggregation shared by companion traces and bundles."""
from __future__ import annotations

from typing import Any, Iterable


class WriteWatchError(ValueError):
    pass


def aggregate(events: Iterable[dict[str, Any]], *, physical_start: int, length: int, hit_limit: int = 10000, output_limit: int = 1000) -> dict[str, Any]:
    if physical_start < 0 or not 1 <= length <= 0x10000:
        raise WriteWatchError("write watch must cover 1..65536 bytes at a non-negative physical address")
    if not 1 <= hit_limit <= 1_000_000 or not 1 <= output_limit <= 100_000:
        raise WriteWatchError("invalid write-watch hit or output limit")
    end = physical_start + length
    groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    total = 0
    truncated = False
    for event in events:
        address = int(event["physical_address"])
        if not physical_start <= address < end:
            continue
        if total >= hit_limit:
            truncated = True
            break
        total += 1
        cs, ip = int(event["cs"]), int(event["ip"])
        width = int(event.get("width", 1))
        key = (cs, ip, address)
        item = groups.setdefault(key, {"writer": f"{cs:04X}:{ip:04X}", "physical_address": address, "width": width, "hits": 0, "first": event, "last": event})
        item["hits"] += 1
        item["last"] = event
    ranked = sorted(groups.values(), key=lambda item: (-item["hits"], item["writer"], item["physical_address"]))
    return {"physical_start": physical_start, "length": length, "hits": total, "unique_writers": len(ranked), "truncated": truncated or len(ranked) > output_limit, "writers": ranked[:output_limit]}


__all__ = ["WriteWatchError", "aggregate"]
