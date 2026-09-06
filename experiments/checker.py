"""Tiny pass/fail collector shared by experiments."""

from __future__ import annotations


class Checker:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)
