"""The one result type every check returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["LABELS", "CheckResult", "Status", "TripwireFailed"]


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    # The tripwire caught its problem. For T7 and T12 that means the server failed;
    # for every other tripwire it means a number from the run is not what it seems.
    FAIL = "fail"
    # The instrument could not decide: a series was missing, a line did not parse.
    # Never folded into PASS. A check that cannot see is not a check that passed.
    UNKNOWN = "unknown"


# How each status is printed. The JSON keeps Status.value ("fail"), which scripts read.
LABELS: dict[Status, str] = {
    Status.PASS: "PASS",
    Status.WARN: "WARN",
    Status.FAIL: "TRIPWIRE-FAILED",
    Status.UNKNOWN: "????",
}


class TripwireFailed(AssertionError):
    def __init__(self, result: CheckResult) -> None:
        super().__init__(f"[{result.tripwire}] {LABELS[result.status]}: {result.message}")
        self.result = result


@dataclass(frozen=True)
class CheckResult:
    tripwire: str
    status: Status
    message: str
    evidence: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def ok(self) -> bool:
        return self.status in (Status.PASS, Status.WARN)

    def raise_for_status(self, *, allow_warn: bool = True, allow_unknown: bool = False) -> None:
        bad = {Status.FAIL}
        if not allow_warn:
            bad.add(Status.WARN)
        if not allow_unknown:
            bad.add(Status.UNKNOWN)
        if self.status in bad:
            raise TripwireFailed(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "tripwire": self.tripwire,
            "status": self.status.value,
            "message": self.message,
            "evidence": self.evidence,
        }
