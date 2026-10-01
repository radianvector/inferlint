"""Charts as SVG strings, computed from data. No plotting library, no network.

Every coordinate comes from one pair of linear scales, so marks, ticks and labels always
agree. Colours are never written into the SVG: marks carry a class (``s1``..``s3``,
``ev``), and the page's stylesheet maps classes to theme tokens, so one chart renders
correctly in light and dark. Text is escaped; labels are data and are never trusted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from html import escape
from typing import Literal

__all__ = [
    "Bar",
    "BarChart",
    "Event",
    "Line",
    "RefLine",
    "Scale",
    "TimeChart",
    "nice_ticks",
    "render_bars",
    "render_time",
]

Fmt = Callable[[float], str]


def _num(v: float) -> str:
    """Compact SVG number: at most two decimals, no trailing zeros."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]: steps of 1, 2, 2.5 or 5 times a power of 10."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("tick range must be finite")
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-12)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step
    n = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(n + 1)]


@dataclass(frozen=True)
class Scale:
    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


@dataclass(frozen=True)
class Line:
    label: str
    ys: Sequence[float | None]
    cls: str = "s1"  # s1..s3: the categorical slot, mapped to a colour by the page
    kind: Literal["line", "step", "area"] = "line"


@dataclass(frozen=True)
class RefLine:
    y: float
    label: str


@dataclass(frozen=True)
class Event:
    x: float
    label: str


@dataclass
class TimeChart:
    id: str
    xs: Sequence[float]
    lines: Sequence[Line]
    y_unit: str
    x_unit: str = "s"
    y_max: float | None = None
    y_fmt: Fmt = field(default=lambda v: f"{v:,.0f}")
    refs: Sequence[RefLine] = ()
    events: Sequence[Event] = ()
    event_label: str = "event"
    aria: str = ""
    width: int = 720
    height: int = 250


# Plot margins, in viewBox units. Right margin holds the direct end labels.
_ML, _MR, _MT, _MB = 48, 92, 18, 34


def _path(xs: Sequence[float], ys: Sequence[float | None], sx: Scale, sy: Scale, step: bool) -> str:
    """SVG path through the points. ``None`` breaks the line; a gap is never bridged."""
    parts: list[str] = []
    pen = False
    for x, y in zip(xs, ys, strict=True):
        if y is None:
            pen = False
            continue
        px, py = sx(x), sy(y)
        if not pen:
            parts.append(f"M{_num(px)} {_num(py)}")
            pen = True
        elif step:
            parts.append(f"H{_num(px)}V{_num(py)}")
        else:
            parts.append(f"L{_num(px)} {_num(py)}")
    return "".join(parts)


def _area(xs: Sequence[float], ys: Sequence[float | None], sx: Scale, sy: Scale) -> str:
    """Closed wash under each unbroken run of points, down to the zero baseline."""
    out: list[str] = []
    run: list[tuple[float, float]] = []

    def close() -> None:
        if len(run) > 1:
            pts = "L".join(f"{_num(sx(x))} {_num(sy(y))}" for x, y in run)
            base = _num(sy(0.0))
            out.append(f"M{_num(sx(run[0][0]))} {base}L{pts}L{_num(sx(run[-1][0]))} {base}Z")
        run.clear()

    for x, y in zip(xs, ys, strict=True):
        if y is None:
            close()
        else:
            run.append((x, y))
    close()
    return "".join(out)


def _json_script(obj: object) -> str:
    # "</" inside a <script> element would end it early.
    text = json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" class="chart-data">{text}</script>'


def render_time(c: TimeChart) -> str:
    """A time-series chart: lines on one y scale, optional reference lines and events."""
    if not c.xs:
        raise ValueError("chart has no x values")
    for ln in c.lines:
        if len(ln.ys) != len(c.xs):
            raise ValueError(f"series {ln.label!r} has {len(ln.ys)} values for {len(c.xs)} xs")
    w, h = c.width, c.height
    x0, x1 = c.xs[0], c.xs[-1]
    ys_all = [y for ln in c.lines for y in ln.ys if y is not None] + [r.y for r in c.refs]
    y_top = c.y_max if c.y_max is not None else max(ys_all or [1.0])
    yt = nice_ticks(0.0, y_top)
    xt = [t for t in nice_ticks(x0, x1, 7) if x0 - 1e-9 <= t <= x1 + 1e-9]
    sx = Scale(x0, x1, _ML, w - _MR)
    sy = Scale(yt[0], yt[-1], h - _MB, _MT)

    o: list[str] = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(c.aria)}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]
    for t in yt:
        y = _num(sy(t))
        cls = "axis" if t == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{_ML}" x2="{w - _MR}" y1="{y}" y2="{y}"/>')
        o.append(
            f'<text class="tick" x="{_ML - 8}" y="{y}" text-anchor="end" '
            f'dominant-baseline="middle">{escape(c.y_fmt(t))}</text>'
        )
    for t in xt:
        o.append(
            f'<text class="tick" x="{_num(sx(t))}" y="{h - _MB + 18}" '
            f'text-anchor="middle">{escape(_num(t))}</text>'
        )
    o.append(
        f'<text class="tick unit" x="{w - _MR + 16}" y="{h - _MB + 18}">{escape(c.x_unit)}</text>'
    )
    for r in c.refs:
        y = _num(sy(r.y))
        o.append(f'<line class="ref" x1="{_ML}" x2="{w - _MR}" y1="{y}" y2="{y}"/>')
        o.append(
            f'<text class="lab" x="{w - _MR + 8}" y="{y}" dominant-baseline="middle">'
            f"{escape(r.label)}</text>"
        )
    for ln in c.lines:
        if ln.kind == "area":
            o.append(f'<path class="wash {ln.cls}" d="{_area(c.xs, ln.ys, sx, sy)}"/>')
        d = _path(c.xs, ln.ys, sx, sy, step=ln.kind == "step")
        o.append(f'<path class="mark {ln.cls}" d="{d}"/>')

    # Direct end labels, only when they cannot collide; the legend carries identity anyway.
    ends: list[tuple[float, Line]] = []
    for ln in c.lines:
        last = next((y for y in reversed(ln.ys) if y is not None), None)
        if last is not None:
            ends.append((sy(last), ln))
    ref_ys = [sy(r.y) for r in c.refs]
    placed = sorted(y for y, _ in ends) + sorted(ref_ys)
    gaps_ok = all(abs(a - b) >= 14 for i, a in enumerate(placed) for b in placed[i + 1 :])
    if len(c.lines) > 1 and gaps_ok:
        for y, ln in ends:
            o.append(
                f'<text class="lab" x="{w - _MR + 8}" y="{_num(y)}" '
                f'dominant-baseline="middle">{escape(ln.label)}</text>'
            )

    for ev in c.events:
        x = sx(ev.x)
        top = _MT - 2
        o.append(
            f'<path class="ev" d="M{_num(x - 5)} {_num(top)}H{_num(x + 5)}L{_num(x)} '
            f'{_num(top + 8)}Z"><title>{escape(ev.label)}</title></path>'
        )

    o.append(f'<line class="cross" x1="0" x2="0" y1="{_MT}" y2="{h - _MB}" visibility="hidden"/>')
    o.append(
        f'<rect class="hit" x="{_ML}" y="{_MT}" width="{w - _ML - _MR}" '
        f'height="{h - _MT - _MB}" tabindex="0" '
        f'aria-label="{escape(c.aria)}. Use the arrow keys to read values."/>'
    )
    o.append("</svg>")
    data = {
        "w": w,
        "x0": x0,
        "x1": x1,
        "px0": _ML,
        "px1": w - _MR,
        "x": [round(x, 3) for x in c.xs],
        "xUnit": c.x_unit,
        "yUnit": c.y_unit,
        "series": [
            {
                "label": ln.label,
                "cls": ln.cls,
                "ys": [None if y is None else round(y, 3) for y in ln.ys],
            }
            for ln in c.lines
        ],
        "events": [{"x": round(e.x, 3), "label": e.label} for e in c.events],
    }
    return "".join(o) + _json_script(data)


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    tip: str = ""


@dataclass
class BarChart:
    id: str
    bars: Sequence[Bar]
    unit: str
    fmt: Fmt = field(default=lambda v: f"{v:,.1f}")
    tick_fmt: Fmt = field(default=lambda v: f"{v:,.0f}" if v == int(v) else f"{v:,.1f}")
    aria: str = ""
    width: int = 720
    thickness: int = 18
    gap: int = 12
    label_width: int = 120


def render_bars(c: BarChart) -> str:
    """Horizontal bars from one baseline: 4px rounded data end, square at the baseline."""
    if not c.bars:
        raise ValueError("bar chart has no bars")
    top, bottom = 8, 30
    h = top + bottom + len(c.bars) * (c.thickness + c.gap) - c.gap
    x_left, x_right = c.label_width, c.width - 72
    xt = nice_ticks(0.0, max(b.value for b in c.bars), 5)
    sx = Scale(0.0, xt[-1], x_left, x_right)
    o = [
        f'<svg viewBox="0 0 {c.width} {h}" role="img" aria-label="{escape(c.aria)}" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]
    for t in xt:
        x = _num(sx(t))
        cls = "axis" if t == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{x}" x2="{x}" y1="{top - 4}" y2="{h - bottom + 4}"/>')
        o.append(
            f'<text class="tick" x="{x}" y="{h - bottom + 20}" text-anchor="middle">'
            f"{escape(c.tick_fmt(t))}</text>"
        )
    # After the last tick's label, which is centred on x_right: about 6.5 px a character.
    unit_x = x_right + 3.3 * len(c.tick_fmt(xt[-1])) + 8
    o.append(
        f'<text class="tick unit" x="{_num(unit_x)}" y="{h - bottom + 20}">{escape(c.unit)}</text>'
    )
    r = 4.0
    for i, b in enumerate(c.bars):
        y = top + i * (c.thickness + c.gap)
        x_end = sx(b.value)
        length = x_end - x_left
        rr = min(r, max(length, 0.0), c.thickness / 2)
        t = c.thickness
        d = (
            f"M{_num(x_left)} {_num(y)}H{_num(x_end - rr)}"
            f"A{_num(rr)} {_num(rr)} 0 0 1 {_num(x_end)} {_num(y + rr)}"
            f"V{_num(y + t - rr)}"
            f"A{_num(rr)} {_num(rr)} 0 0 1 {_num(x_end - rr)} {_num(y + t)}"
            f"H{_num(x_left)}Z"
        )
        tip = b.tip or f"{b.label}: {c.fmt(b.value)} {c.unit}"
        o.append(f'<path class="bar s1" d="{d}" tabindex="0" data-tip="{escape(tip)}"/>')
        o.append(
            f'<text class="lab" x="{x_left - 10}" y="{_num(y + t / 2)}" text-anchor="end" '
            f'dominant-baseline="middle">{escape(b.label)}</text>'
        )
        o.append(
            f'<text class="val" x="{_num(x_end + 8)}" y="{_num(y + t / 2)}" '
            f'dominant-baseline="middle">{escape(c.fmt(b.value))}</text>'
        )
    o.append("</svg>")
    return "".join(o)
