from __future__ import annotations

import json
import re
from xml.etree import ElementTree as ET

import pytest

from inferlint.svgchart import (
    Bar,
    BarChart,
    Event,
    Line,
    RefLine,
    Scale,
    TimeChart,
    nice_ticks,
    render_bars,
    render_time,
)

SVG_NS = "{http://www.w3.org/2000/svg}"


def svg_of(html: str) -> ET.Element:
    body = html.split("<script", 1)[0].replace(
        "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
    )
    return ET.fromstring(body)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0, 32, [0, 10, 20, 30, 40]),
        (0, 100, [0, 25, 50, 75, 100]),
        (0, 10, [0, 2.5, 5, 7.5, 10]),
        (0, 0.93, [0, 0.25, 0.5, 0.75, 1.0]),
        (0, 93.7, [0, 25, 50, 75, 100]),
        (5, 5, [5, 5.25, 5.5, 5.75, 6]),
    ],
)
def test_nice_ticks(lo: float, hi: float, expected: list[float]) -> None:
    assert nice_ticks(lo, hi) == expected


def test_ticks_cover_the_range() -> None:
    for hi in (0.3, 7, 42, 341.44, 21768, 1e6 + 3):
        t = nice_ticks(0, hi)
        assert t[0] <= 0 and t[-1] >= hi and len(t) <= 7


def test_scale() -> None:
    s = Scale(0, 10, 100, 200)
    assert (s(0), s(5), s(10)) == (100, 150, 200)
    assert Scale(0, 10, 200, 100)(10) == 100  # inverted y


def chart(**kw: object) -> TimeChart:
    base: dict[str, object] = {
        "id": "t",
        "xs": [0.0, 1.0, 2.0, 3.0],
        "lines": [Line("running", [1.0, 2.0, None, 4.0])],
        "y_unit": "req",
    }
    base.update(kw)
    return TimeChart(**base)  # type: ignore[arg-type]


def test_gap_breaks_the_line() -> None:
    root = svg_of(render_time(chart()))
    d = next(p.get("d") for p in root.iter(f"{SVG_NS}path") if "mark" in (p.get("class") or ""))
    assert d is not None and d.count("M") == 2  # two runs, never bridged across None


def test_step_line_uses_horizontal_then_vertical() -> None:
    root = svg_of(render_time(chart(lines=[Line("fits", [3.0, 3.0, 2.0, 2.0], "s3", "step")])))
    d = next(p.get("d") for p in root.iter(f"{SVG_NS}path") if "mark" in (p.get("class") or ""))
    assert d is not None and "H" in d and "V" in d and "L" not in d


def test_all_marks_inside_the_plot() -> None:
    c = chart(lines=[Line("a", [0.0, 5.0, 9.0, 3.0]), Line("b", [2.0, 2.0, 2.0, 2.0], "s2")])
    html = render_time(c)
    nums = [
        float(v)
        for d in re.findall(r'class="mark[^"]*" d="([^"]+)"', html)
        for v in re.findall(r"-?\d+(?:\.\d+)?", d)
    ]
    xs, ys = nums[0::2], nums[1::2]
    assert min(xs) >= 48 and max(xs) <= c.width - 92
    assert min(ys) >= 18 and max(ys) <= c.height - 34


def test_labels_are_escaped() -> None:
    html = render_time(
        chart(
            lines=[Line("<script>alert(1)</script>", [1.0, 2.0, 3.0, 4.0])],
            refs=[RefLine(2.0, 'x"><b>')],
        )
    )
    assert "<script>alert" not in html.split('<script type="application/json"')[0]
    assert "&lt;script&gt;" in html or "\\u003cscript" in html or "<\\/script>" in html


def test_json_payload_cannot_close_its_script() -> None:
    html = render_time(chart(lines=[Line("</script><b>", [1.0, 2.0, 3.0, 4.0])]))
    payload = html.split('<script type="application/json" class="chart-data">', 1)[1]
    inner = payload.rsplit("</script>", 1)[0]
    assert "</script>" not in inner
    data = json.loads(inner.replace("<\\/", "</"))
    assert data["series"][0]["ys"] == [1.0, 2.0, 3.0, 4.0]


def test_events_become_markers() -> None:
    html = render_time(chart(events=[Event(1.0, "2 preemptions"), Event(3.0, "1 preemption")]))
    assert html.count('class="ev"') == 2
    assert "2 preemptions" in html


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="values for"):
        render_time(chart(lines=[Line("a", [1.0])]))


def test_data_changes_the_picture() -> None:
    a = render_time(chart(lines=[Line("a", [1.0, 2.0, 3.0, 4.0])]))
    b = render_time(chart(lines=[Line("a", [1.0, 2.0, 3.0, 1.0])]))
    assert a != b


def test_bars_are_proportional_and_rounded_at_the_end_only() -> None:
    html = render_bars(BarChart("b", [Bar("10 running", 30.0), Bar("7 running", 15.0)], "s"))
    ds = re.findall(r'class="bar s1" d="([^"]+)"', html)
    assert len(ds) == 2
    ends = [float(re.search(r"A[\d.]+ [\d.]+ 0 0 1 ([\d.]+)", d).group(1)) for d in ds]  # type: ignore[union-attr]
    start = 120.0
    assert (ends[0] - start) == pytest.approx(2 * (ends[1] - start), abs=0.02)
    assert all(d.startswith("M120 ") and d.endswith("H120Z") for d in ds)  # square baseline


def test_bars_need_data() -> None:
    with pytest.raises(ValueError):
        render_bars(BarChart("b", [], "s"))
