"""Draw the gate's precision/recall curve as a standalone SVG file, with no plotting library.

Two series on one recall (x) by precision (y) plot: the cutoffs sorted by evidence tier then
confidence, and by confidence alone. The ungated review (estimated, equal to its official
score) and the gated review's official score are marked as single points. Colors are the
first two slots of a validated categorical palette (blue, orange), with darker steps for a
dark page; each point carries a native tooltip (<title>) with its cutoff and counts. A legend
names both series (end labels would collide where the lines meet); the two single points are
labeled next to them. gate_report.md holds the same numbers as tables.

Imported by other modules; the file isn't run directly.
"""

import math
from collections.abc import Sequence
from html import escape

from pr_reviewer.gate.curve import CurvePoint
from pr_reviewer.reviewer.compare import SubsetMetrics

# Plot size and margins, in pixels.
WIDTH, HEIGHT = 720, 460
LEFT, RIGHT, TOP, BOTTOM = 64, 150, 48, 56

_STYLE = """
  :root { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e4e3df;
          --s1: #2a78d6; --s2: #eb6834; }
  @media (prefers-color-scheme: dark) {
    :root { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --grid: #383835;
            --s1: #3987e5; --s2: #d95926; }
  }
  .bg { fill: var(--surface); }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2);
         font-size: 12px; }
  .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--ink2); stroke-width: 1; }
  .s1 { stroke: var(--s1); fill: none; stroke-width: 2; }
  .s2 { stroke: var(--s2); fill: none; stroke-width: 2; }
  .m1 { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
  .m2 { fill: var(--s2); stroke: var(--surface); stroke-width: 2; }
  .ref { fill: var(--surface); stroke: var(--ink); stroke-width: 2; }
  .label { fill: var(--ink); }
"""


def nice_top(value: float) -> float:
    """Round an axis maximum up to a tidy number.

    Args:
        value (float): The largest value to show.

    Returns:
        float: A round number at least as large, such as 0.25 for 0.21; 0.1 when value is 0.
    """
    if value <= 0:
        return 0.1
    step = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if value <= factor * step:
            return factor * step
    return 10.0 * step


def _tooltip(point: CurvePoint) -> str:
    """Write a point's tooltip text.

    Args:
        point (CurvePoint): The point.

    Returns:
        str: Its cutoff, counts and scores.
    """
    return (
        f"{point.label}: {point.comments} comments, {point.hits} hits, "
        f"{point.false_alarms} false alarms ({point.clean_false_alarms} on clean PRs); "
        f"precision {point.precision:.3f}, recall {point.recall:.3f}"
    )


def curve_svg(
    evidence: Sequence[CurvePoint],
    confidence: Sequence[CurvePoint],
    ungated: CurvePoint,
    gated: SubsetMetrics,
) -> str:
    """Draw both curves with the ungated and gated reviews marked.

    Args:
        evidence (Sequence[CurvePoint]): Cutoffs by tier, then confidence, strictest first.
        confidence (Sequence[CurvePoint]): Cutoffs by confidence alone, strictest first.
        ungated (CurvePoint): The whole ungated review.
        gated (SubsetMetrics): The gated review's official scores.

    Returns:
        str: The SVG file's text.
    """
    every = [*evidence, *confidence, ungated]
    x_top = nice_top(max([p.recall for p in every] + [gated.recall]))
    y_top = nice_top(max([p.precision for p in every] + [gated.precision]))
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(recall: float) -> float:
        return LEFT + plot_w * recall / x_top

    def y(precision: float) -> float:
        return TOP + plot_h * (1 - precision / y_top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        'aria-label="Precision against recall at each gate cutoff">',
        f"<style>{_STYLE}</style>",
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="title" x="{LEFT}" y="26">Precision and recall at each gate cutoff '
        "(estimated)</text>",
    ]
    # Five even steps: every nice_top() value (1, 2, 2.5 or 5 times a power of ten) splits
    # into round numbers.
    for i in range(6):
        fx, fy = round(x_top * i / 5, 6), round(y_top * i / 5, 6)
        parts.append(
            f'<line class="grid" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{y(fy):.1f}" '
            f'y2="{y(fy):.1f}"/>'
        )
        parts.append(f'<text x="{LEFT - 8}" y="{y(fy) + 4:.1f}" text-anchor="end">{fy:g}</text>')
        parts.append(
            f'<text x="{x(fx):.1f}" y="{TOP + plot_h + 18}" text-anchor="middle">{fx:g}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{TOP + plot_h}" '
        f'y2="{TOP + plot_h}"/>'
    )
    parts.append(
        f'<text x="{LEFT + plot_w / 2}" y="{HEIGHT - 14}" text-anchor="middle">Recall</text>'
    )
    parts.append(
        f'<text transform="translate(18 {TOP + plot_h / 2}) rotate(-90)" '
        'text-anchor="middle">Precision</text>'
    )
    # The legend names both lines; end labels would collide where the lines meet.
    for points, line, mark in ((confidence, "s2", "m2"), (evidence, "s1", "m1")):
        if not points:
            continue
        path = " ".join(f"{x(p.recall):.1f},{y(p.precision):.1f}" for p in points)
        parts.append(f'<polyline class="{line}" points="{path}"/>')
        for p in points:
            parts.append(
                f'<circle class="{mark}" cx="{x(p.recall):.1f}" cy="{y(p.precision):.1f}" r="4">'
                f"<title>{escape(_tooltip(p))}</title></circle>"
            )
    for label, recall, precision, tip in (
        ("Ungated", ungated.recall, ungated.precision, _tooltip(ungated)),
        (
            "Gated (official)",
            gated.recall,
            gated.precision,
            f"Gated, official score: {gated.hits} hits, {gated.false_alarms} false alarms; "
            f"precision {gated.precision:.3f}, recall {gated.recall:.3f}",
        ),
    ):
        parts.append(
            f'<rect class="ref" x="{x(recall) - 5:.1f}" y="{y(precision) - 5:.1f}" '
            f'width="10" height="10" rx="2"><title>{escape(tip)}</title></rect>'
        )
        parts.append(
            f'<text class="label" x="{x(recall) + 9:.1f}" y="{y(precision) - 8:.1f}">'
            f"{escape(label)}</text>"
        )
    legend_x = WIDTH - RIGHT + 16
    for row, (mark, name) in enumerate(
        (("m1", "Evidence tier, then confidence"), ("m2", "Confidence alone"))
    ):
        ly = TOP + 8 + row * 40
        parts.append(f'<circle class="{mark}" cx="{legend_x}" cy="{ly}" r="5"/>')
        first, _, rest = name.partition(", ")
        parts.append(f'<text x="{legend_x + 10}" y="{ly + 4}">{escape(first)}</text>')
        if rest:
            parts.append(f'<text x="{legend_x + 10}" y="{ly + 19}">{escape(rest)}</text>')
    ly = TOP + 88
    parts.append(
        f'<rect class="ref" x="{legend_x - 5}" y="{ly - 5}" width="10" height="10" rx="2"/>'
    )
    parts.append(f'<text x="{legend_x + 10}" y="{ly + 4}">Whole review</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
