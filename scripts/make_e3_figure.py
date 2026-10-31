"""E3 main exhibit: unsupported-claim rate against requirement coverage for variants A-D, one panel per fit
level plus an overall panel. Plain SVG, no plotting library needed (the venv pins old numpy/torch).

    python scripts/make_e3_figure.py
  -> runs/<run>/figures/e3_tradeoff.svg
Reads runs/<run>/e3_summary.csv. Top left is best: fewer unsupported claims, more requirements covered.
"""
import csv

from _common import load_config, parser, run_dir

COLORS = {"A": "#c0392b", "B": "#e67e22", "C": "#2980b9", "D": "#27ae60"}
NAMES = {"A": "A baseline", "B": "B grounded", "C": "C verify", "D": "D grounded + verify"}
PANELS = [("all", "All 36 pairs"), ("good", "Good fit (12)"), ("potential", "Potential fit (12)"), ("none", "No fit (12)")]
W, H, PW, PH, ML, MT = 1000, 545, 200, 330, 110, 75   # canvas, panel size, left margin, top margin
GAP = 28
XMAX, YMAX = 60.0, 55.0   # axis ranges in percent


def main(config):
    cfg = load_config(config)
    out = run_dir(cfg)
    rows = {(r["variant"], r["fit"]): r for r in csv.DictReader(open(out / "e3_summary.csv", encoding="utf-8"))}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">'
           'Fabrication against relevance: unsupported-claim rate and requirement coverage, by fit level</text>',
           f'<text x="{W / 2}" y="48" text-anchor="middle" fill="#555">Top left is best: fewer unsupported claims and more requirements covered</text>']
    for p, (fit, title) in enumerate(PANELS):
        x0, y0 = ML + p * (PW + GAP) - 50, MT
        svg.append(f'<text x="{x0 + PW / 2}" y="{y0 - 10}" text-anchor="middle" font-weight="bold">{title}</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{PW}" height="{PH}" fill="#fafafa" stroke="#999"/>')
        for t in range(0, int(XMAX) + 1, 20):   # x grid / ticks
            x = x0 + PW * t / XMAX
            svg.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + PH}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{x}" y="{y0 + PH + 16}" text-anchor="middle" fill="#555">{t}</text>')
        for t in range(0, int(YMAX) + 1, 10):   # y grid / ticks
            y = y0 + PH - PH * t / YMAX
            svg.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + PW}" y2="{y}" stroke="#e5e5e5"/>')
            if p == 0:
                svg.append(f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end" fill="#555">{t}</text>')
        svg.append(f'<text x="{x0 + PW / 2}" y="{y0 + PH + 36}" text-anchor="middle" fill="#333">unsupported claims (%)</text>')
        for v in "ABCD":
            r = rows[(v, fit)]
            ux, cv = 100 * float(r["unsupported_rate"]), 100 * float(r["coverage"])
            x, y = x0 + PW * ux / XMAX, y0 + PH - PH * cv / YMAX
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLORS[v]}" fill-opacity="0.85" stroke="white"/>')
            dx, dy, anchor = {"A": (0, -12, "middle"), "B": (0, 20, "middle"), "C": (11, 4, "start"), "D": (-11, 4, "end")}[v]
            svg.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" font-weight="bold" fill="{COLORS[v]}">{v}</text>')
    svg.append(f'<text x="14" y="{MT + PH / 2}" text-anchor="middle" fill="#333" transform="rotate(-90 14 {MT + PH / 2})">'
               'requirement coverage (%)</text>')
    lx = 60   # legend
    for v in "ABCD":
        svg.append(f'<circle cx="{lx}" cy="{H - 56}" r="6" fill="{COLORS[v]}"/>'
                   f'<text x="{lx + 12}" y="{H - 52}">{NAMES[v]}</text>')
        lx += 210
    svg.append(f'<text x="60" y="{H - 28}" fill="#666">Source: runs/{cfg["run_name"]}/e3_summary.csv. Unsupported claims judged by gpt-oss-120b '
               '(agreement with human labels 0.90, kappa 0.74).</text>')
    svg.append(f'<text x="60" y="{H - 12}" fill="#666">Coverage = share of the 8 job requirements addressed by a supported claim.</text>')
    svg.append("</svg>")
    d = out / "figures"
    d.mkdir(exist_ok=True)
    (d / "e3_tradeoff.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"wrote {d / 'e3_tradeoff.svg'}")


if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    main(a.config)
