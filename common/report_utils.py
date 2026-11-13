"""Shared helpers for the report-asset scripts: LaTeX-safe tables (+ CSV/Markdown) and figure saving."""
from __future__ import annotations

import csv
import re
from pathlib import Path

TEXT_WIDTH_IN = 5.5  # NeurIPS text width

# Unicode notation in headers/captions -> LaTeX (applied via placeholders before escaping)
TEX_TOKENS = [("‖θ−θ0‖₂", r"$\|\theta-\theta_0\|_2$"), ("‖ΔW‖F", r"$\|\Delta W\|_F$"),
              ("‖Uk^T B~‖", r"$\|U_k^\top \tilde B\|$"), ("‖A~ Vk‖", r"$\|\tilde A V_k\|$"),
              ("λ = 10⁻³", r"$\lambda = 10^{-3}$"), ("λΩ", r"$\lambda\Omega$"), ("ρk", r"$\rho_k$"), ("η", r"$\eta$"), ("ΔNLL", r"$\Delta$NLL"), ("Δ", r"$\Delta$"), ("±", r"$\pm$"),
              ("→", r"$\to$"), ("≥", r"$\geq$"), ("×", r"$\times$"), ("λ", r"$\lambda$"), ("θ0", r"$\theta_0$"),
              ("†", r"$^\dagger$"), ("−", r"$-$"), ("≤", r"$\leq$"), ("γ", r"$\gamma$"), ("τ", r"$\tau$"),
              ("·", r"$\cdot$")]


def tex_escape(s) -> str:
    s = str(s)
    holders = {}
    for i, (tok, tex) in enumerate(TEX_TOKENS):
        if tok in s:
            key = f"\x00{i}\x00"
            s = s.replace(tok, key)
            holders[key] = tex
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    for key, tex in holders.items():
        s = s.replace(key, tex)
    return s


def header_cell(h) -> str:
    """Header text; a "\\n" in the label becomes a two-line \\makecell header."""
    parts = str(h).split("\n")
    if len(parts) == 1:
        return tex_escape(parts[0])
    # "\\" followed by "[" would be read as \\[<length>] and hang LaTeX; "{}" stops that
    return r"\makecell{" + r"\\{}".join(tex_escape(x) for x in parts) + "}"


def existing_caption(path: Path) -> str | None:
    """The \\caption{...} already in a table file (e.g. one you rewrote by hand), so regenerating keeps it."""
    if not path.exists():
        return None
    t = path.read_text(encoding="utf-8")
    i = t.find(r"\caption{")
    if i < 0:
        return None
    k, d = i + len(r"\caption{"), 1
    while k < len(t) and d:
        d += {"{": 1, "}": -1}.get(t[k], 0)
        k += 1
    return t[i + len(r"\caption{"):k - 1]


def write_table(path: Path, header: list[str], rows: list[list], caption: str, label: str, align: str,
                midrules_after: list[int] | None = None, keep_caption: bool = True) -> None:
    """booktabs table; header cells may contain "\\n" (two-line header). An existing caption is kept."""
    cap = existing_caption(path) if keep_caption else None
    align = re.sub(r"(?<!\\arraybackslash\})p\{", r">{\\raggedright\\arraybackslash}p{", align)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + (cap if cap is not None else tex_escape(caption)) + "}",
           rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(header_cell(h) for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        tex.append(" & ".join(tex_escape(c) for c in r) + r" \\")
        if midrules_after and i in midrules_after and i != len(rows) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def save_table(folder: Path, name: str, header: list[str], rows: list[list], caption: str, label: str,
               align: str | None = None, midrules_after: list[int] | None = None) -> None:
    """Write <name>.csv, <name>.md and an \\input-ready booktabs <name>.tex (existing caption kept)."""
    folder.mkdir(parents=True, exist_ok=True)
    flat = [str(h).replace("\n", " ") for h in header]
    with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(flat)
        w.writerows(rows)
    md = ["| " + " | ".join(flat) + " |", "|" + "---|" * len(flat)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (folder / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    write_table(folder / f"{name}.tex", header, rows, caption, label, align or "l" + "r" * (len(header) - 1),
                midrules_after)


def save_fig(fig, folder: Path, name: str) -> None:
    """Save at the size the figure is drawn: include it in LaTeX at scale 1 (\\includegraphics{...})."""
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def plain_log_ticks(ax, subs=(1.0, 2.0, 5.0)) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def ema(x, span: int = 50):
    import numpy as np

    a, out, m = 2 / (span + 1), [], None
    for v in x:
        m = v if m is None else a * v + (1 - a) * m
        out.append(m)
    return np.array(out)


# Figures are drawn at their printed size (include with scale 1); every text element is >= 7 pt in print.
FONT_PT = {"base": 8, "tick": 7, "legend": 7, "title": 8, "small": 7}
SERIF = ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes", "STIXGeneral",
         "DejaVu Serif"]


def rcparams() -> None:
    """Times-style serif text matching the NeurIPS body font, STIX math, readable sizes."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "font.size": FONT_PT["base"], "axes.titlesize": FONT_PT["title"], "axes.labelsize": FONT_PT["base"],
        "xtick.labelsize": FONT_PT["tick"], "ytick.labelsize": FONT_PT["tick"], "legend.fontsize": FONT_PT["legend"],
        "axes.titlepad": 3, "axes.labelpad": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "lines.markersize": 3.5, "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 100})
