"""Shared helpers for the report-asset scripts: LaTeX-safe tables (+ CSV/Markdown) and figure saving."""
from __future__ import annotations

import csv
from pathlib import Path

TEXT_WIDTH_IN = 5.5  # NeurIPS text width

# Unicode notation in headers/captions -> LaTeX (applied via placeholders before escaping)
TEX_TOKENS = [("‖θ−θ0‖₂", r"$\|\theta-\theta_0\|_2$"), ("ΔNLL", r"$\Delta$NLL"), ("Δ", r"$\Delta$"), ("±", r"$\pm$"),
              ("→", r"$\to$"), ("≥", r"$\geq$"), ("×", r"$\times$"), ("λ", r"$\lambda$"), ("θ0", r"$\theta_0$"),
              ("†", r"$^\dagger$")]


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


def save_table(folder: Path, name: str, header: list[str], rows: list[list], caption: str, label: str,
               align: str | None = None, midrules_after: list[int] | None = None) -> None:
    """Write <name>.csv, <name>.md and an \\input-ready booktabs <name>.tex."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (folder / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    align = align or "l" + "r" * (len(header) - 1)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        tex.append(" & ".join(tex_escape(c) for c in r) + r" \\")
        if midrules_after and i in midrules_after and i != len(rows) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (folder / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def save_fig(fig, folder: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", dpi=200)
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


def rcparams() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})
