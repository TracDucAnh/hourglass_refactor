# Style hình (Cell D4)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 38 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D4 — Style hình (giữ style Study 1: PDF, chữ serif, không tiêu đề, chú thích (a)(b) dưới trục, chú giải chung phía trên)
# ============================================================
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

def setup_fonts():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        if name in avail:
            plt.rcParams["font.family"] = name; plt.rcParams["mathtext.fontset"] = "stix"
            return name
setup_fonts()
plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110})
C = dict(LM="#1f77b4", SEM="#d62728", SURF="#2ca02c", init="#7f7f7f", o0="#1f77b4", o100="#ff7f0e",
         decoder="#1f77b4", encoder="#9467bd", moe="#2ca02c", diffusion="#d62728")
LCOL = {2: "#1f77b4", 3: "#17becf", 4: "#ff7f0e", 6: "#8c564b", 8: "#2ca02c"}
KP = K_LAYERS
RHO = np.arange(KP + 1) / KP

def _stack(rows, key, variant="trained", norm_L=False):
    out = []
    for r in rows.itertuples():
        c = cv(r.rid, variant)
        if c is not None:
            v = np.asarray(c[key], float)
            out.append(v / math.log(r.L) if norm_L else v)
    return np.array(out)

def draw(ax, rows, key, color, ls="-", label=None, variant="trained", norm_L=False, mark_peak=False, mark_min=False):
    M = _stack(rows, key, variant, norm_L)
    if M.size == 0:
        return None
    m, lo, hi = tcrit_band(M)
    ax.plot(RHO, m, color=color, ls=ls, lw=1.6, label=label)
    ax.fill_between(RHO, lo, hi, color=color, alpha=0.15, lw=0)
    if mark_peak:
        ax.plot(RHO[int(np.nanargmax(m))], np.nanmax(m), "o", color=color, ms=5, mfc="white", mew=1.4)
    if mark_min:
        ax.plot(RHO[int(np.nanargmin(m))], np.nanmin(m), "v", color=color, ms=5, mfc="white", mew=1.4)
    return m

def subcap(ax, text):
    ax.text(0.5, -0.30, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

def legend_top(fig, handles, ncol=None, y=1.02):
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), frameon=False, bbox_to_anchor=(0.5, y))

def save_fig(fig, name):
    p = f"{FIG_DIR}/{name}.pdf"
    fig.savefig(p, bbox_inches="tight"); plt.show(); plt.close(fig)
    print("đã lưu", p)

def _lines(entries):
    return [Line2D([0], [0], color=c, ls=ls, lw=1.6, label=lb) for c, ls, lb in entries]

def _xlab(ax):
    ax.set_xlabel(r"relative depth $\rho=k/K$"); ax.set_xlim(0, 1)
