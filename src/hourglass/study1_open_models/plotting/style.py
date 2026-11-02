# Quy ước hình (Cell D2)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 14 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D2 — Quy ước hình: PDF, Times New Roman, không title lớn (tên file là định danh), chỉ có sub-caption (a), (b), ...
# ============================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

LEGEND_FS = 16     # legend font size
SUBFIG_FS = 28     # (a), (b), (c) font size
LABEL_FS  = 24     # axis labels
TICK_FS   = 16     # tick labels
SUBFIG_WITH_DESC = True     # True: "(a) $G_k$"; False: chỉ "(a)"

def setup_fonts():
    avail = {f.name for f in fm.fontManager.ttflist}
    for cand in ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"]:
        if cand in avail:
            chosen = cand; break
    else:
        chosen = "DejaVu Serif"
    mpl.rcParams.update({"font.family": "serif", "font.serif": [chosen], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    if chosen != "Times New Roman":
        print(f"[font] Không có Times New Roman -> dùng '{chosen}' (cài fonts-liberation hoặc msttcorefonts để đúng font).")
    return chosen
setup_fonts()

X_RHO = r"relative depth $\rho=k/K$"
PANELS = {"G": (r"alignment $\bar{G}_k$", r"$\bar{G}_k$"), "D": (r"separation $D_k$", r"$D_k$"),
          "CKA": ("linear CKA", "CKA"), "A": (r"Procrustes $A_k$", r"$A_k$"),
          "R": ("MRR (CSLS)", r"$R_k$"), "U": (r"$U_k/\log|L|$", r"$U_k$")}
RANDOM_COLOR = "#555555"

from hourglass.common.plotting import make_fig

def style_ax(ax, xlabel, ylabel, cap, desc=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    txt = f"{cap} {desc}" if (SUBFIG_WITH_DESC and desc) else cap
    ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha="center", va="top", fontsize=SUBFIG_FS)

def add_legend(fig, handles, labels, ncol=3):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=ncol,
               fontsize=LEGEND_FS, frameon=False, handlelength=2.2, columnspacing=1.6)

def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)
    plt.show(); plt.close(fig)

from hourglass.common.plotting import band
