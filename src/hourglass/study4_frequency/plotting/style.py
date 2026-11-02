# Quy ước hình (Cell D1)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 39 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell D1 — Quy ước hình (giống Cell D2 của Study 1): PDF, Times New Roman, không title lớn (tên file là định danh), chỉ có sub-caption (a), (b), ...
# ============================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

LEGEND_FS = 16     # legend font size
SUBFIG_FS = 28     # (a), (b), (c) font size
LABEL_FS  = 24     # axis labels
TICK_FS   = 16     # tick labels
SUBFIG_WITH_DESC = True     # True: "(a) XLM-R large"; False: chỉ "(a)"
STACK_FS  = dict(label=19, tick=14, cap=22, legend=12.5)      # hình nhiều panel (>= 4)

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

X_RANK  = r"$\log_{10}$ frequency rank $r$"
X_RHO   = r"relative depth $\rho=k/K$"
Y_R     = r"retrieval $R_k$ (MRR)"
RANDOM_COLOR = "#555555"
FUNC_COLOR, CONTENT_COLOR, OTHER_COLOR = "#d62728", "#1f77b4", "#7f7f7f"

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
    path = f"{FIG_DIR}/{name}{RUN_TAG}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)
    plt.show(); plt.close(fig)

from hourglass.common.plotting import band

def grid_axes(n, ncols=4, panel_w=5.6, panel_h=3.9, hspace=0.75, top=0.86, sharey=False):
    """Lưới panel kiểu hình stack của Study 1 (chữ nhỏ hơn), mỗi panel có sub-caption riêng bên dưới."""
    nr = int(math.ceil(n / ncols)); nc = min(n, ncols)
    fig, axes = plt.subplots(nr, nc, figsize=(panel_w * nc, panel_h * nr), squeeze=False, sharey=sharey)
    fig.subplots_adjust(wspace=0.28, hspace=hspace, top=top)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()[:n], nr

def panel_caption(ax, text, fs=STACK_FS, cap_y=-0.30):
    ax.text(0.5, cap_y, text, transform=ax.transAxes, ha="center", va="top", fontsize=fs["cap"])

def no_result(ax, fs=STACK_FS, msg="chưa có kết quả"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=fs["tick"], color="gray")

def _load_s3(key):
    p = f"{ANALYSIS_DIR}/s3_{key}{RUN_TAG}.npz"
    return np.load(p) if os.path.exists(p) else None

def _models_with(kind="s3", keys=None):
    keys = list(keys) if keys else [k for k in MODELS if os.path.exists(f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz")]
    return [k for k in keys if os.path.exists(f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz")]
