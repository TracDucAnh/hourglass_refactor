"""Primitive vẽ dùng chung (Study 1, Study 4)."""

import matplotlib.pyplot as plt


def band(ax, x, m, lo, hi, color, label=None, ls="-", lw=2.4, alpha=0.18):
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def make_fig(nrows, ncols, pw=8.2, ph=5.4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows), squeeze=False)
    fig.subplots_adjust(wspace=0.32, hspace=1.0 if nrows > 1 else 0.3, top=0.88)
    return fig, axes
