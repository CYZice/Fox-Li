# 陈昀泽 · U202413823 · 光实2401
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator


BLUE = "#3C5488"
RED = "#E64B35"
GREEN = "#00A087"
GRAY = "#666666"
LIGHT_GRAY = "#B8B8B8"
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.figsize": (4.6, 3.4),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "lines.markeredgewidth": 1,
            "axes.formatter.use_mathtext": True,
            "axes.formatter.useoffset": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.handlelength": 2.5,
        }
    )


def finish_figure(fig, name):
    FIGURE_DIR.mkdir(exist_ok=True)
    # 减少刻度，保留相位和色条的固定刻度
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        if ax.get_xscale() == "linear" and not isinstance(
            ax.xaxis.get_major_locator(), FixedLocator
        ):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if ax.get_yscale() == "linear" and not isinstance(
            ax.yaxis.get_major_locator(), FixedLocator
        ):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.savefig(FIGURE_DIR / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / (name + ".pdf"), bbox_inches="tight")

    if "agg" not in plt.get_backend().lower():
        plt.show()

    plt.close(fig)


def plot_base_amplitude(x, amplitude, gaussian, ax):
    ax.plot(x, amplitude, color=BLUE, label="Numerical")
    ax.plot(x, gaussian, "--", color=RED, label="Gaussian")
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$|U(x)|$")
    ax.set_ylim(bottom=0)
    ax.legend()


def plot_error(error_list, ax):
    iteration = np.arange(1, len(error_list) + 1)
    ax.plot(iteration, error_list, color=BLUE)
    ax.axhline(
        1e-4, color=GRAY, linestyle="--", linewidth=1.2, label=r"$10^{-4}$ threshold"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Residual $\epsilon_n$")
    ax.legend()


def plot_base_compare(x, amplitude_1d, amplitude_2d, gaussian, ax):
    ax.plot(x, amplitude_1d, color=BLUE, label="1D Fox-Li")
    ax.plot(
        x,
        amplitude_2d,
        "o",
        color=GREEN,
        markerfacecolor="none",
        markevery=4,
        zorder=3,
        label="2D center line",
    )
    ax.plot(x, gaussian, "--", color=RED, label="Gaussian")
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$|U|$")
    ax.set_ylim(bottom=0)
    ax.legend()


def plot_tilt_profile(y, amplitude_zero, amplitude_tilt, theta, ax):
    ax.plot(y, amplitude_zero, "--", color=GRAY, label=r"$\theta=0$")
    ax.plot(y, amplitude_tilt, color=BLUE, label=rf"$\theta={theta:.3f}$ rad")
    ax.set_xlabel(r"$y / \lambda$")
    ax.set_ylabel(r"$|U(y)|$")
    ax.set_ylim(bottom=0)
    ax.legend()


def plot_tilt_center(theta, center, distance, ax):
    ax.plot(
        theta,
        center,
        "o",
        color=BLUE,
        markerfacecolor="none",
        zorder=3,
        label="Numerical",
    )
    ax.plot(theta, -distance * theta, "--", color=RED, label=r"$-L\theta$")
    ax.axhline(0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    ax.axvline(0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    ax.set_xlabel(r"Mirror tilt $\theta$ (rad)")
    ax.set_ylabel(r"Beam center $y_c / \lambda$")
    ax.legend()


def plot_tilt_width(theta, width, ax):
    ax.plot(theta, width, "o-", color=BLUE, markerfacecolor="white")
    ax.set_xlabel(r"Mirror tilt $\theta$ (rad)")
    ax.set_ylabel(r"Mode radius $w_y / \lambda$")


def plot_tilt_loss(theta, loss, ax):
    ax.plot(theta, np.asarray(loss) * 100, "o-", color=RED, markerfacecolor="white")
    ax.set_xlabel(r"Mirror tilt $\theta$ (rad)")
    ax.set_ylabel("Round-trip diffraction loss (%)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylim(bottom=0)


def plot_tilt_2d(x, fields, theta_list, name="tilt_2d"):
    fig, axes = plt.subplots(
        1,
        len(theta_list),
        figsize=(9.2, 3.1),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    extent = [x[0], x[-1], x[0], x[-1]]

    for index, (ax, u, theta) in enumerate(zip(axes, fields, theta_list)):
        image = ax.imshow(
            np.abs(u),
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(rf"$\theta={theta:.3f}$ rad")
        ax.set_xlabel(r"$x / \lambda$")
        ax.set_xticks(np.linspace(x[0], x[-1], 5))
        ax.set_yticks(np.linspace(x[0], x[-1], 5))
        if index == 0:
            ax.set_ylabel(r"$y / \lambda$")

    cbar = fig.colorbar(
        image, ax=axes.tolist(), pad=0.02, fraction=0.025, ticks=np.linspace(0, 1, 6)
    )
    cbar.set_label(r"$|U(x,y)|$")
    finish_figure(fig, name)


def plot_report_base(x, u, gaussian, x2, amplitude_2d):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    amplitude = np.abs(u)
    plot_base_amplitude(x, amplitude, gaussian, ax=axes[0])
    # 只选择中心相位为参考，不改变计算场
    center = np.argmin(np.abs(x))
    phase = np.angle(u * np.exp(-1j * np.angle(u[center])))
    axes[1].plot(x, phase, color=BLUE)
    axes[1].set_xlabel(r"$x / \lambda$")
    axes[1].set_ylabel(r"$\phi(x)-\phi(0)$ (rad)")
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_yticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    plot_base_compare(x2, amplitude, amplitude_2d, gaussian, ax=axes[2])
    axes[0].set_title("(a) 1D amplitude", loc="left")
    axes[1].set_title("(b) Relative phase", loc="left")
    axes[2].set_title("(c) 1D/2D comparison", loc="left")
    finish_figure(fig, "report_base_validation")


def plot_report_convergence(gamma_list, error_list):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
    iteration = np.arange(1, len(gamma_list) + 1)
    axes[0].plot(iteration, np.abs(gamma_list), color=BLUE)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$|\gamma_n|$")
    plot_error(error_list, ax=axes[1])
    axes[0].set_title("(a) Eigenvalue magnitude", loc="left")
    axes[1].set_title("(b) Iteration residual", loc="left")
    finish_figure(fig, "report_convergence")


def plot_report_base_2d(x, u2d):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="compressed")
    extent = [x[0], x[-1], x[0], x[-1]]
    phase_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]

    for index, ax in enumerate(axes):
        if index == 0:
            data = np.abs(u2d)
            cmap, low, high = "magma", 0, 1
            ticks = np.linspace(0, 1, 6)
            label = r"$|U(x,y)|$"
            title = "(a) Amplitude"
        else:
            # 用振幅峰值处作相位参考，显示整个区域
            amplitude = np.abs(u2d)
            data = np.angle(u2d * np.exp(-1j * np.angle(u2d.flat[np.argmax(amplitude)])))
            cmap = "twilight"
            low, high = -np.pi, np.pi
            ticks = phase_ticks
            label = "Relative phase (rad)"
            title = "(b) Phase"

        image = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=low,
            vmax=high,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xlabel(r"$x / \lambda$")
        ax.set_ylabel(r"$y / \lambda$")
        ax.set_xticks(np.linspace(x[0], x[-1], 5))
        ax.set_yticks(np.linspace(x[0], x[-1], 5))
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(image, ax=ax, pad=0.03, fraction=0.046, ticks=ticks)
        cbar.set_label(label)
        if index == 1:
            cbar.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    finish_figure(fig, "report_base_2d")


def plot_report_tilt(
    y,
    amplitude_zero,
    amplitude_tilt,
    theta,
    theta_list,
    center_list,
    distance,
    theta_change,
    width_change,
    loss_change,
):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7))
    plot_tilt_profile(y, amplitude_zero, amplitude_tilt, theta, ax=axes[0, 0])
    plot_tilt_center(theta_list, center_list, distance, ax=axes[0, 1])
    plot_tilt_width(theta_change, width_change, ax=axes[1, 0])
    plot_tilt_loss(theta_change, loss_change, ax=axes[1, 1])
    titles = [
        "(a) Mode profiles",
        "(b) Beam displacement",
        "(c) Mode radius",
        "(d) Diffraction loss",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left")
    finish_figure(fig, "report_tilt_quantitative")


def plot_bump_fields(x, results, titles, name="bump_mode"):
    # 凸起图保留低振幅相位掩膜，颜色和坐标范围保持一致
    fig, axes = plt.subplots(3, len(results), figsize=(10.8, 9),
                             sharex=True, sharey=True, layout="compressed")
    extent = [x[0], x[-1], x[0], x[-1]]
    phase_map = plt.get_cmap("twilight").copy()
    phase_map.set_bad("white")
    height_max = max(np.max(result["height"]) for result in results)
    for col, (result, title) in enumerate(zip(results, titles)):
        if not result["converged"]:
            raise ValueError("不能把未收敛场绘制为稳态模式")
        u = result["field"]
        amplitude = np.abs(u)
        # 用振幅峰值处作相位参考，避免离轴模式中心接近零
        phase = np.angle(u * np.exp(-1j * np.angle(u.flat[np.argmax(amplitude)])))
        phase = np.ma.masked_where(amplitude < 0.1 * amplitude.max(), phase)
        data = [amplitude, phase, result["height"]]
        for row, (values, cmap, low, high) in enumerate(zip(
                data, ["magma", phase_map, "viridis"],
                [0, -np.pi, 0], [1, np.pi, height_max])):
            ax = axes[row, col]
            image = ax.imshow(values, extent=extent, origin="lower", cmap=cmap,
                              vmin=low, vmax=high, interpolation="nearest", aspect="equal")
            ax.set_xticks(np.linspace(x[0], x[-1], 5))
            ax.set_yticks(np.linspace(x[0], x[-1], 5))
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(r"$y / \lambda$")
            if row == 2:
                ax.set_xlabel(r"$x / \lambda$")
            if col == len(results) - 1:
                ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi] if row == 1 else None
                cbar = fig.colorbar(image, ax=axes[row].tolist(), pad=0.02,
                                    fraction=0.025, ticks=ticks)
                cbar.set_label([r"$|U(x,y)|$", r"Relative phase (rad)", r"$h / \lambda$"][row])
                if row == 1:
                    cbar.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    finish_figure(fig, name)
