"""从保存的计算结果整理报告图件和比较数据。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base import make_kernel, calc_stats
from bump import make_bump, overlap
from tilt import calc_tilt
from plot_results import (
    set_plot_style, finish_figure, BLUE, RED,
    plot_base_amplitude, plot_base_compare, plot_error,
    plot_tilt_profile, plot_tilt_center, plot_tilt_width, plot_tilt_loss,
    plot_tilt_2d,
)


DATA = Path(__file__).resolve().parent.parent / "figures" / "data"


def save_single(name, function, *args):
    fig, ax = plt.subplots()
    function(*args, ax=ax)
    finish_figure(fig, name)


def make_single_figures():
    base = np.load(DATA / "base_1d.npz")
    base2d = np.load(DATA / "base_2d.npz")
    x, u, gaussian = base["x"], base["field"], base["gaussian"]
    save_single("base_1d_amplitude", plot_base_amplitude, x, abs(u), gaussian)
    save_single("base_1d_2d_compare", plot_base_compare, x, abs(u),
                abs(base2d["field"][len(x) // 2]), gaussian)
    save_single("base_error", plot_error, base["residual_history"])
    phase = np.angle(u * np.exp(-1j * np.angle(u[len(x) // 2])))
    for name, values, xlabel, ylabel in [
            ("base_1d_phase", phase, r"$x / \lambda$", "Relative phase (rad)"),
            ("base_gamma_abs", abs(base["gamma_history"]), "Iteration", r"$|\gamma_n|$"),
            ("base_gamma_phase", np.angle(base["gamma_history"]), "Iteration", r"$\arg\gamma_n$ (rad)")]:
        fig, ax = plt.subplots()
        axis = x if name == "base_1d_phase" else np.arange(1, len(values) + 1)
        ax.plot(axis, values, color=BLUE)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        finish_figure(fig, name)
    field = base2d["field"]
    amplitude = abs(field)
    phase = np.angle(field * np.exp(-1j * np.angle(field.flat[np.argmax(amplitude)])))
    for name, values, cmap, low, high, label in [
            ("base_2d_amplitude", amplitude, "magma", 0, 1, r"$|U(x,y)|$"),
            ("base_2d_phase", phase,
             "twilight", -np.pi, np.pi, "Relative phase (rad)")]:
        fig, ax = plt.subplots(layout="compressed")
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad("white")
        image = ax.imshow(values, extent=[x[0], x[-1], x[0], x[-1]],
                          origin="lower", cmap=cmap, vmin=low, vmax=high,
                          interpolation="nearest")
        ax.set(xlabel=r"$x / \lambda$", ylabel=r"$y / \lambda$")
        fig.colorbar(image, ax=ax).set_label(label)
        finish_figure(fig, name)

    scan = np.genfromtxt(DATA / "tilt_scan.csv", delimiter=",", names=True)
    change = np.genfromtxt(DATA / "tilt_change.csv", delimiter=",", names=True)
    normal = np.load(DATA / "tilt_2d_0.000.npz")
    tilted = np.load(DATA / "tilt_2d_0.005.npz")
    # 单图直接取保存的二维中心截线，和比较表对应同一个场
    save_single("tilt_profile", plot_tilt_profile, x,
                abs(normal["field"][:, len(x) // 2]),
                abs(tilted["field"][:, len(x) // 2]), 0.005)
    save_single("tilt_center", plot_tilt_center, scan["theta_rad"], scan["y_center_lambda"], 100)
    save_single("tilt_width", plot_tilt_width, change["theta_rad"], change["w_y_lambda"])
    save_single("tilt_loss", plot_tilt_loss, change["theta_rad"], change["round_trip_loss_2d"])
    fields = [np.load(DATA / f"tilt_2d_{angle:.3f}.npz")["field"] for angle in [0, 0.025, 0.05]]
    plot_tilt_2d(x, fields, [0, 0.025, 0.05])


def make_comparison():
    reference = np.load(DATA / "tilt_2d_0.000.npz")
    x, normal = reference["x"], reference["field"]
    kernel = make_kernel(x)
    base_loss = calc_stats(x, normal, kernel)["loss"]
    rows = []
    cases = [("normal", "tilt_2d_0.000", "uniform", "baseline")]
    for angle in [0.005, 0.025, 0.05]:
        cases.append(("tilt", f"tilt_2d_{angle:.3f}", "uniform", "main"))
    for index in [1, 2, 4]:
        cases.append(("bump", f"bump_radius_{index}", "uniform", "main"))
    cases.append(("bump", "bump_position_5", "uniform", "main"))
    cases.append(("bump", "bump_seed_asymmetric", "asymmetric", "appendix"))
    for kind, name, seed, use in cases:
        data = np.load(DATA / f"{name}.npz")
        assert data["converged"] and np.array_equal(x, data["x"])
        theta = float(data["theta"]) if kind == "tilt" else 0
        x0, y0, rb = [float(data[key]) if kind == "bump" else 0
                      for key in ["x0", "y0", "radius"]]
        phase = make_bump(x, x0, y0, rb)[1] if kind == "bump" else np.exp(1j * 4 * np.pi * theta * x)[:, None]
        stats = calc_stats(x, data["field"], kernel, phase)
        assert stats["residual"] < 1e-4
        assert abs(stats["loss"] - stats["power_loss"]) < 1e-8
        rows.append([kind, theta, x0, y0, rb, seed, 101, 1,
                     len(data["residual_history"]), stats["residual"],
                     stats["xc"], stats["yc"], stats["wx"], stats["wy"],
                     abs(stats["mu"]), stats["loss"], stats["loss"] - base_loss,
                     overlap(normal, data["field"]), use, f"{name}.npz"])
    header = "kind,theta_rad,x0_lambda,y0_lambda,rb_lambda,seed,N,converged,iteration,residual,xc_lambda,yc_lambda,wx_lambda,wy_lambda,mu_abs,round_trip_loss,extra_loss,mode_overlap,report_use,source"
    np.savetxt(DATA / "mirror_comparison.csv", rows, fmt="%s", delimiter=",", comments="", header=header)
    print("Comparison rows:", len(rows))
    for row in rows:
        print(row[0], row[1:6], "loss (%)", float(row[15]) * 100,
              "overlap", row[17], flush=True)


def select_bump_figures():
    # 正文采用精简图；原完整相位/形貌图保留作附录材料
    files = ["bump_radius_0", "bump_radius_2", "bump_position_5"]
    titles = ["No bump", r"$r_b=2\lambda$, $P=(0,0)$",
              r"$r_b=2\lambda$, $P=(6\lambda,0)$"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1),
                             sharex=True, sharey=True, layout="compressed")
    for ax, name, title in zip(axes, files, titles):
        data = np.load(DATA / f"{name}.npz")
        assert data["converged"]
        x = data["x"]
        image = ax.imshow(abs(data["field"]), extent=[x[0], x[-1], x[0], x[-1]],
                          origin="lower", cmap="magma", vmin=0, vmax=1,
                          interpolation="nearest")
        ax.set(title=title, xlabel=r"$x / \lambda$")
        ax.set_xticks(np.linspace(-10, 10, 5))
        ax.set_yticks(np.linspace(-10, 10, 5))
    axes[0].set_ylabel(r"$y / \lambda$")
    fig.colorbar(image, ax=axes.tolist(), pad=0.02, fraction=0.025).set_label(r"$|U(x,y)|$")
    finish_figure(fig, "report_bump_2d")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7))
    for col, name in enumerate(["radius", "position"]):
        data = np.genfromtxt(DATA / f"bump_{name}.csv", delimiter=",", names=True)
        axis = data["R_lambda"] if name == "radius" else data["x0_lambda"]
        xlabel = r"Bump radius $r_b / \lambda$" if name == "radius" else r"Position $x_0 / \lambda$ ($r_b=2\lambda$, $y_0=0$)"
        axes[0, col].plot(axis, data["round_trip_loss"] * 100, "o-", color=RED, markerfacecolor="white")
        axes[0, col].set(ylabel="Round-trip diffraction loss (%)", ylim=(0, None))
        axes[1, col].plot(axis, data["mode_overlap"], "o-", color=BLUE, markerfacecolor="white")
        axes[1, col].set(ylabel="Overlap with unperturbed mode", ylim=(0, 1.05))
        for ax in axes[:, col]:
            ax.set_xlabel(xlabel)
            if name == "position":
                ax.axvline(3, color="#666666", linestyle=":", linewidth=0.8)
        if name == "position":
            axes[0, col].text(0.98, 0.95, r"$x_0=3\lambda$: not converged",
                              ha="right", va="top", transform=axes[0, col].transAxes, fontsize=8)
    for ax, title in zip(axes.flat, ["(a) Radius: loss", "(b) Position: loss", "(c) Radius: overlap", "(d) Position: overlap"]):
        ax.set_title(title, loc="left")
    finish_figure(fig, "report_bump_quantitative")


def check_separability():
    data = np.load(DATA / "tilt_2d_0.000.npz")
    x = data["x"]
    kernel = make_kernel(x)
    u0, _, _, _, converged = calc_tilt(0, x, kernel)
    assert converged
    rows = []
    for theta in [0, 0.005, 0.025, 0.05]:
        full = np.load(DATA / f"tilt_2d_{theta:.3f}.npz")
        uy, _, _, _, converged = calc_tilt(theta, x, kernel)
        field = np.outer(uy, u0)
        stats = calc_stats(x, field, kernel, np.exp(1j * 4 * np.pi * theta * x)[:, None])
        fidelity = overlap(field, full["field"])
        assert converged and stats["residual"] < 1e-4 and fidelity > 0.9999
        rows.append([theta, stats["residual"], float(full["residual"]),
                     abs(stats["loss"] - full["loss"]), fidelity,
                     abs(stats["wx"] - full["wx"]), abs(stats["wy"] - full["wy"])])
    np.savetxt(DATA / "tilt_separability_check.csv", rows, delimiter=",", comments="",
               header="theta_rad,separable_residual,full_residual,loss_abs_diff,mode_overlap,wx_abs_diff_lambda,wy_abs_diff_lambda")


if __name__ == "__main__":
    set_plot_style()
    make_single_figures()
    make_comparison()
    select_bump_figures()
    check_separability()
