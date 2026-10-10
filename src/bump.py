# 陈昀泽 · U202413823 · 光实2401
"""镜面局部半球凸起的二维 Fox-Li 仿真。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base import make_kernel, propagate2d, calc_stats

from plot_results import (
    set_plot_style, finish_figure, BLUE, RED, GREEN, GRAY,
    plot_bump_fields,
)


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
DATA = FIGURES / "data"


def make_bump(x, x0, y0, radius):
    X, Y = np.meshgrid(x, x)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    # 半球朝腔内凸起；radius=0 用于无扰动对照
    h = np.sqrt(np.maximum(radius ** 2 - r2, 0))
    return h, np.exp(1j * 4 * np.pi * h)


def calc_bump(x, kernel, x0=0, y0=0, radius=2, iteration=6000, field0=None):
    h, phase = make_bump(x, x0, y0, radius)
    u = np.ones((len(x), len(x)), dtype=complex)
    if field0 is not None:
        u = np.array(field0, dtype=complex, copy=True)
    errors = []
    mu_list = []
    stable = 0
    mu = 0j

    for n in range(iteration):
        # 正常镜 -> 凸起镜 -> 正常镜，完整往返
        u2 = propagate2d(u, kernel)
        u2 *= phase
        u2 = propagate2d(u2, kernel)
        mu = np.vdot(u, u2) / np.vdot(u, u)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        errors.append(error)
        mu_list.append(mu)
        u = u2 / np.max(np.abs(u2))
        stable = stable + 1 if error < 1e-4 else 0
        if stable >= 10:
            break

    # 对最终保存的场重新计算，不把上一轮的本征值配给新场
    u2 = propagate2d(propagate2d(u, kernel) * phase, kernel)
    stats = calc_stats(x, u, kernel, phase)
    # 两次往返的残差仅用于诊断，不替代单次往返收敛判据
    u4 = propagate2d(propagate2d(u2, kernel) * phase, kernel)
    mu2 = np.vdot(u, u4) / np.vdot(u, u)
    residual2 = np.linalg.norm(u4 - mu2 * u) / np.linalg.norm(u4)

    return {
        "field": u, "height": h, "errors": np.array(errors),
        "mu_list": np.array(mu_list), "two_round_residual": residual2,
        "converged": stable >= 10 and stats["residual"] < 1e-4,
        "iteration": len(errors), **stats,
    }


def save_result(name, x, result, x0, y0, radius):
    # 保存复场和迭代历史，后续作图不必重新计算
    np.savez_compressed(DATA / f"bump_{name}.npz", x=x,
                        x0=x0, y0=y0, radius=radius,
                        field=result["field"], height=result["height"],
                        mu=result["mu"], mu_history=result["mu_list"],
                        residual_history=result["errors"],
                        converged=result["converged"],
                        residual=result["residual"],
                        two_round_residual=result["two_round_residual"],
                        power_loss=result["power_loss"])


def overlap(u, v):
    return abs(np.vdot(u, v)) ** 2 / (np.vdot(u, u).real * np.vdot(v, v).real)


def scan_results(x, kernel, reference):
    # 分别只改变半径和位置，其余参数保持固定
    scans = [("radius", [(0, 0, r) for r in [0, 1, 2, 3, 4]]),
             ("position", [(p, 0, 2) for p in [0, 1, 2, 3, 4, 6]])]
    results = {}
    for name, points in scans:
        rows = []
        for x0, y0, radius in points:
            result = calc_bump(x, kernel, x0, y0, radius)
            results[(x0, y0, radius)] = result
            save_result(f"{name}_{len(rows)}", x, result, x0, y0, radius)
            fidelity = overlap(reference["field"], result["field"])
            # 未收敛时仅保留迭代与残差，不将暂态场作为物理结果
            if not result["converged"]:
                fidelity = np.nan
            valid = result["converged"] and reference["converged"]
            values = [result[key] if valid else np.nan
                      for key in ("xc", "yc", "wx", "wy")]
            loss = result["loss"] if valid else np.nan
            rows.append([x0, y0, radius, int(result["converged"]),
                         result["iteration"], result["residual"], *values,
                         abs(result["mu"]) if valid else np.nan, loss,
                         loss - reference["loss"], fidelity if valid else np.nan])
            print(name, (x0, y0, radius), "converged:", result["converged"],
                  "iterations:", result["iteration"], "error:", result["errors"][-1],
                  "loss:", loss, "overlap:", fidelity, flush=True)

        rows = np.array(rows)
        np.savetxt(DATA / f"bump_{name}.csv", rows, delimiter=",",
                   comments="", header="x0_lambda,y0_lambda,R_lambda,converged,iteration,residual,xc_lambda,yc_lambda,wx_lambda,wy_lambda,mu_abs,round_trip_loss,extra_loss,mode_overlap")
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7))
        axes = axes.ravel()
        axis = rows[:, 2] if name == "radius" else rows[:, 0]
        xlabel = r"Bump radius $r_b / \lambda$" if name == "radius" else r"Bump position $x_0 / \lambda$ ($y_0=0$)"
        axes[0].plot(axis, rows[:, 11] * 100, "o-", color=RED, markerfacecolor="white", label="Total")
        axes[0].plot(axis, rows[:, 12] * 100, "s--", color=GRAY, markerfacecolor="white", label="Extra")
        axes[0].set(ylabel="Round-trip diffraction loss (%)", ylim=(0, None))
        axes[0].legend()
        axes[1].plot(axis, rows[:, 13], "o-", color=BLUE, markerfacecolor="white")
        axes[1].set(ylabel="Overlap with unperturbed mode", ylim=(0, 1.05))
        axes[2].plot(axis, rows[:, 8], "o-", color=BLUE, markerfacecolor="white", label=r"$w_x$")
        axes[2].plot(axis, rows[:, 9], "s--", color=GREEN, markerfacecolor="white", label=r"$w_y$")
        axes[2].set(ylabel=r"Second-moment radius $w / \lambda$")
        axes[2].legend()
        axes[3].semilogy(axis, rows[:, 5], "o-", color=BLUE, markerfacecolor="white")
        axes[3].axhline(1e-4, color=GRAY, linestyle="--")
        for p, residual in zip(axis[rows[:, 3] == 0], rows[rows[:, 3] == 0, 5]):
            for ax in axes:
                ax.axvline(p, color=GRAY, linestyle=":", linewidth=0.8)
            axes[3].annotate("Not converged", (p, residual),
                             xytext=(5, -15), textcoords="offset points", fontsize=8)
        axes[3].set(ylabel="Final self-reproduction residual")
        for ax, title in zip(axes, ["(a) Diffraction loss", "(b) Mode overlap", "(c) Mode radius", "(d) Convergence status"]):
            ax.set(xlabel=xlabel, title=title)
        finish_figure(fig, f"bump_{name}")
    return results


def main():
    set_plot_style()
    DATA.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)
    normal = calc_bump(x, kernel, radius=0)
    assert normal["converged"]
    assert abs(normal["loss"] - normal["power_loss"]) < 1e-8
    print("normal:", normal["converged"], normal["iteration"], normal["loss"])
    results = scan_results(x, kernel, normal)
    selected = [results[p] for p in [(0, 0, 0), (0, 0, 2), (6, 0, 2)]]
    plot_bump_fields(x, selected, ["No bump", r"$r_b=2\lambda$, $P=(0,0)$", r"$r_b=2\lambda$, $P=(6\lambda,0)$"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for point, color in zip([(0, 0, 0), (0, 0, 2), (6, 0, 2), (3, 0, 2)], [GRAY, BLUE, GREEN, RED]):
        result = results[point]
        label = f"r={point[2]}, x0={point[0]}"
        steps = np.arange(1, result["iteration"] + 1)
        axes[0].semilogy(steps, result["errors"], color=color, label=label)
        axes[1].semilogy(steps, np.abs(1 / result["mu_list"]), color=color, label=label)
    axes[0].axhline(1e-4, color=GRAY, linestyle="--")
    axes[0].set(xlabel="Iteration", ylabel="Self-reproduction residual", title="(a) Field residual")
    axes[1].set(xlabel="Iteration", ylabel=r"$|\gamma_n|$", title="(b) Round-trip eigenvalue estimate")
    for ax in axes:
        ax.legend()
    finish_figure(fig, "bump_convergence")

    # 用同一物理扰动检查网格加密时损耗是否稳定
    checks = []
    for n in (51, 81, 101, 141, 201, 301):
        xcheck = np.linspace(-10, 10, n)
        kcheck = make_kernel(xcheck)
        base = calc_bump(xcheck, kcheck, radius=0)
        defect = calc_bump(xcheck, kcheck, radius=2)
        checks.append([n, base["loss"], defect["loss"],
                       defect["loss"] - base["loss"],
                       int(base["converged"]), int(defect["converged"]),
                       base["residual"], defect["residual"]])
        print("grid:", checks[-1], flush=True)
    np.savetxt(DATA / "bump_grid_check.csv", checks, delimiter=",", comments="",
               header="N,base_loss,bump_R2_loss,extra_loss,base_converged,bump_converged,base_residual,bump_residual")
    checks = np.array(checks)
    fig, ax = plt.subplots()
    valid = checks[:, 5] == 1
    ax.plot(checks[valid, 0], checks[valid, 2] * 100, "o-", color=RED, markerfacecolor="white", label="Total loss")
    ax.plot(checks[valid, 0], checks[valid, 3] * 100, "s--", color=BLUE, markerfacecolor="white", label="Extra loss")
    ax.set(xlabel="Grid points per axis N", ylabel="Round-trip diffraction loss (%)")
    ax.legend()
    finish_figure(fig, "bump_grid_check")

    # 保持算法和判据不变，只检查初始场与迭代上限的影响
    X, Y = np.meshgrid(x, x)
    gaussian = np.exp(-(X ** 2 + Y ** 2) / (100 / np.pi))
    seeds = [("uniform", None), ("gaussian", gaussian),
             ("asymmetric", gaussian * (1 + 0.05 * X + 0.03 * Y))]
    rows = []
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for (name, seed), color in zip(seeds, [BLUE, RED, GREEN]):
        result = calc_bump(x, kernel, x0=3, iteration=10000, field0=seed)
        save_result(f"seed_{name}", x, result, 3, 0, 2)
        rows.append([name, int(result["converged"]), result["iteration"],
                     result["residual"], result["two_round_residual"],
                     result["loss"] if result["converged"] else np.nan])
        ax.semilogy(np.arange(1, result["iteration"] + 1), result["errors"], color=color, label=name)
        print("seed:", rows[-1], flush=True)
    np.savetxt(DATA / "bump_seed_check.csv", rows, fmt="%s", delimiter=",", comments="",
               header="seed,converged,iteration,residual,two_round_residual,round_trip_loss")
    ax.axhline(1e-4, color=GRAY, linestyle="--")
    ax.set(xlabel="Iteration", ylabel="Self-reproduction residual")
    ax.legend()
    finish_figure(fig, "bump_seed_check")
    if result["converged"]:
        plot_bump_fields(x, [results[(1, 0, 2)], result, results[(4, 0, 2)]],
                         [r"$P=(\lambda,0)$; uniform seed",
                          r"$P=(3\lambda,0)$; asymmetric seed",
                          r"$P=(4\lambda,0)$; uniform seed"],
                         name="bump_offaxis_mode")


if __name__ == "__main__":
    main()
