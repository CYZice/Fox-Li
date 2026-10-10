# 陈昀泽 · U202413823 · 光实2401
from pathlib import Path

import numpy as np

from plot_results import (
    plot_report_base,
    plot_report_base_2d,
    plot_report_convergence,
)


def make_kernel(x):
    dx = x[1] - x[0]
    # 波长单位1
    k = 2 * np.pi
    distance = 100
    radius = distance  # 对称共焦腔

    c = np.sqrt(1j / distance)
    kernel = np.zeros((len(x), len(x)), dtype=complex)

    for j in range(len(x)):
        for i in range(len(x)):
            # 两镜面之间的近轴光程
            rho = (
                distance
                + (x[j] - x[i]) ** 2 / (2 * distance)
                - (x[j] ** 2 + x[i] ** 2) / (2 * radius)
            )
            kernel[j, i] = c * np.exp(-1j * k * rho) * dx

    return kernel


def propagate(u, kernel):
    u2 = kernel @ u

    return u2


def propagate2d(u, kernel):
    return kernel @ u @ kernel.T


def calc_stats(x, u, kernel, phase=1):
    # 最终场的完整往返比例、残差和功率损耗
    u2 = propagate2d(propagate2d(u, kernel) * phase, kernel)
    mu = np.vdot(u, u2) / np.vdot(u, u)
    residual = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
    intensity = np.abs(u) ** 2
    weight = np.sum(intensity)
    xc = np.sum(x[None, :] * intensity) / weight
    yc = np.sum(x[:, None] * intensity) / weight
    wx = 2 * np.sqrt(np.sum((x[None, :] - xc) ** 2 * intensity) / weight)
    wy = 2 * np.sqrt(np.sum((x[:, None] - yc) ** 2 * intensity) / weight)
    return {"mu": mu, "residual": residual, "loss": 1 - abs(mu) ** 2,
            "power_loss": 1 - np.vdot(u2, u2).real / np.vdot(u, u).real,
            "xc": xc, "yc": yc, "wx": wx, "wy": wy}


# 基础任务：不加扰动，计算一维、二维自再现模式并输出报告图
def run_base():
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)
    u = np.ones(101, dtype=complex)
    iteration = 1000
    gamma_list = []
    error_list = []
    stable = 0

    for n in range(iteration):
        u2 = propagate(u, kernel)
        mu = np.vdot(u, u2) / np.vdot(u, u)
        gamma_list.append(1 / mu)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        error_list.append(error)
        u = u2 / np.max(np.abs(u2))

        if error < 1e-4:
            stable += 1
        else:
            stable = 0

        if stable >= 10:
            print("收敛迭代次数:", n + 1)
            break

    amplitude = np.abs(u)
    field1 = propagate(u, kernel)
    mu1 = np.vdot(u, field1) / np.vdot(u, u)
    residual1 = np.linalg.norm(field1 - mu1 * u) / np.linalg.norm(field1)
    print("基础gamma:", np.abs(1 / mu1))
    print("基础 arg(gamma) (rad):", np.angle(1 / mu1))
    print("基础单程mu:", np.abs(mu1))
    print("基础单程功率损耗:", 1 - np.abs(mu1) ** 2)
    gaussian = np.exp(-(x**2) / (100 / np.pi))
    positive = x >= 0
    index = np.argmin(np.abs(amplitude[positive] - 1 / np.e))
    print("数值 1/e 半径:", x[positive][index])
    print("理论 1/e 半径:", np.sqrt(100 / np.pi))

    x2 = np.linspace(-10, 10, 101)
    kernel2d = make_kernel(x2)
    u2d = np.ones((len(x2), len(x2)), dtype=complex)
    errors2d = []
    stable2d = 0

    for n in range(iteration):
        field2 = propagate2d(u2d, kernel2d)
        mu2d = np.vdot(u2d, field2) / np.vdot(u2d, u2d)
        error2d = np.linalg.norm(field2 - mu2d * u2d) / np.linalg.norm(field2)
        errors2d.append(error2d)
        u2d = field2 / np.max(np.abs(field2))
        stable2d = stable2d + 1 if error2d < 1e-4 else 0
        if stable2d >= 10:
            break

    stats = calc_stats(x2, u2d, kernel2d)
    assert stable >= 10 and stable2d >= 10 and stats["residual"] < 1e-4
    print("基础二维迭代次数、往返残差、往返损耗:", len(errors2d), stats["residual"], stats["loss"])
    data = Path(__file__).resolve().parent.parent / "figures" / "data"
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data / "base_1d.npz", x=x, field=u, gaussian=gaussian,
                        gamma_history=gamma_list, residual_history=error_list,
                        mu_single=mu1, residual_single=residual1, converged=stable >= 10)
    np.savez_compressed(data / "base_2d.npz", x=x2, field=u2d,
                        residual_history=errors2d, converged=True, **stats)

    center = len(x2) // 2
    print("一维/二维中心截线 RMS:", np.sqrt(np.mean((np.abs(u) - np.abs(u2d[center])) ** 2)))

    plot_report_base(x, u, gaussian, x2, np.abs(u2d[center]))
    plot_report_convergence(gamma_list, error_list)
    plot_report_base_2d(x2, u2d)
