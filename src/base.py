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


def make_kernel2d(x):
    dx = x[1] - x[0]
    k = 2 * np.pi
    distance = 100
    kernel = np.zeros((len(x), len(x)), dtype=complex)

    for j in range(len(x)):
        for i in range(len(x)):
            kernel[j, i] = np.exp(1j * k * x[j] * x[i] / distance) * dx

    return kernel


def propagate2d(u, kernel):
    return kernel @ u @ kernel.T


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
    print("基础gamma:", np.abs(gamma_list[-1]))
    print("基础 arg(gamma) (rad):", np.angle(gamma_list[-1]))
    print("基础单程mu:", np.abs(1 / gamma_list[-1]))
    print("基础单程功率损耗:", 1 - np.abs(1 / gamma_list[-1]) ** 2)
    gaussian = np.exp(-(x**2) / (100 / np.pi))
    positive = x >= 0
    index = np.argmin(np.abs(amplitude[positive] - 1 / np.e))
    print("数值 1/e 半径:", x[positive][index])
    print("理论 1/e 半径:", np.sqrt(100 / np.pi))

    x2 = np.linspace(-10, 10, 51)
    kernel2d = make_kernel2d(x2)
    u2d = np.ones((51, 51), dtype=complex)

    for n in range(len(gamma_list)):
        u2d = propagate2d(u2d, kernel2d)
        u2d = u2d / np.max(np.abs(u2d))

    center = len(x2) // 2

    plot_report_base(x, u, gaussian, x2, np.abs(u2d[center]))
    plot_report_convergence(gamma_list, error_list)
    plot_report_base_2d(x2, u2d)
