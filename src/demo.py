# 陈昀泽 · U202413823 · 光实2401
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import bump
from base import make_kernel, propagate, propagate2d, calc_stats
from tilt import calc_tilt, calc_tilt2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, Rectangle


# 配色方案
COLORS = {
    'primary': '#3C5488',      # 主色（蓝）
    'accent': '#E64B35',       # 强调色（红）
    'success': '#00A087',      # 成功色（绿）
    'warning': '#F39B7F',      # 警告色（橙）
    'background': '#F5F5F5',   # 背景色（浅灰）
    'surface': '#FFFFFF',      # 卡片背景（白）
    'text': '#2C3E50',         # 主文字色（深灰）
    'text_secondary': '#7F8C8D' # 次要文字色（中灰）
}

BLUE = COLORS['primary']
RED = COLORS['accent']
GRAY = "#666666"
PHASE_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PHASE_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

# 复用正式计算函数；不调用扫描和文件输出入口
def compute_base():
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)
    u = np.ones(101, dtype=complex)
    gamma_list = []
    error_list = []
    stable = 0

    for n in range(1000):
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
            break

    amplitude = np.abs(u)
    positive = x >= 0
    index = np.argmin(np.abs(amplitude[positive] - 1 / np.e))
    x2 = x
    u2d = np.ones((len(x), len(x)), dtype=complex)
    stable2d = 0
    errors2d = []
    # 二维单独判断，不照搬一维迭代次数
    for n in range(1000):
        field = propagate2d(u2d, kernel)
        mu = np.vdot(u2d, field) / np.vdot(u2d, u2d)
        error = np.linalg.norm(field - mu * u2d) / np.linalg.norm(field)
        errors2d.append(error)
        u2d = field / np.max(np.abs(field))
        stable2d = stable2d + 1 if error < 1e-4 else 0
        if stable2d >= 10:
            break
    stats = calc_stats(x, u2d, kernel)
    return {"x": x, "u": u, "x2": x2, "u2d": u2d,
            "gamma": np.array(gamma_list), "error": np.array(error_list),
            "radius": x[positive][index], "converged": stable >= 10,
            "iteration2d": len(errors2d), "error2d": stats["residual"],
            "converged2d": stable2d >= 10 and stats["residual"] < 1e-4}


def set_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.formatter.useoffset": False,
        "axes.facecolor": "#FAFAFA",      # 浅灰背景
        "figure.facecolor": "#FFFFFF",    # 白色图底
        "axes.edgecolor": "#CCCCCC",      # 浅色边框
        "grid.color": "#E0E0E0",          # 淡网格线
        "grid.alpha": 0.5,
        "axes.grid": True,                # 显示网格
        "axes.axisbelow": True,           # 网格在数据下方
        "axes.spines.top": False,         # 去掉上边框
        "axes.spines.right": False,       # 去掉右边框
    })


def make_tilt_reference():
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)
    u_zero, _, _, _, converged = calc_tilt(0, x, kernel)
    if not converged:
        raise ValueError("无扰动参考场未收敛")
    # 计算二维基础场用于差异对比
    result_zero = calc_tilt2d(0, x, kernel)
    u_zero_2d = result_zero["field"] if result_zero["converged"] else np.outer(u_zero, u_zero)
    return {"x": x, "kernel": kernel, "x2": x, "u_zero": u_zero, "u_zero_2d": u_zero_2d}


def compute_tilt(theta, reference):
    x, kernel = reference["x"], reference["kernel"]
    u, _, error, iteration, converged = calc_tilt(theta, x, kernel)
    phase = np.exp(1j * 4 * np.pi * theta * x)[:, None]
    stats = calc_stats(x, np.outer(u, reference["u_zero"]), kernel, phase)
    result = calc_tilt2d(theta, x, kernel)
    valid = converged and stats["residual"] < 1e-4
    return {"theta": theta, "u": u, "u2d": result["field"],
            "center": stats["yc"] if valid else np.nan,
            "width": stats["wy"] if valid else np.nan,
            "loss": result["loss"] if result["converged"] else np.nan,
            "error": error, "iteration": iteration,
            "converged": valid, "converged2d": result["converged"],
            "center2d": result["yc"] if result["converged"] else np.nan,
            "width2d": result["wy"] if result["converged"] else np.nan,
            "error2d": result["residual"], "iteration2d": result["iteration"]}


def compute_scan(reference):
    x, kernel = reference["x"], reference["kernel"]
    scans = []
    seconds = []
    for theta_list in [np.linspace(-0.01, 0.01, 21), np.linspace(0, 0.05, 11)]:
        start = time.perf_counter()
        rows = []
        for theta in theta_list:
            u, _, _, iteration, converged = calc_tilt(theta, x, kernel)
            phase = np.exp(1j * 4 * np.pi * theta * x)[:, None]
            stats = calc_stats(x, np.outer(u, reference["u_zero"]), kernel, phase)
            valid = converged and stats["residual"] < 1e-4
            values = [stats[key] if valid else np.nan for key in ("yc", "wy")]
            mu_abs = abs(stats["mu"]) if valid else np.nan
            loss = stats["loss"] if valid else np.nan
            rows.append([theta, *values, mu_abs, loss, iteration, stats["residual"]])
        scans.append(np.array(rows))
        seconds.append(time.perf_counter() - start)
    return {"center": scans[0], "change": scans[1], "seconds": seconds}


def draw_scan(fig, result):
    fig.clear()
    axes = fig.subplots(2, 2)
    center, change = result["center"], result["change"]
    axes[0, 0].plot(center[:, 0], -100 * center[:, 0], "--", color=RED,
                    label=r"理论值 $-L\theta$")
    axes[0, 0].plot(center[:, 0], center[:, 1], "o", color=BLUE,
                    markersize=4, markerfacecolor="white", label="数值解")
    axes[0, 0].set(title="光束中心", xlabel=r"镜面倾角 $\theta$ (rad)",
                   ylabel=r"$y_c/\lambda$")
    axes[0, 0].legend()
    axes[0, 1].plot(change[:, 0], change[:, 2], "o-", color=BLUE, markersize=4)
    axes[0, 1].set(title="模半径", xlabel=r"镜面倾角 $\theta$ (rad)",
                   ylabel=r"$w_y/\lambda$")
    axes[1, 0].plot(change[:, 0], change[:, 4] * 100, "o-", color=BLUE, markersize=4)
    axes[1, 0].set(title="往返衍射损耗", xlabel=r"镜面倾角 $\theta$ (rad)", ylabel="损耗 (%)")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    slope = np.polyfit(center[:, 0], center[:, 1], 1)[0]
    error = np.max(np.abs(center[:, 1] + 100 * center[:, 0]))
    axes[1, 1].set_axis_off()
    axes[1, 1].text(0.05, 0.9,
        f"扫描结果\n\n中心拟合斜率：{slope:.6f} λ/rad\n"
        f"理论斜率：−100 λ/rad\n最大中心偏差：{error:.3e} λ\n\n"
        f"θ = 0 rad\n模半径：{change[0, 2]:.6f} λ    损耗：{change[0, 4]*100:.6f}%\n\n"
        f"θ = 0.05 rad\n模半径：{change[-1, 2]:.6f} λ    损耗：{change[-1, 4]*100:.6f}%",
        transform=axes[1, 1].transAxes, va="top", fontsize=10)


def draw_field(fig, ax, x, field, phase=False):
    amplitude = np.abs(field)
    if phase:
        data = np.angle(field * np.exp(-1j * np.angle(field.flat[np.argmax(amplitude)])))
        cmap = "twilight"
        low, high = -np.pi, np.pi
    else:
        data, cmap, low, high = amplitude, "magma", 0, 1
    image = ax.imshow(data, extent=[x[0], x[-1], x[0], x[-1]],
                      origin="lower", cmap=cmap, vmin=low, vmax=high,
                      aspect="equal", interpolation="nearest")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    cbar = fig.colorbar(image, ax=ax, pad=0.03, fraction=0.046)
    if phase:
        cbar.set_ticks(PHASE_TICKS, labels=PHASE_LABELS)
        cbar.set_label(r"$\arg U$ (rad)")
    else:
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.set_label(r"$|U|$")


def draw_base(fig, result):
    fig.clear()
    axes = fig.subplots(2, 2)
    x, u = result["x"], result["u"]
    ax = axes[0, 0]
    ax.plot(x, np.abs(u), color=BLUE, label="Fox-Li 数值解")
    ax.plot(x, np.exp(-x**2 / (100 / np.pi)), "--", color=RED, label="高斯理论解")
    ax.set(title="一维振幅", xlabel=r"$x/\lambda$", ylabel=r"$|U(x)|$", ylim=(0, 1.05))
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    ax.plot(x, np.angle(u * np.exp(-1j * np.angle(u[len(x)//2]))), color=BLUE)
    ax.set(title="一维相位", xlabel=r"$x/\lambda$", ylabel="相位 (rad)", ylim=(-np.pi, np.pi))
    ax.set_yticks(PHASE_TICKS, labels=PHASE_LABELS)
    draw_field(fig, axes[1, 0], result["x2"], result["u2d"])
    axes[1, 0].set_title("二维振幅")
    draw_field(fig, axes[1, 1], result["x2"], result["u2d"], phase=True)
    axes[1, 1].set_title("二维相位")


def draw_convergence(fig, result):
    fig.clear()
    axes = fig.subplots(1, 2)
    iteration = np.arange(1, len(result["gamma"]) + 1)
    axes[0].plot(iteration, np.abs(result["gamma"]), color=BLUE)
    axes[0].set(title=r"$|\gamma_n|$ 收敛曲线", xlabel="迭代次数", ylabel=r"$|\gamma_n|$")
    axes[1].semilogy(iteration, result["error"], color=BLUE)
    axes[1].axhline(1e-4, color=GRAY, linestyle="--", linewidth=1.2, label=r"收敛阈值 $10^{-4}$")
    axes[1].set(title=r"残差 $\epsilon_n$", xlabel="迭代次数", ylabel=r"$\epsilon_n$")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def draw_tilt(fig, reference, result):
    fig.clear()
    grid = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    # 第一行：三幅图并排（基础场、倾斜场、差异图）
    # 1. 基础场（θ=0）
    ax = fig.add_subplot(grid[0, 0])
    amplitude_base = np.abs(reference["u_zero_2d"])
    im = ax.imshow(amplitude_base, extent=[reference["x2"][0], reference["x2"][-1],
                                            reference["x2"][0], reference["x2"][-1]],
                   origin="lower", cmap="magma", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    ax.set_title(r"$\theta = 0$ (基础)")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_label(r"$|U|$")

    # 2. 倾斜场
    ax = fig.add_subplot(grid[0, 1])
    amplitude_tilt = np.abs(result["u2d"])
    im = ax.imshow(amplitude_tilt, extent=[reference["x2"][0], reference["x2"][-1],
                                            reference["x2"][0], reference["x2"][-1]],
                   origin="lower", cmap="magma", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    ax.set_title(rf"$\theta = {result['theta']:.3f}$ rad")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_label(r"$|U|$")

    # 3. 差异图（新增）
    ax = fig.add_subplot(grid[0, 2])
    diff = amplitude_tilt - amplitude_base
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, extent=[reference["x2"][0], reference["x2"][-1],
                                  reference["x2"][0], reference["x2"][-1]],
                   origin="lower", cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff,
                   aspect="equal", interpolation="nearest")
    ax.set_title("振幅差异")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])

    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label(r"$\Delta|U|$")

    # 第二行左：一维截线对比
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(reference["x"], np.abs(reference["u_zero"]), "--", color=GRAY,
            linewidth=2, label=r"$\theta=0$")
    ax.plot(reference["x"], np.abs(result["u"]), color=BLUE, linewidth=2,
            label=rf"$\theta={result['theta']:.3f}$ rad")
    ax.set(title="一维截线对比", xlabel=r"$y/\lambda$", ylabel=r"$|U(y)|$", ylim=(0, 1.05))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 第二行中+右：光束中心对比（占两格）
    ax = fig.add_subplot(grid[1, 1:])
    theta = np.array([0, 0.05])
    ax.plot(theta, -100 * theta, "--", color=RED, linewidth=2.5, label=r"理论值 $y_c = -L\theta$")
    ax.plot(result["theta"], result["center"], "o", color=BLUE, markersize=10,
            markerfacecolor="white", markeredgewidth=2, label="当前数值解", zorder=3)
    ax.set(title="光束中心对比", xlabel=r"镜面倾角 $\theta$ (rad)",
           ylabel=r"$y_c/\lambda$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def draw_bump(fig, x, reference_field, result):
    """绘制局部凸起的完整可视化"""
    fig.clear()
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    converged = result["converged"]

    # 第一行：镜面形貌、二维振幅、二维相位
    # 1. 镜面形貌（显示半球凸起）
    ax = fig.add_subplot(grid[0, 0])
    x0, y0, rb = result.get('x0', 0), result.get('y0', 0), result.get('radius', 2)

    # 导入 bump 模块的 make_bump 函数
    from bump import make_bump
    bump_profile, _ = make_bump(x, x0, y0, rb)  # 解包 (h, phase)

    # 显示镜面形貌
    im = ax.imshow(bump_profile, extent=[x[0], x[-1], x[0], x[-1]],
                   origin="lower", cmap="YlOrRd", vmin=0, vmax=rb*1.2,
                   aspect="equal", interpolation="bilinear")
    ax.contour(bump_profile, levels=5, colors='black', linewidths=0.5, alpha=0.3,
               extent=[x[0], x[-1], x[0], x[-1]])
    ax.plot(x0, y0, 'o', color='blue', markersize=8, markeredgecolor='black',
            markeredgewidth=1, label=f'P=({x0:.1f}, {y0:.1f})λ')
    ax.set_title(f"镜面半球凸起 (rb={rb:.1f}λ)")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label(r"高度 $h/\lambda$")
    ax.legend(loc='upper right', fontsize=8)

    # 2. 二维振幅分布
    ax = fig.add_subplot(grid[0, 1])
    draw_field(fig, ax, x, result["field"])
    ax.set_title("自再现模式振幅" if converged else "暂态振幅（未收敛）")

    # 3. 二维相位分布
    ax = fig.add_subplot(grid[0, 2])
    draw_field(fig, ax, x, result["field"], phase=True)
    ax.set_title("相位分布" if converged else "暂态相位（未收敛）")

    # 第二行：一维截线、收敛曲线、统计信息
    # 4. 一维中心截线对比
    ax = fig.add_subplot(grid[1, 0])
    center_idx = len(x) // 2
    ax.plot(x, np.abs(reference_field[center_idx, :]), '--', color=GRAY,
            linewidth=1.5, label='无凸起')
    ax.plot(x, np.abs(result["field"][center_idx, :]), color=BLUE,
            linewidth=1.5, label=f'rb={rb:.1f}λ')
    ax.set_title("中心截线对比 (y=0)")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$|U|$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. 收敛曲线
    ax = fig.add_subplot(grid[1, 1])
    if 'errors' in result and len(result['errors']) > 0:
        iterations = np.arange(1, len(result['errors']) + 1)
        ax.semilogy(iterations, result['errors'], color=BLUE, linewidth=1.5)
        ax.axhline(1e-4, color=GRAY, linestyle='--', linewidth=1.2,
                   label=r'阈值 $10^{-4}$')
        ax.set_title("残差收敛过程")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel(r"残差 $\epsilon$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.text(0.5, 0.5, '收敛数据不可用', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()

    # 6. 统计信息文本框
    ax = fig.add_subplot(grid[1, 2])
    ax.set_axis_off()

    state = "已收敛" if converged else "未收敛"
    iteration = result["iteration"]
    residual = result["residual"]

    info_text = (
        f"凸起参数\n"
        f"  位置: ({x0:.1f}, {y0:.1f})λ\n"
        f"  半径: {rb:.2f}λ\n"
        f"  高度: {rb:.2f}λ\n\n"
        f"收敛状态\n"
        f"  状态: {state}\n"
        f"  迭代次数: {iteration}\n"
        f"  最终残差: {residual:.2e}"
    )
    if converged:
        info_text += f"\n\n模式特性\n  往返损耗: {result['loss'] * 100:.4f}%"
        if 'overlap' in result:
            info_text += f"\n  与基模重叠度: {result['overlap']:.4f}"
        else:
            info_text += "\n  与基模重叠度: 未计算"
    else:
        info_text += "\n\n仅展示暂态场和残差"

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


class Demo:
    # 一个窗口保存少量界面状态，不参与物理计算
    def __init__(self, root):
        self.root = root
        self.base = None
        self.tilt_reference = None
        self.tilt = None
        self.buttons = []
        self.busy_controls = []
        self.busy = False
        self.pending_tilt = None
        self.theta_dragging = False
        self.scan = None
        self.scan_after = None
        self.bump_reference = None
        self.bump_after = None
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.title("Fox-Li 光学谐振腔仿真")
        root.configure(bg=COLORS['background'])
        width = min(1200, root.winfo_screenwidth() - 60)
        height = min(800, root.winfo_screenheight() - 90)
        root.geometry(f"{width}x{height}")

        # 配置 ttk 样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=COLORS['background'])
        self.style.configure('Header.TFrame', background=COLORS['surface'])
        self.style.configure('TLabel', background=COLORS['background'],
                           foreground=COLORS['text'])
        self.style.configure('Header.TLabel', background=COLORS['surface'],
                           foreground=COLORS['text'])

        header = ttk.Frame(root, padding=10, style='Header.TFrame')
        header.pack(fill="x")
        ttk.Label(header, text="Fox-Li 光学谐振腔仿真",
                  font=("Microsoft YaHei", 16, "bold"),
                  style='Header.TLabel').pack(side="left")
        ttk.Label(header, text="陈昀泽 · U202413823 · 光实2401",
                  style='Header.TLabel').pack(side="left", padx=16)
        self.status = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(header, textvariable=self.status,
                                     style='Header.TLabel')
        self.status_label.pack(side="right", padx=10)

        self.quick_button = self.add_button(header, "快速演示", self.quick_demo)
        self.quick_button.pack(side="right", padx=10)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.base_page = ttk.Frame(self.tabs)
        self.tabs.add(self.base_page, text="📊 基础模式")
        bar = ttk.Frame(self.base_page, padding=8)
        bar.pack(fill="x")
        self.add_button(bar, "运行", self.run_base).pack(side="left", padx=(0, 12))
        ttk.Label(bar, text="对称共焦腔   L/λ = R/λ = 100   "
                  "网格：一维 101，二维 101×101   收敛阈值：1e-4（连续 10 次）").pack(side="left")
        self.base_info = tk.StringVar(value="")
        ttk.Label(self.base_page, textvariable=self.base_info, padding=(10, 0, 10, 5)).pack(fill="x")
        self.base_fig, self.base_canvas = self.make_canvas(self.base_page)
        self.convergence_page = ttk.Frame(self.tabs)
        self.tabs.add(self.convergence_page, text="📈 收敛与本征值")
        bar = ttk.Frame(self.convergence_page, padding=8)
        bar.pack(fill="x")
        self.add_button(bar, "运行", self.run_convergence).pack(side="left", padx=(0, 12))
        ttk.Label(bar, text="γ = 1 / μ").pack(side="left")
        self.convergence_info = tk.StringVar(value="")
        ttk.Label(self.convergence_page, textvariable=self.convergence_info,
                  padding=(10, 0, 10, 5)).pack(fill="x")
        self.convergence_fig, self.convergence_canvas = self.make_canvas(self.convergence_page)
        self.tilt_page = ttk.Frame(self.tabs)
        self.tabs.add(self.tilt_page, text="🔄 镜面倾斜")
        controls = ttk.Frame(self.tilt_page, padding=10, width=250)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)
        ttk.Label(controls, text="镜面倾角 θ (rad)", font=("Microsoft YaHei", 11)).pack(anchor="w", pady=(0, 8))
        self.theta_text = tk.StringVar(value="θ = 0.02500 rad")
        ttk.Label(controls, textvariable=self.theta_text).pack(anchor="w")
        self.theta_scale_value = tk.DoubleVar(value=0.025)
        self.theta_scale = ttk.Scale(controls, from_=0, to=0.05,
                                     variable=self.theta_scale_value, command=self.on_scale)
        self.theta_scale.pack(fill="x", pady=8)
        self.theta_scale.bind("<ButtonPress-1>", self.start_theta_drag)
        self.theta_scale.bind("<ButtonRelease-1>", self.end_theta_drag)
        self.busy_controls.append(self.theta_scale)

        # 状态指示器
        self.theta_status = tk.StringVar(value="")
        status_label = ttk.Label(controls, textvariable=self.theta_status,
                                foreground=COLORS['text_secondary'],
                                font=('Microsoft YaHei', 9))
        status_label.pack(anchor="w", pady=(2, 8))

        shortcuts = ttk.Frame(controls)
        shortcuts.pack(fill="x", pady=(10, 10))
        for value in [0, 0.005, 0.025, 0.05]:
            button = self.add_button(shortcuts, str(value),
                                     lambda value=value: self.choose_theta(value))
            button.configure(width=5)
            button.pack(side="left", expand=True)
        self.tilt_info = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.tilt_info, wraplength=225,
                  justify="left").pack(anchor="w", fill="x")
        plots = ttk.Frame(self.tilt_page)
        plots.pack(side="left", fill="both", expand=True)
        self.tilt_fig, self.tilt_canvas = self.make_canvas(plots)
        self.bump_page = ttk.Frame(self.tabs)
        self.tabs.add(self.bump_page, text="🏔️ 镜面局部凸起")

        # 左侧控制面板
        controls = ttk.Frame(self.bump_page, padding=10, width=280)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)

        # 标题
        ttk.Label(controls, text="凸起参数",
                  font=("Microsoft YaHei", 11, "bold")).pack(anchor="w", pady=(0,10))

        # 镜面预览图
        self.bump_preview_fig = Figure(figsize=(2, 2), dpi=80)
        self.bump_preview_ax = self.bump_preview_fig.add_subplot(111)
        self.bump_preview_canvas = FigureCanvasTkAgg(self.bump_preview_fig, master=controls)
        self.bump_preview_canvas.get_tk_widget().pack(pady=4)

        # 参数滑块
        self.bump_x = tk.DoubleVar(value=0)
        self.bump_y = tk.DoubleVar(value=0)
        self.bump_r = tk.DoubleVar(value=2)

        ttk.Label(controls, text="位置 x₀/λ", font=("Microsoft YaHei", 9)).pack(anchor="w", pady=(6,2))
        self.bump_x_text = tk.StringVar(value="x₀ = 0.0 λ")
        ttk.Label(controls, textvariable=self.bump_x_text,
                  foreground=COLORS['text_secondary']).pack(anchor="w")
        x_scale = ttk.Scale(controls, from_=-8, to=8,
                           variable=self.bump_x,
                           command=self.on_bump_param_change)
        x_scale.pack(fill="x", pady=5)
        self.busy_controls.append(x_scale)

        ttk.Label(controls, text="位置 y₀/λ", font=("Microsoft YaHei", 9)).pack(anchor="w", pady=(6,2))
        self.bump_y_text = tk.StringVar(value="y₀ = 0.0 λ")
        ttk.Label(controls, textvariable=self.bump_y_text,
                  foreground=COLORS['text_secondary']).pack(anchor="w")
        y_scale = ttk.Scale(controls, from_=-8, to=8,
                           variable=self.bump_y,
                           command=self.on_bump_param_change)
        y_scale.pack(fill="x", pady=5)
        self.busy_controls.append(y_scale)

        ttk.Label(controls, text="半径 rb/λ", font=("Microsoft YaHei", 9)).pack(anchor="w", pady=(6,2))
        self.bump_r_text = tk.StringVar(value="rb = 2.0 λ")
        ttk.Label(controls, textvariable=self.bump_r_text,
                  foreground=COLORS['text_secondary']).pack(anchor="w")
        r_scale = ttk.Scale(controls, from_=0, to=4,
                           variable=self.bump_r,
                           command=self.on_bump_param_change)
        r_scale.pack(fill="x", pady=5)
        self.busy_controls.append(r_scale)

        # 快捷位置按钮
        ttk.Label(controls, text="快捷位置", font=("Microsoft YaHei", 9)).pack(anchor="w", pady=(8,5))
        shortcuts = ttk.Frame(controls)
        shortcuts.pack(fill="x", pady=(0, 10))
        positions = [
            ("中心", 0, 0, 2),
            ("偏1λ", 1, 0, 2),
            ("偏4λ", 4, 0, 2),
            ("偏6λ", 6, 0, 2),
        ]
        for label, x, y, r in positions:
            btn = self.add_button(shortcuts, text=label,
                           command=lambda x=x,y=y,r=r: self.set_bump_params(x,y,r))
            btn.configure(width=5)
            btn.pack(side="left", expand=True, padx=1)

        # 计算按钮
        calc_btn = self.add_button(controls, "🔄 计算凸起效果", self.run_bump)
        calc_btn.pack(pady=8, fill='x')

        # 右侧结果区域
        plots = ttk.Frame(self.bump_page)
        plots.pack(side="left", fill="both", expand=True)
        self.bump_info = tk.StringVar(value="调节参数后点击计算")
        info_label = ttk.Label(plots, textvariable=self.bump_info,
                               justify="left", padding=10, wraplength=600,
                               font=("Microsoft YaHei", 9))
        info_label.pack(fill="x")
        info_label.bind("<Configure>",
                        lambda event: info_label.configure(wraplength=max(100, event.width - 20)))
        self.bump_fig, self.bump_canvas = self.make_canvas(plots)

        # 初始化预览图
        self.update_bump_preview()
        self.scan_page = ttk.Frame(self.tabs)
        self.tabs.add(self.scan_page, text="🔍 参数扫描")
        bar = ttk.Frame(self.scan_page, padding=8)
        bar.pack(fill="x")
        self.scan_button = self.add_button(bar, "运行参数扫描", self.run_scan)
        self.scan_button.pack(side="left")
        self.scan_fig, self.scan_canvas = self.make_canvas(self.scan_page)

        # 总览页面（快速对比模式）
        self.overview_page = ttk.Frame(self.tabs)
        self.tabs.insert(0, self.overview_page, text="📊 总览")
        overview_controls = ttk.Frame(self.overview_page, padding=10)
        overview_controls.pack(side="top", fill="x")
        ttk.Label(overview_controls, text="快速验证所有核心结果",
                  font=("Microsoft YaHei", 11, "bold")).pack(side="left", padx=10)
        self.add_button(overview_controls, "🔄 生成总览", self.generate_overview).pack(side="left", padx=5)
        self.overview_fig, self.overview_canvas = self.make_canvas(self.overview_page)

        # 计算时遮住旧结果和页签，避免误操作
        self.busy_panel = ttk.Frame(root, padding=30)
        card = ttk.Frame(self.busy_panel)
        card.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(card, text="正在计算，请稍候…", foreground=BLUE,
                  font=("Microsoft YaHei", 24, "bold")).pack(pady=(0, 18))
        ttk.Label(card, textvariable=self.status, wraplength=500,
                  justify="center", font=("Microsoft YaHei", 12)).pack(pady=(0, 15))
        self.progress = ttk.Progressbar(card, mode="indeterminate", length=360)
        self.progress.pack(pady=(0, 18))
        ttk.Label(card, text="计算完成后自动恢复操作；期间不能切换页面或修改参数",
                  font=("Microsoft YaHei", 10)).pack()

    def add_button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.buttons.append(button)
        return button

    def make_canvas(self, parent):
        fig = Figure(figsize=(10, 6), dpi=100, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def set_busy(self, busy, message="计算中…"):
        self.busy = busy
        if busy:
            self.cancel_tilt()
            self.status.set(message)
            self.status_label.config(foreground=RED)
            self.active_tab = self.tabs.select()
            self.tab_states = {tab: self.tabs.tab(tab, "state") for tab in self.tabs.tabs()}
            for tab in self.tabs.tabs():
                if tab != self.active_tab:
                    self.tabs.tab(tab, state="disabled")
            self.busy_panel.place(in_=self.tabs, x=0, y=0, relwidth=1, relheight=1)
            self.busy_panel.lift()
            self.root.configure(cursor="watch")
            self.progress.start()
        else:
            self.status_label.config(foreground=COLORS['success'])
            self.progress.stop()
            self.busy_panel.place_forget()
            self.root.configure(cursor="")
            for tab, state in self.tab_states.items():
                self.tabs.tab(tab, state=state)
        for widget in self.buttons + self.busy_controls:
            widget.configure(state="disabled" if busy else "normal")
        self.root.update_idletasks()

    def run_base(self):
        if self.busy:
            return
        self.set_busy(True, "计算基础模式…")
        start = time.perf_counter()
        try:
            self.base = compute_base()
            self.show_base()
            self.show_convergence()
            self.base_seconds = time.perf_counter() - start
            self.status.set(f"✅ 完成 · 基础模式 {self.base_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("❌ 计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def show_base(self):
        draw_base(self.base_fig, self.base)
        self.base_canvas.draw()
        self.base_info.set(
            f"一维迭代：{len(self.base['gamma'])}   数值 1/e 半径：{self.base['radius']:.4f} λ   "
            f"理论半径：{np.sqrt(100 / np.pi):.4f} λ   "
            f"二维迭代：{self.base['iteration2d']}，往返残差：{self.base['error2d']:.3e}"
            + ("" if self.base["converged"] and self.base["converged2d"] else "   未达到收敛判据")
        )

    def show_convergence(self):
        result = self.base
        gamma = result["gamma"][-1]
        converged = "是" if result["converged"] else "否"
        self.convergence_info.set(
            f"已收敛：{converged}   迭代次数：{len(result['gamma'])}   "
            f"最终残差：{result['error'][-1]:.3e}   "
            f"|γ|：{abs(gamma):.8f}   arg γ：{np.angle(gamma):.6f} rad"
        )
        draw_convergence(self.convergence_fig, result)
        self.convergence_canvas.draw()

    def run_convergence(self):
        if self.busy:
            return
        if self.base is None:
            self.run_base()
        else:
            self.show_convergence()
            self.status.set("✅ 完成")

    def cancel_tilt(self):
        if self.pending_tilt is not None:
            self.root.after_cancel(self.pending_tilt)
            self.pending_tilt = None

    def set_theta(self, theta):
        self.cancel_tilt()
        self.theta_scale_value.set(theta)
        self.theta_text.set(f"θ = {theta:.5f} rad")

    def choose_theta(self, theta):
        if self.busy:
            return
        self.set_theta(theta)
        self.run_tilt()

    def on_scale(self, value):
        self.theta_text.set(f"θ = {float(value):.5f} rad")
        self.cancel_tilt()
        if self.busy:
            return
        if self.theta_dragging:
            self.theta_status.set("松开滑块后自动计算")
        else:
            self.theta_status.set("等待自动计算…")
            self.pending_tilt = self.root.after(200, self.run_tilt)

    def start_theta_drag(self, event):
        if self.busy:
            return
        self.theta_dragging = True
        self.cancel_tilt()

    def end_theta_drag(self, event):
        self.theta_dragging = False
        if not self.busy:
            self.on_scale(self.theta_scale_value.get())

    def run_scan(self):
        if self.busy:
            return
        self.set_busy(True, "运行参数扫描…")
        self.scan_start = time.perf_counter()
        # 后台只算 NumPy；主线程定时读取结果并绘图
        self.scan_result = None
        self.scan_error = None
        self.scan_done = threading.Event()
        threading.Thread(target=self.scan_worker, daemon=True).start()
        self.scan_after = self.root.after(50, self.check_scan)

    def scan_worker(self):
        try:
            reference = self.tilt_reference
            if reference is None:
                reference = make_tilt_reference()
            self.scan_result = compute_scan(reference)
        except Exception as error:
            self.scan_error = error
        finally:
            self.scan_done.set()

    def check_scan(self):
        self.scan_after = None
        if not self.scan_done.is_set():
            self.scan_after = self.root.after(50, self.check_scan)
            return
        try:
            if self.scan_error is not None:
                raise self.scan_error
            self.scan = self.scan_result
            draw_scan(self.scan_fig, self.scan)
            self.scan_canvas.draw()
            self.scan_seconds = time.perf_counter() - self.scan_start
            self.status.set(f"✅ 参数扫描完成 · {self.scan_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("❌ 计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def close(self):
        self.cancel_tilt()
        if self.scan_after is not None:
            self.root.after_cancel(self.scan_after)
        if self.bump_after is not None:
            self.root.after_cancel(self.bump_after)
        self.root.destroy()

    def run_bump(self):
        if self.busy:
            return
        x0, y0, radius = self.bump_x.get(), self.bump_y.get(), self.bump_r.get()
        if radius < 0:
            messagebox.showerror("参数错误", "凸起半径不能为负数", parent=self.root)
            return
        self.set_busy(True, f"计算局部凸起：P=({x0:.2f}, {y0:.2f})λ，rb={radius:.2f}λ")
        self.bump_info.set("二维往返计算中……")
        self.bump_done = threading.Event()
        self.bump_error = None
        self.bump_input = (x0, y0, radius)
        threading.Thread(target=self.bump_worker, daemon=True).start()
        self.bump_after = self.root.after(50, self.check_bump)

    def bump_worker(self):
        try:
            x = np.linspace(-10, 10, 101)
            kernel = bump.make_kernel(x)
            if self.bump_reference is None:
                self.bump_reference = bump.calc_bump(x, kernel, radius=0)
            result = bump.calc_bump(x, kernel, *self.bump_input)
            self.bump_result = (x, result)
        except Exception as error:
            self.bump_error = error
        finally:
            self.bump_done.set()

    def check_bump(self):
        self.bump_after = None
        if not self.bump_done.is_set():
            self.bump_after = self.root.after(50, self.check_bump)
            return
        try:
            if self.bump_error is not None:
                raise self.bump_error
            x, result = self.bump_result
            x0, y0, radius = self.bump_input

            # 使用新的绘图函数
            result_with_params = {
                **result,
                'x0': x0,
                'y0': y0,
                'radius': radius,
            }
            if result["converged"] and self.bump_reference["converged"]:
                result_with_params['overlap'] = bump.overlap(result['field'], self.bump_reference['field'])
            draw_bump(self.bump_fig, x, self.bump_reference["field"], result_with_params)
            self.bump_canvas.draw()

            # 更新信息栏
            converged = "已收敛" if result["converged"] else "未收敛（达到最大迭代次数）"
            info = (f"P=({x0:.2f}, {y0:.2f})λ，r_b={radius:.2f}λ | {converged}\n"
                    f"迭代 {result['iteration']} 次，残差 {result['residual']:.2e}")
            if result["converged"]:
                info += f" | 往返损耗 {result['loss'] * 100:.4f}%"
                if 'overlap' in result_with_params:
                    info += f" | 与无缺陷基模重叠度 {result_with_params['overlap']:.4f}"
                else:
                    info += " | 重叠度未计算（无凸起参考场未收敛）"
            self.bump_info.set(info)
            self.status.set("局部凸起计算完成（已收敛）" if result["converged"]
                            else "局部凸起迭代结束（未收敛）")
        except Exception as error:
            self.status.set("❌ 计算失败")
            self.bump_info.set(f"计算失败：{error}")
            messagebox.showerror("局部凸起计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def update_bump_preview(self):
        """更新凸起镜面预览图"""
        x0 = self.bump_x.get()
        y0 = self.bump_y.get()
        rb = self.bump_r.get()

        # 更新文本显示
        self.bump_x_text.set(f"x₀ = {x0:.1f} λ")
        self.bump_y_text.set(f"y₀ = {y0:.1f} λ")
        self.bump_r_text.set(f"rb = {rb:.1f} λ")

        ax = self.bump_preview_ax
        ax.clear()

        # 计算孔径为 [-10, 10]λ × [-10, 10]λ
        aperture = Rectangle((-10, -10), 20, 20, edgecolor='lightgray',
                             fill=False, linewidth=2.5, linestyle='--', label='镜面边界')
        ax.add_patch(aperture)

        # 绘制坐标原点
        ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)

        # 绘制凸起位置
        if rb > 0:
            bump_circle = Circle((x0, y0), rb, facecolor='#E64B35',
                                alpha=0.35, edgecolor='#E64B35', linewidth=2.5)
            ax.add_patch(bump_circle)
            ax.plot(x0, y0, 'o', color='#E64B35', markersize=10, markeredgewidth=1.5,
                   markeredgecolor='darkred')

            # 标注信息
            if abs(x0) < 8 and abs(y0) < 8:
                ax.annotate(f'P({x0:.1f}, {y0:.1f})λ\nrb={rb:.1f}λ',
                           xy=(x0, y0), xytext=(x0+2, y0+2),
                           fontsize=8, color='darkred', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x/\lambda$', fontsize=9)
        ax.set_ylabel(r'$y/\lambda$', fontsize=9)
        ax.set_title('镜面预览（方形孔径）', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(labelsize=8)

        self.bump_preview_fig.tight_layout()
        self.bump_preview_canvas.draw()

    def on_bump_param_change(self, value):
        """凸起参数变化时更新预览"""
        self.update_bump_preview()

    def set_bump_params(self, x, y, r):
        """设置凸起参数并自动计算"""
        if self.busy:
            return
        self.bump_x.set(x)
        self.bump_y.set(y)
        self.bump_r.set(r)
        self.update_bump_preview()
        self.run_bump()

    def show_tilt(self):
        result = self.tilt
        theory = -100 * result["theta"]
        self.tilt_info.set(
            f"倾角 θ：{result['theta']:.5f} rad\n\n"
            f"一维截线\n"
            f"光束中心 y_c/λ：{result['center']:.6f}\n"
            f"理论值 -Lθ/λ：{theory:.6f}\n"
            f"偏差/λ：{result['center'] - theory:+.3e}\n"
            f"模半径 w_y/λ：{result['width']:.6f}\n"
            f"迭代次数：{result['iteration']}\n"
            f"最终残差：{result['error']:.3e}\n\n"
            f"二维场（101×101）\n"
            + (f"二维往返损耗：{result['loss'] * 100:.6f}%\n"
               if result["converged2d"] else "未收敛，不报告稳态损耗\n")
            +
            f"光束中心 y_c/λ：{result['center2d']:.6f}\n"
            f"模半径 w_y/λ：{result['width2d']:.6f}\n"
            f"迭代次数：{result['iteration2d']}\n"
            f"最终残差：{result['error2d']:.3e}"
        )
        draw_tilt(self.tilt_fig, self.tilt_reference, result)
        self.tilt_canvas.draw()

    def run_tilt(self):
        if self.busy:
            return
        self.cancel_tilt()
        theta = self.theta_scale_value.get()
        self.set_busy(True, f"计算镜面倾斜：θ={theta:.3f} rad…")
        self.theta_status.set("🔄 计算中...")
        start = time.perf_counter()
        try:
            if self.tilt_reference is None:
                self.tilt_reference = make_tilt_reference()
            self.tilt = compute_tilt(theta, self.tilt_reference)
            self.show_tilt()
            self.tilt_seconds = time.perf_counter() - start
            self.theta_status.set(f"✅ 完成 ({self.tilt_seconds:.2f}s)")
            self.status.set(f"✅ 完成 · 倾斜仿真 {self.tilt_seconds:.2f} 秒")
        except Exception as error:
            self.theta_status.set("❌ 计算失败")
            self.status.set("❌ 计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def quick_demo(self):
        if self.busy:
            return
        self.set_busy(True, "快速演示计算中…")
        start = time.perf_counter()
        try:
            self.base = compute_base()
            self.show_base()
            self.show_convergence()
            self.base_seconds = time.perf_counter() - start
            self.set_theta(0.025)
            tilt_start = time.perf_counter()
            if self.tilt_reference is None:
                self.tilt_reference = make_tilt_reference()
            self.tilt = compute_tilt(0.025, self.tilt_reference)
            self.show_tilt()
            self.tilt_seconds = time.perf_counter() - tilt_start
            self.quick_seconds = time.perf_counter() - start
            self.status.set(f"✅ 完成 · 快速演示 {self.quick_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("❌ 计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def generate_overview(self):
        """生成总览页面 - 3×2 网格展示所有关键结果"""
        if self.busy:
            return
        self.set_busy(True, "生成总览中…")
        start = time.perf_counter()
        try:
            # 确保所有基础数据已计算
            if self.base is None:
                self.base = compute_base()
            if self.tilt_reference is None:
                self.tilt_reference = make_tilt_reference()
            if self.tilt is None:
                self.tilt = compute_tilt(0.025, self.tilt_reference)

            # 绘制总览图
            self.draw_overview()

            elapsed = time.perf_counter() - start
            self.status.set(f"✅ 完成 · 总览生成 {elapsed:.2f} 秒")
        except Exception as error:
            self.status.set("❌ 生成失败")
            messagebox.showerror("生成失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def draw_overview(self):
        """绘制总览页面：3×2 网格"""
        self.overview_fig.clear()
        gs = self.overview_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # 左上：基础模式一维验证
        ax = self.overview_fig.add_subplot(gs[0, 0])
        x, u = self.base["x"], self.base["u"]
        ax.plot(x, np.abs(u), color=BLUE, linewidth=2, label="数值解")
        ax.plot(x, np.exp(-x**2 / (100 / np.pi)), "--", color=RED, linewidth=2, label="理论解")
        ax.set_title("基础模式验证", fontweight='bold', fontsize=10)
        ax.set_xlabel(r"$x/\lambda$", fontsize=9)
        ax.set_ylabel(r"$|U(x)|$", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 右上：收敛曲线
        ax = self.overview_fig.add_subplot(gs[0, 1])
        iteration = np.arange(1, len(self.base["gamma"]) + 1)
        ax.semilogy(iteration, self.base["error"], color=BLUE, linewidth=2)
        ax.axhline(1e-4, color=GRAY, linestyle="--", linewidth=1.5, label=r"阈值 $10^{-4}$")
        ax.set_title("收敛性", fontweight='bold', fontsize=10)
        ax.set_xlabel("迭代次数", fontsize=9)
        ax.set_ylabel(r"残差 $\epsilon_n$", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

        # 左中：倾斜场对比
        ax = self.overview_fig.add_subplot(gs[1, 0])
        ax.plot(self.tilt_reference["x"], np.abs(self.tilt_reference["u_zero"]),
                "--", color=GRAY, linewidth=2, label=r"$\theta=0$")
        ax.plot(self.tilt_reference["x"], np.abs(self.tilt["u"]),
                color=BLUE, linewidth=2, label=rf"$\theta={self.tilt['theta']:.3f}$ rad")
        ax.set_title("镜面倾斜效果", fontweight='bold', fontsize=10)
        ax.set_xlabel(r"$y/\lambda$", fontsize=9)
        ax.set_ylabel(r"$|U(y)|$", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 右中：光束中心 vs 理论
        ax = self.overview_fig.add_subplot(gs[1, 1])
        theta_range = np.array([0, 0.05])
        ax.plot(theta_range, -100 * theta_range, "--", color=RED, linewidth=2.5,
                label=r"理论 $y_c=-L\theta$")
        ax.plot(self.tilt["theta"], self.tilt["center"], "o", color=BLUE,
                markersize=12, markerfacecolor="white", markeredgewidth=2.5,
                label="数值解", zorder=3)
        ax.set_title("倾斜定量规律", fontweight='bold', fontsize=10)
        ax.set_xlabel(r"$\theta$ (rad)", fontsize=9)
        ax.set_ylabel(r"$y_c/\lambda$", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 左下：二维场对比（基础 vs 倾斜）
        ax = self.overview_fig.add_subplot(gs[2, 0])
        extent = [self.tilt_reference["x2"][0], self.tilt_reference["x2"][-1],
                  self.tilt_reference["x2"][0], self.tilt_reference["x2"][-1]]
        im = ax.imshow(np.abs(self.tilt_reference["u_zero_2d"]), extent=extent,
                      origin="lower", cmap="magma", vmin=0, vmax=1,
                      aspect="equal", interpolation="nearest")
        ax.set_title(r"基础场 ($\theta=0$)", fontweight='bold', fontsize=10)
        ax.set_xlabel(r"$x/\lambda$", fontsize=9)
        ax.set_ylabel(r"$y/\lambda$", fontsize=9)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])

        # 右下：倾斜场
        ax = self.overview_fig.add_subplot(gs[2, 1])
        im = ax.imshow(np.abs(self.tilt["u2d"]), extent=extent,
                      origin="lower", cmap="magma", vmin=0, vmax=1,
                      aspect="equal", interpolation="nearest")
        ax.set_title(rf"倾斜场 ($\theta={self.tilt['theta']:.3f}$ rad)",
                    fontweight='bold', fontsize=10)
        ax.set_xlabel(r"$x/\lambda$", fontsize=9)
        ax.set_ylabel(r"$y/\lambda$", fontsize=9)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])

        self.overview_canvas.draw()


def main():
    set_style()
    root = tk.Tk()
    Demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
