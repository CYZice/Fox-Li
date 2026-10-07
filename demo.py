import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator


BLUE = "#3C5488"
RED = "#E64B35"
GRAY = "#666666"
PHASE_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PHASE_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]

# 核心函数与 main.py 一致；不导入其顶层扫描和文件输出
def make_kernel(x):
    dx = x[1] - x[0]
    # 长度都用波长作单位
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


def calc_tilt(theta, x, kernel, iteration=3000):
    k = 2 * np.pi
    u = np.ones(len(x), dtype=complex)
    stable = 0

    for n in range(iteration):
        # 从正常镜传播到倾斜镜
        u2 = propagate(u, kernel)

        # 倾斜镜反射后产生线性相位
        phase = np.exp(1j * 2 * k * theta * x)
        u2 = u2 * phase

        # 再传播回正常镜
        u2 = propagate(u2, kernel)

        mu = np.vdot(u, u2) / np.vdot(u, u)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        u = u2 / np.max(np.abs(u2))

        if error < 1e-4:
            stable += 1
        else:
            stable = 0

        if stable >= 10:
            break

    return u, mu, error, n + 1


def make_kernel2d(x):
    dx = x[1] - x[0]
    k = 2 * np.pi
    distance = 100
    kernel = np.zeros((len(x), len(x)), dtype=complex)

    for j in range(len(x)):
        for i in range(len(x)):
            kernel[j, i] = (
                np.exp(1j * k * x[j] * x[i] / distance)
                * dx
            )

    return kernel


def propagate2d(u, kernel):
    return kernel @ u @ kernel.T


def calc_tilt2d(theta, x, kernel, iteration=3000):
    k = 2 * np.pi
    u = np.ones((len(x), len(x)), dtype=complex)
    phase = np.exp(1j * 2 * k * theta * x)[:, None]
    stable = 0

    for n in range(iteration):
        u2 = propagate2d(u, kernel)
        u2 = phase * u2
        u2 = propagate2d(u2, kernel)

        mu = np.vdot(u, u2) / np.vdot(u, u)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        u = u2 / np.max(np.abs(u2))

        if error < 1e-4:
            stable += 1
        else:
            stable = 0

        if stable >= 10:
            break

    return u, error, n + 1


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
    x2 = np.linspace(-10, 10, 51)
    kernel2d = make_kernel2d(x2)
    u2d = np.ones((51, 51), dtype=complex)
    # 基础二维与 main.py 一样，使用一维迭代次数
    for n in range(len(gamma_list)):
        u2d = propagate2d(u2d, kernel2d)
        u2d = u2d / np.max(np.abs(u2d))

    return {"x": x, "u": u, "x2": x2, "u2d": u2d,
            "gamma": np.array(gamma_list), "error": np.array(error_list),
            "radius": x[positive][index], "converged": stable >= 10}


def set_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5,
        "legend.fontsize": 8, "legend.frameon": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.formatter.useoffset": False,
    })


def make_tilt_reference():
    x = np.linspace(-10, 10, 101)
    x2 = np.linspace(-10, 10, 51)
    kernel = make_kernel(x)
    u_zero, mu_x, _, _ = calc_tilt(0, x, kernel)
    return {"x": x, "kernel": kernel, "x2": x2,
            "kernel2d": make_kernel2d(x2), "u_zero": u_zero, "mu_x": mu_x}


def compute_tilt(theta, reference):
    x, x2 = reference["x"], reference["x2"]
    u, mu, error, iteration = calc_tilt(theta, x, reference["kernel"])
    intensity = np.abs(u) ** 2
    center = np.sum(x * intensity) / np.sum(intensity)
    width = 2 * np.sqrt(np.sum((x - center) ** 2 * intensity) / np.sum(intensity))
    # 两个物理方向的完整往返本征值；与报告损耗口径一致
    loss = 1 - np.abs(reference["mu_x"] * mu) ** 2
    u2d, error2d, iteration2d = calc_tilt2d(theta, x2, reference["kernel2d"])
    intensity2d = np.abs(u2d) ** 2
    center2d = np.sum(x2[:, None] * intensity2d) / np.sum(intensity2d)
    width2d = 2 * np.sqrt(
        np.sum((x2[:, None] - center2d) ** 2 * intensity2d) / np.sum(intensity2d)
    )
    return {"theta": theta, "u": u, "u2d": u2d, "center": center,
            "width": width, "loss": loss, "error": error, "iteration": iteration,
            "center2d": center2d, "width2d": width2d,
            "error2d": error2d, "iteration2d": iteration2d}


def read_theta(text):
    try:
        theta = float(text)
    except ValueError:
        raise ValueError("请输入数值倾角，范围为 0～0.05 rad。") from None
    if not np.isfinite(theta) or not 0 <= theta <= 0.05:
        raise ValueError("倾角必须在 0～0.05 rad 之间。")
    return theta


def draw_field(fig, ax, x, field, phase=False):
    amplitude = np.abs(field)
    if phase:
        data = np.ma.masked_where(amplitude < 0.1 * amplitude.max(), np.angle(field))
        cmap = matplotlib.colormaps["twilight"].copy()
        cmap.set_bad("white")
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
    ax.plot(x, np.angle(u), color=BLUE)
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
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax = fig.add_subplot(grid[0, 0])
    draw_field(fig, ax, reference["x2"], result["u2d"])
    ax.set_title(rf"二维振幅 ($\theta={result['theta']:.3f}$ rad)")
    ax = fig.add_subplot(grid[0, 1])
    ax.plot(reference["x"], np.abs(reference["u_zero"]), "--", color=GRAY, label=r"$\theta=0$")
    ax.plot(reference["x"], np.abs(result["u"]), color=BLUE,
            label=rf"$\theta={result['theta']:.3f}$ rad")
    ax.set(title="一维截线对比", xlabel=r"$y/\lambda$", ylabel=r"$|U(y)|$", ylim=(0, 1.05))
    ax.legend(loc="upper right")
    ax = fig.add_subplot(grid[1, :])
    theta = np.array([0, 0.05])
    ax.plot(theta, -100 * theta, "--", color=RED, label=r"理论值 $-L\theta$")
    ax.plot(result["theta"], result["center"], "o", color=BLUE,
            markerfacecolor="white", label="当前数值解", zorder=3)
    ax.set(title="光束中心对比", xlabel=r"镜面倾角 $\theta$ (rad)",
           ylabel=r"$y_c/\lambda$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc="upper right")


class Demo:
    # 一个窗口保存少量界面状态，不参与物理计算
    def __init__(self, root):
        self.root = root
        self.base = None
        self.tilt_reference = None
        self.tilt = None
        self.buttons = []
        root.title("Fox-Li 光学谐振腔仿真")
        width = min(1200, root.winfo_screenwidth() - 60)
        height = min(800, root.winfo_screenheight() - 90)
        root.geometry(f"{width}x{height}")
        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Fox-Li 光学谐振腔仿真",
                  font=("Microsoft YaHei", 16)).pack(side="left")
        self.status = tk.StringVar(value="就绪")
        ttk.Label(header, textvariable=self.status).pack(side="right", padx=10)
        self.quick_button = self.add_button(header, "快速演示", self.quick_demo)
        self.quick_button.pack(side="right", padx=10)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.base_page = ttk.Frame(self.tabs)
        self.tabs.add(self.base_page, text="基础模式")
        bar = ttk.Frame(self.base_page, padding=8)
        bar.pack(fill="x")
        self.add_button(bar, "运行", self.run_base).pack(side="left", padx=(0, 12))
        ttk.Label(bar, text="对称共焦腔   L/λ = R/λ = 100   "
                  "网格：一维 101，二维 51×51   收敛阈值：1e-4（连续 10 次）").pack(side="left")
        self.base_info = tk.StringVar(value="")
        ttk.Label(self.base_page, textvariable=self.base_info, padding=(10, 0, 10, 5)).pack(fill="x")
        self.base_fig, self.base_canvas = self.make_canvas(self.base_page)
        self.convergence_page = ttk.Frame(self.tabs)
        self.tabs.add(self.convergence_page, text="收敛与本征值")
        bar = ttk.Frame(self.convergence_page, padding=8)
        bar.pack(fill="x")
        self.add_button(bar, "运行", self.run_convergence).pack(side="left", padx=(0, 12))
        ttk.Label(bar, text="γ = 1 / μ").pack(side="left")
        self.convergence_info = tk.StringVar(value="")
        ttk.Label(self.convergence_page, textvariable=self.convergence_info,
                  padding=(10, 0, 10, 5)).pack(fill="x")
        self.convergence_fig, self.convergence_canvas = self.make_canvas(self.convergence_page)
        self.tilt_page = ttk.Frame(self.tabs)
        self.tabs.add(self.tilt_page, text="镜面倾斜")
        controls = ttk.Frame(self.tilt_page, padding=10, width=250)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)
        ttk.Label(controls, text="镜面倾角 θ (rad)", font=("Microsoft YaHei", 11)).pack(anchor="w", pady=(0, 8))
        self.theta_text = tk.StringVar(value="0.025")
        self.theta_entry = ttk.Entry(controls, textvariable=self.theta_text)
        self.theta_entry.pack(fill="x")
        self.theta_scale_value = tk.DoubleVar(value=0.025)
        self.theta_scale = ttk.Scale(controls, from_=0, to=0.05,
                                     variable=self.theta_scale_value, command=self.on_scale)
        self.theta_scale.pack(fill="x", pady=8)
        shortcuts = ttk.Frame(controls)
        shortcuts.pack(fill="x", pady=(0, 10))
        for value in [0, 0.005, 0.025, 0.05]:
            button = self.add_button(shortcuts, str(value),
                                     lambda value=value: self.set_theta(value))
            button.configure(width=5)
            button.pack(side="left", expand=True)
        self.add_button(controls, "运行倾斜仿真", self.run_tilt).pack(fill="x", pady=(0, 12))
        self.tilt_info = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.tilt_info, wraplength=225,
                  justify="left").pack(anchor="w", fill="x")
        plots = ttk.Frame(self.tilt_page)
        plots.pack(side="left", fill="both", expand=True)
        self.tilt_fig, self.tilt_canvas = self.make_canvas(plots)

    def add_button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.buttons.append(button)
        return button

    def make_canvas(self, parent):
        fig = Figure(figsize=(10, 6), dpi=100, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def set_busy(self, busy):
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        self.theta_entry.configure(state="disabled" if busy else "normal")
        self.theta_scale.configure(state="disabled" if busy else "normal")
        if busy:
            self.status.set("计算中…")
        self.root.update_idletasks()

    def run_base(self):
        self.set_busy(True)
        start = time.perf_counter()
        try:
            self.base = compute_base()
            self.show_base()
            self.show_convergence()
            self.base_seconds = time.perf_counter() - start
            self.status.set(f"完成 · 基础模式 {self.base_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def show_base(self):
        draw_base(self.base_fig, self.base)
        self.base_canvas.draw()
        self.base_info.set(
            f"迭代次数：{len(self.base['gamma'])}   数值 1/e 半径：{self.base['radius']:.4f} λ   "
            f"理论半径：{np.sqrt(100 / np.pi):.4f} λ   最终残差：{self.base['error'][-1]:.3e}"
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
        if self.base is None:
            self.run_base()
        else:
            self.show_convergence()
            self.status.set("完成")

    def set_theta(self, theta):
        self.theta_text.set(f"{theta:.3f}")
        self.theta_scale_value.set(theta)

    def on_scale(self, value):
        self.theta_text.set(f"{float(value):.5f}")

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
            f"往返衍射损耗：{result['loss'] * 100:.6f}%\n"
            f"迭代次数：{result['iteration']}\n"
            f"最终残差：{result['error']:.3e}\n\n"
            f"二维场\n"
            f"光束中心 y_c/λ：{result['center2d']:.6f}\n"
            f"模半径 w_y/λ：{result['width2d']:.6f}\n"
            f"迭代次数：{result['iteration2d']}\n"
            f"最终残差：{result['error2d']:.3e}"
        )
        draw_tilt(self.tilt_fig, self.tilt_reference, result)
        self.tilt_canvas.draw()

    def run_tilt(self):
        try:
            theta = read_theta(self.theta_text.get())
        except ValueError as error:
            messagebox.showerror("倾角输入错误", str(error), parent=self.root)
            return
        self.theta_scale_value.set(theta)
        self.set_busy(True)
        start = time.perf_counter()
        try:
            if self.tilt_reference is None:
                self.tilt_reference = make_tilt_reference()
            self.tilt = compute_tilt(theta, self.tilt_reference)
            self.show_tilt()
            self.tilt_seconds = time.perf_counter() - start
            self.status.set(f"完成 · 倾斜仿真 {self.tilt_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

    def quick_demo(self):
        self.set_busy(True)
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
            self.status.set(f"完成 · 快速演示 {self.quick_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("计算失败")
            messagebox.showerror("计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)


def main():
    set_style()
    root = tk.Tk()
    Demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
