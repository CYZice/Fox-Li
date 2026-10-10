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
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator


BLUE = "#3C5488"
RED = "#E64B35"
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
    kernel = make_kernel(x)
    u_zero, _, _, _, converged = calc_tilt(0, x, kernel)
    if not converged:
        raise ValueError("无扰动参考场未收敛")
    return {"x": x, "kernel": kernel, "x2": x, "u_zero": u_zero}


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
        self.busy = False
        self.pending_tilt = None
        self.scan = None
        self.scan_after = None
        self.bump_reference = None
        self.bump_after = None
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.title("Fox-Li 光学谐振腔仿真")
        width = min(1200, root.winfo_screenwidth() - 60)
        height = min(800, root.winfo_screenheight() - 90)
        root.geometry(f"{width}x{height}")
        header = ttk.Frame(root, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Fox-Li 光学谐振腔仿真",
                  font=("Microsoft YaHei", 16)).pack(side="left")
        ttk.Label(header, text="陈昀泽 · U202413823 · 光实2401").pack(side="left", padx=16)
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
                  "网格：一维 101，二维 101×101   收敛阈值：1e-4（连续 10 次）").pack(side="left")
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
        self.theta_text = tk.StringVar(value="θ = 0.02500 rad")
        ttk.Label(controls, textvariable=self.theta_text).pack(anchor="w")
        self.theta_scale_value = tk.DoubleVar(value=0.025)
        self.theta_scale = ttk.Scale(controls, from_=0, to=0.05,
                                     variable=self.theta_scale_value, command=self.on_scale)
        self.theta_scale.pack(fill="x", pady=8)
        shortcuts = ttk.Frame(controls)
        shortcuts.pack(fill="x", pady=(0, 10))
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
        self.tabs.add(self.bump_page, text="镜面局部凸起")
        bar = ttk.Frame(self.bump_page, padding=8)
        bar.pack(fill="x")
        self.bump_x = tk.DoubleVar(value=0)
        self.bump_y = tk.DoubleVar(value=0)
        self.bump_r = tk.DoubleVar(value=2)
        for text, variable in [("x₀/λ", self.bump_x), ("y₀/λ", self.bump_y),
                               ("r_b/λ", self.bump_r)]:
            ttk.Label(bar, text=text).pack(side="left", padx=(8, 3))
            ttk.Spinbox(bar, from_=-8 if variable != self.bump_r else 0,
                        to=8, increment=0.5, width=6,
                        textvariable=variable).pack(side="left")
        self.add_button(bar, "计算局部凸起", self.run_bump).pack(side="left", padx=15)
        self.bump_info = tk.StringVar(value="101×101 网格；损耗为有限网格结果。设置半球位置和半径后计算")
        ttk.Label(self.bump_page, textvariable=self.bump_info,
                  padding=(12, 0, 10, 4)).pack(fill="x")
        self.bump_fig, self.bump_canvas = self.make_canvas(self.bump_page)
        self.scan_page = ttk.Frame(self.tabs)
        self.tabs.add(self.scan_page, text="参数扫描")
        bar = ttk.Frame(self.scan_page, padding=8)
        bar.pack(fill="x")
        self.scan_button = self.add_button(bar, "运行参数扫描", self.run_scan)
        self.scan_button.pack(side="left")
        self.scan_fig, self.scan_canvas = self.make_canvas(self.scan_page)

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
        self.busy = busy
        if busy:
            self.cancel_tilt()
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
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
        if self.base is None:
            self.run_base()
        else:
            self.show_convergence()
            self.status.set("完成")

    def cancel_tilt(self):
        if self.pending_tilt is not None:
            self.root.after_cancel(self.pending_tilt)
            self.pending_tilt = None

    def set_theta(self, theta):
        self.cancel_tilt()
        self.theta_scale_value.set(theta)
        self.theta_text.set(f"θ = {theta:.5f} rad")

    def choose_theta(self, theta):
        self.set_theta(theta)
        self.run_tilt()

    def on_scale(self, value):
        self.theta_text.set(f"θ = {float(value):.5f} rad")
        self.cancel_tilt()
        if not self.busy:
            self.pending_tilt = self.root.after(200, self.run_tilt)

    def run_scan(self):
        self.set_busy(True)
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
            self.status.set(f"参数扫描完成 · {self.scan_seconds:.2f} 秒")
        except Exception as error:
            self.status.set("计算失败")
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
        self.set_busy(True)
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
            self.bump_fig.clear()
            axes = self.bump_fig.subplots(2, 2)
            axes[0, 0].imshow(result["height"], extent=(-10, 10, -10, 10),
                              origin="lower", cmap="viridis")
            axes[0, 0].set_title("半球凸起高度")
            axes[0, 1].imshow(np.abs(result["field"]), extent=(-10, 10, -10, 10),
                              origin="lower", cmap="magma", vmin=0, vmax=1)
            axes[0, 1].set_title("自再现模式振幅" if result["converged"] else "暂态振幅（未收敛）")
            axes[1, 0].plot(x, np.abs(self.bump_reference["field"][len(x)//2]),
                            "--", color=GRAY, label="无凸起")
            axes[1, 0].plot(x, np.abs(result["field"][len(x)//2]),
                            color=BLUE, label="局部凸起")
            axes[1, 0].set(xlabel="x / λ", ylabel="中心截线振幅")
            axes[1, 0].legend()
            axes[1, 1].semilogy(result["errors"], color=BLUE)
            axes[1, 1].axhline(1e-4, color=GRAY, linestyle="--")
            axes[1, 1].set(xlabel="迭代次数", ylabel="自再现残差")
            self.bump_canvas.draw()
            converged = "已收敛" if result["converged"] else "未达到收敛判据"
            info = (f"P=({x0:g}, {y0:g})λ，r_b={radius:g}λ | {converged} | "
                    f"迭代 {result['iteration']} 次，残差 {result['residual']:.2e}")
            if result["converged"] and self.bump_reference["converged"]:
                info += (f" | 往返损耗 {result['loss'] * 100:.4f}% | "
                         f"与无缺陷基模重叠度 {bump.overlap(result['field'], self.bump_reference['field']):.4f}")
            self.bump_info.set(info)
            self.status.set("局部凸起计算完成")
        except Exception as error:
            self.status.set("计算失败")
            messagebox.showerror("局部凸起计算失败", str(error), parent=self.root)
        finally:
            self.set_busy(False)

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
        self.cancel_tilt()
        theta = self.theta_scale_value.get()
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
