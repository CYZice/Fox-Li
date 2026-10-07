import numpy as np
from plot_results import (
    set_plot_style,
    plot_base_amplitude,
    plot_base_phase,
    plot_gamma,
    plot_error,
    plot_base_2d,
    plot_base_compare,
    plot_tilt_profile,
    plot_tilt_center,
    plot_tilt_width,
    plot_tilt_loss,
    plot_tilt_2d,
    plot_report_base,
    plot_report_convergence,
    plot_report_base_2d,
    plot_report_tilt,
)

# 姓名：
# 学号：
# 班级：

set_plot_style()


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
gaussian = np.exp(-x**2 / (100 / np.pi))
positive = x >= 0
index = np.argmin(np.abs(amplitude[positive] - 1 / np.e))
print("数值 1/e 半径:", x[positive][index])
print("理论 1/e 半径:", np.sqrt(100 / np.pi))


plot_base_amplitude(x, amplitude, gaussian)
plot_base_phase(x, u)
plot_gamma(gamma_list)
plot_error(error_list)


x2 = np.linspace(-10, 10, 51)
kernel2d = make_kernel2d(x2)
u2d = np.ones((51, 51), dtype=complex)

for n in range(len(gamma_list)):
    u2d = propagate2d(u2d, kernel2d)
    u2d = u2d / np.max(np.abs(u2d))

center = len(x2) // 2
plot_base_2d(x2, u2d)
plot_base_compare(
    x2,
    amplitude[::2],
    np.abs(u2d[center]),
    np.exp(-x2**2 / (100 / np.pi)),
)


# 镜面倾斜：一维里的 x 在这里对应物理上的 y 方向
theta = 0.005
# x 方向不受倾斜，二维往返本征值由两个方向相乘
u_zero, mu_x, _, _ = calc_tilt(0, x, kernel)
u_tilt, mu_tilt, error, tilt_iteration = calc_tilt(theta, x, kernel)
print("倾斜腔收敛迭代次数:", tilt_iteration)

amplitude_tilt = np.abs(u_tilt)
amplitude_zero = np.abs(u_zero)
intensity_tilt = amplitude_tilt ** 2
center_tilt = np.sum(x * intensity_tilt) / np.sum(intensity_tilt)
width_tilt = 2 * np.sqrt(
    np.sum((x - center_tilt) ** 2 * intensity_tilt) / np.sum(intensity_tilt)
)
mu_total = mu_x * mu_tilt
loss_tilt = 1 - np.abs(mu_total) ** 2

print("theta:", theta)
print("width:", width_tilt)
print("|mu_2D|:", np.abs(mu_total))
print("2D round trip loss:", loss_tilt)
print("倾斜场最后残差:", error)
print("光束中心:", center_tilt)
print(
    "倾斜场与基础场 RMS 差:",
    np.sqrt(np.mean((amplitude_tilt - amplitude_zero) ** 2)),
)

plot_tilt_profile(x, amplitude_zero, amplitude_tilt, theta)


# 连续扫描小倾角，最大几何偏移约为 1 lambda
theta_list = np.linspace(-0.01, 0.01, 21)
center_list = []
width_list = []
loss_list = []
mu_list = []
iteration_list = []
scan_error_list = []

for theta_scan in theta_list:
    u_scan, mu_scan, error_scan, n_scan = calc_tilt(theta_scan, x, kernel)

    intensity_scan = np.abs(u_scan) ** 2
    center_scan = np.sum(x * intensity_scan) / np.sum(intensity_scan)
    width_scan = 2 * np.sqrt(
        np.sum((x - center_scan) ** 2 * intensity_scan) / np.sum(intensity_scan)
    )
    mu_total_scan = mu_x * mu_scan
    loss_scan = 1 - np.abs(mu_total_scan) ** 2

    center_list.append(center_scan)
    width_list.append(width_scan)
    loss_list.append(loss_scan)
    mu_list.append(np.abs(mu_total_scan))
    iteration_list.append(n_scan)
    scan_error_list.append(error_scan)

    print(theta_scan, n_scan, center_scan, width_scan, loss_scan)

fit = np.polyfit(theta_list, center_list, 1)
print("center-theta slope:", fit[0])
center_error = np.max(np.abs(np.array(center_list) + 100 * theta_list))
print("center max error:", center_error)

scan_data = np.column_stack(
    [
        theta_list,
        center_list,
        width_list,
        mu_list,
        loss_list,
        iteration_list,
        scan_error_list,
    ]
)
np.savetxt(
    "tilt_scan.csv",
    scan_data,
    delimiter=",",
    header="theta_rad,y_center_lambda,w_y_lambda,mu_2d_abs,round_trip_loss_2d,iteration,error",
    comments="",
)

plot_tilt_center(theta_list, center_list, 100)

# 再把角度放大一些，看模宽和损耗怎么变化
theta_change = np.linspace(0, 0.05, 11)
width_change = []
loss_change = []

for theta_scan in theta_change:
    u_scan, mu_scan, error_scan, n_scan = calc_tilt(theta_scan, x, kernel)
    intensity_scan = np.abs(u_scan) ** 2
    center_scan = np.sum(x * intensity_scan) / np.sum(intensity_scan)
    width_scan = 2 * np.sqrt(
        np.sum((x - center_scan) ** 2 * intensity_scan) / np.sum(intensity_scan)
    )
    mu_total_scan = mu_x * mu_scan
    loss_scan = 1 - np.abs(mu_total_scan) ** 2

    width_change.append(width_scan)
    loss_change.append(loss_scan)

np.savetxt(
    "tilt_change.csv",
    np.column_stack([theta_change, width_change, loss_change]),
    delimiter=",",
    header="theta_rad,w_y_lambda,round_trip_loss_2d",
    comments="",
)

plot_tilt_width(theta_change, width_change)
plot_tilt_loss(theta_change, loss_change)


# 二维结果只选几个代表角度
theta_show = [0, 0.025, 0.05]
tilt_fields = []

for theta2d in theta_show:
    u_tilt2d, error2d, n2d = calc_tilt2d(theta2d, x2, kernel2d)
    tilt_fields.append(u_tilt2d)
    intensity2d = np.abs(u_tilt2d) ** 2
    x_center2d = np.sum(x2[None, :] * intensity2d) / np.sum(intensity2d)
    y_center2d = np.sum(x2[:, None] * intensity2d) / np.sum(intensity2d)
    width_x2d = 2 * np.sqrt(
        np.sum((x2[None, :] - x_center2d) ** 2 * intensity2d) / np.sum(intensity2d)
    )
    width_y2d = 2 * np.sqrt(
        np.sum((x2[:, None] - y_center2d) ** 2 * intensity2d) / np.sum(intensity2d)
    )
    print(
        "2D theta:", theta2d,
        "x center:", x_center2d,
        "y center:", y_center2d,
        "wx:", width_x2d,
        "wy:", width_y2d,
        "iteration:", n2d,
        "error:", error2d,
    )

plot_tilt_2d(x2, tilt_fields, theta_show)

# 报告组合图，直接使用上面的计算结果
plot_report_base(x, amplitude, gaussian, x2, np.abs(u2d[center]))
plot_report_convergence(gamma_list, error_list)
plot_report_base_2d(x2, u2d)
plot_report_tilt(
    x, amplitude_zero, amplitude_tilt, theta,
    theta_list, center_list, 100,
    theta_change, width_change, loss_change,
)
plot_tilt_2d(x2, tilt_fields, theta_show, name="report_tilt_2d")
