from pathlib import Path

import numpy as np

from base import make_kernel, make_kernel2d, propagate, propagate2d
from plot_results import plot_report_tilt, plot_tilt_2d


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "figures" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


# 进阶任务：单镜倾斜，扫描倾角并输出报告图
def run_tilt():
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)

    # 镜面倾斜：一维里的 x 在这里对应物理上的 y 方向
    theta = 0.005
    # x 方向不受倾斜，二维往返本征值由两个方向相乘
    u_zero, mu_x, _, _ = calc_tilt(0, x, kernel)
    u_tilt, mu_tilt, error, tilt_iteration = calc_tilt(theta, x, kernel)
    print("倾斜腔收敛迭代次数:", tilt_iteration)

    amplitude_tilt = np.abs(u_tilt)
    amplitude_zero = np.abs(u_zero)
    intensity_tilt = amplitude_tilt**2
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
        DATA_DIR / "tilt_scan.csv",
        scan_data,
        delimiter=",",
        header="theta_rad,y_center_lambda,w_y_lambda,mu_2d_abs,round_trip_loss_2d,iteration,error",
        comments="",
    )

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
        if theta_scan in (0, 0.025, 0.05):
            print(
                "代表倾角:",
                theta_scan,
                "|mu_2D|:",
                np.abs(mu_total_scan),
                "往返功率损耗:",
                loss_scan,
            )

    np.savetxt(
        DATA_DIR / "tilt_change.csv",
        np.column_stack([theta_change, width_change, loss_change]),
        delimiter=",",
        header="theta_rad,w_y_lambda,round_trip_loss_2d",
        comments="",
    )

    # 二维结果只选几个代表角度
    x2 = np.linspace(-10, 10, 51)
    kernel2d = make_kernel2d(x2)
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
            "2D theta:",
            theta2d,
            "x center:",
            x_center2d,
            "y center:",
            y_center2d,
            "wx:",
            width_x2d,
            "wy:",
            width_y2d,
            "iteration:",
            n2d,
            "error:",
            error2d,
        )

    plot_tilt_2d(x2, tilt_fields, theta_show, name="report_tilt_2d")

    # 报告组合图，直接使用上面的计算结果
    plot_report_tilt(
        x,
        amplitude_zero,
        amplitude_tilt,
        theta,
        theta_list,
        center_list,
        100,
        theta_change,
        width_change,
        loss_change,
    )
