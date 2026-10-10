from pathlib import Path

import numpy as np

from base import make_kernel, propagate, propagate2d, calc_stats
from plot_results import plot_report_tilt, plot_tilt_2d


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "figures" / "data"


def calc_tilt(theta, x, kernel, iteration=3000):
    k = 2 * np.pi
    u = np.ones(len(x), dtype=complex)
    stable = 0
    phase = np.exp(1j * 2 * k * theta * x)

    for n in range(iteration):
        # 从正常镜传播到倾斜镜
        u2 = propagate(u, kernel)

        # 倾斜镜反射后产生线性相位
        u2 = u2 * phase

        # 再传播回正常镜
        u2 = propagate(u2, kernel)

        mu = np.vdot(u, u2) / np.vdot(u, u)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        u = u2 / np.max(np.abs(u2))

        # 两方向合成后仍需满足二维 1e-4 残差
        if error < 1e-4 / np.sqrt(2):
            stable += 1
        else:
            stable = 0

        if stable >= 10:
            break

    u2 = propagate(propagate(u, kernel) * phase, kernel)
    mu = np.vdot(u, u2) / np.vdot(u, u)
    error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
    return u, mu, error, n + 1, stable >= 10


def calc_tilt2d(theta, x, kernel, iteration=3000):
    k = 2 * np.pi
    u = np.ones((len(x), len(x)), dtype=complex)
    phase = np.exp(1j * 2 * k * theta * x)[:, None]
    stable = 0
    errors = []
    mu_list = []

    for n in range(iteration):
        u2 = propagate2d(u, kernel)
        u2 = phase * u2
        u2 = propagate2d(u2, kernel)

        mu = np.vdot(u, u2) / np.vdot(u, u)
        error = np.linalg.norm(u2 - mu * u) / np.linalg.norm(u2)
        errors.append(error)
        mu_list.append(mu)
        u = u2 / np.max(np.abs(u2))

        if error < 1e-4:
            stable += 1
        else:
            stable = 0

        if stable >= 10:
            break

    stats = calc_stats(x, u, kernel, phase)
    return {"field": u, "iteration": n + 1,
            "converged": stable >= 10 and stats["residual"] < 1e-4,
            "residual_history": np.array(errors),
            "mu_history": np.array(mu_list), **stats}


# 进阶任务：单镜倾斜，扫描倾角并输出报告图
def run_tilt():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-10, 10, 101)
    kernel = make_kernel(x)

    # 镜面倾斜：一维里的 x 在这里对应物理上的 y 方向
    theta = 0.005
    # x 方向不受倾斜，二维往返本征值由两个方向相乘
    u_zero, mu_x, _, _, zero_converged = calc_tilt(0, x, kernel)
    u_tilt, mu_tilt, error, tilt_iteration, converged = calc_tilt(theta, x, kernel)
    assert zero_converged and converged
    print("倾斜腔收敛迭代次数:", tilt_iteration)

    amplitude_tilt = np.abs(u_tilt)
    amplitude_zero = np.abs(u_zero)
    intensity_tilt = amplitude_tilt**2
    center_tilt = np.sum(x * intensity_tilt) / np.sum(intensity_tilt)
    width_tilt = 2 * np.sqrt(
        np.sum((x - center_tilt) ** 2 * intensity_tilt) / np.sum(intensity_tilt)
    )
    stats = calc_stats(x, np.outer(u_tilt, u_zero), kernel,
                       np.exp(1j * 4 * np.pi * theta * x)[:, None])
    assert stats["residual"] < 1e-4
    mu_total = stats["mu"]
    loss_tilt = stats["loss"]

    print("theta:", theta)
    print("width:", width_tilt)
    print("|mu_2D|:", np.abs(mu_total))
    print("2D round trip loss:", loss_tilt)
    print("倾斜场二维最终残差:", stats["residual"])
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
        u_scan, mu_scan, error_scan, n_scan, converged = calc_tilt(theta_scan, x, kernel)
        stats = calc_stats(x, np.outer(u_scan, u_zero), kernel,
                           np.exp(1j * 4 * np.pi * theta_scan * x)[:, None])
        assert converged and stats["residual"] < 1e-4

        intensity_scan = np.abs(u_scan) ** 2
        center_scan = np.sum(x * intensity_scan) / np.sum(intensity_scan)
        width_scan = 2 * np.sqrt(
            np.sum((x - center_scan) ** 2 * intensity_scan) / np.sum(intensity_scan)
        )
        mu_total_scan = stats["mu"]
        loss_scan = stats["loss"]

        center_list.append(center_scan)
        width_list.append(width_scan)
        loss_list.append(loss_scan)
        mu_list.append(np.abs(mu_total_scan))
        iteration_list.append(n_scan)
        scan_error_list.append(stats["residual"])

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
        header="theta_rad,y_center_lambda,w_y_lambda,mu_2d_abs,round_trip_loss_2d,iteration,residual_2d",
        comments="",
    )

    # 再把角度放大一些，看模宽和损耗怎么变化
    theta_change = np.linspace(0, 0.05, 11)
    width_change = []
    loss_change = []
    change_rows = []

    for theta_scan in theta_change:
        u_scan, mu_scan, error_scan, n_scan, converged = calc_tilt(theta_scan, x, kernel)
        stats = calc_stats(x, np.outer(u_scan, u_zero), kernel,
                           np.exp(1j * 4 * np.pi * theta_scan * x)[:, None])
        assert converged and stats["residual"] < 1e-4
        intensity_scan = np.abs(u_scan) ** 2
        center_scan = np.sum(x * intensity_scan) / np.sum(intensity_scan)
        width_scan = 2 * np.sqrt(
            np.sum((x - center_scan) ** 2 * intensity_scan) / np.sum(intensity_scan)
        )
        mu_total_scan = stats["mu"]
        loss_scan = stats["loss"]

        width_change.append(width_scan)
        loss_change.append(loss_scan)
        change_rows.append([theta_scan, width_scan, loss_scan, center_scan,
                            stats["wx"], abs(stats["mu"]), stats["residual"],
                            n_scan, int(converged)])
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
        change_rows,
        delimiter=",",
        header="theta_rad,w_y_lambda,round_trip_loss_2d,y_center_lambda,w_x_lambda,mu_2d_abs,residual_2d,iteration_1d,converged",
        comments="",
    )

    # 二维结果只选几个代表角度
    x2 = np.linspace(-10, 10, 101)
    kernel2d = make_kernel(x2)
    theta_show = [0, 0.025, 0.05]
    tilt_fields = []

    for theta2d in [0, 0.005, 0.025, 0.05]:
        result = calc_tilt2d(theta2d, x2, kernel2d)
        assert result["converged"]
        np.savez_compressed(DATA_DIR / f"tilt_2d_{theta2d:.3f}.npz",
                            x=x2, theta=theta2d, **result)
        u_tilt2d = result["field"]
        error2d, n2d = result["residual"], result["iteration"]
        if theta2d in theta_show:
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
