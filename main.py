import numpy as np
import matplotlib.pyplot as plt


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


def propagate2d(u, x):
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

    return kernel @ u @ kernel.T


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


plt.figure()
plt.plot(x, amplitude, label="numerical")
plt.plot(x, gaussian, label="Gaussian")
plt.xlabel("x / lambda")
plt.ylabel("|u|")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, np.angle(u))
plt.xlabel("x / lambda")
plt.ylabel("phase")
plt.show()

plt.figure()
plt.plot(range(1, len(gamma_list) + 1), np.abs(gamma_list))
plt.xlabel("iteration")
plt.ylabel("|gamma|")
plt.show()

plt.figure()
plt.plot(range(1, len(gamma_list) + 1), np.angle(gamma_list))
plt.xlabel("iteration")
plt.ylabel("arg(gamma)")
plt.show()

plt.figure()
plt.plot(range(1, len(error_list) + 1), error_list)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("error")
plt.show()


x2 = np.linspace(-10, 10, 51)
u2d = np.ones((51, 51), dtype=complex)

for n in range(len(gamma_list)):
    u2d = propagate2d(u2d, x2)
    u2d = u2d / np.max(np.abs(u2d))

plt.figure()
plt.imshow(
    np.abs(u2d),
    extent=[-10, 10, -10, 10],
    origin="lower",
)
plt.xlabel("x / lambda")
plt.ylabel("y / lambda")
plt.colorbar()
plt.show()


center = len(x2) // 2

plt.figure()
plt.plot(x2, amplitude[::2], label="1D")
plt.plot(x2, np.abs(u2d[center]), label="2D center")
plt.plot(x2, np.exp(-x2**2 / (100 / np.pi)), label="Gaussian")
plt.xlabel("x / lambda")
plt.ylabel("|u|")
plt.legend()
plt.show()

plt.figure()
plt.imshow(
    np.angle(u2d),
    extent=[-10, 10, -10, 10],
    origin="lower",
)
plt.xlabel("x / lambda")
plt.ylabel("y / lambda")
plt.colorbar()
plt.show()


# 镜面倾斜：先测试一个小角度
theta = 0.005
u_tilt, mu_tilt, error, tilt_iteration = calc_tilt(theta, x, kernel)
print("倾斜腔收敛迭代次数:", tilt_iteration)

amplitude_tilt = np.abs(u_tilt)
intensity_tilt = amplitude_tilt ** 2
center_tilt = np.sum(x * intensity_tilt) / np.sum(intensity_tilt)
width_tilt = 2 * np.sqrt(
    np.sum((x - center_tilt) ** 2 * intensity_tilt) / np.sum(intensity_tilt)
)
loss_tilt = 1 - np.abs(mu_tilt) ** 2

print("theta:", theta)
print("width:", width_tilt)
print("|mu|:", np.abs(mu_tilt))
print("round trip loss:", loss_tilt)
print("倾斜场最后残差:", error)
print("光束中心:", center_tilt)
print(
    "倾斜场与基础场 RMS 差:",
    np.sqrt(np.mean((amplitude_tilt - amplitude) ** 2)),
)

plt.figure()
plt.plot(x, amplitude, label="basic")
plt.plot(x, amplitude_tilt, label="round trip")
plt.xlabel("x / lambda")
plt.ylabel("|u|")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, np.angle(u_tilt))
plt.xlabel("x / lambda")
plt.ylabel("phase")
plt.show()


# 连续扫描倾角，先看光束中心随 theta 的变化
theta_list = np.linspace(-0.01, 0.01, 21)
center_list = []
width_list = []
loss_list = []

for theta_scan in theta_list:
    u_scan, mu_scan, error_scan, n_scan = calc_tilt(theta_scan, x, kernel)

    intensity_scan = np.abs(u_scan) ** 2
    center_scan = np.sum(x * intensity_scan) / np.sum(intensity_scan)
    width_scan = 2 * np.sqrt(
        np.sum((x - center_scan) ** 2 * intensity_scan) / np.sum(intensity_scan)
    )
    loss_scan = 1 - np.abs(mu_scan) ** 2

    center_list.append(center_scan)
    width_list.append(width_scan)
    loss_list.append(loss_scan)

    print(theta_scan, n_scan, center_scan, width_scan, loss_scan)

fit = np.polyfit(theta_list, center_list, 1)
print("center-theta slope:", fit[0])

plt.figure()
plt.plot(theta_list, center_list, "o-", label="numerical")
plt.plot(theta_list, -100 * theta_list, "--", label="-L theta")
plt.xlabel("theta / rad")
plt.ylabel("beam center / lambda")
plt.legend()
plt.tight_layout()
plt.savefig("tilt_center.png", dpi=300)
plt.show()
