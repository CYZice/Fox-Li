import numpy as np
import matplotlib.pyplot as plt


def propagate(u, x):
    u2 = np.zeros(len(x), dtype=complex)
    dx = x[1] - x[0]
    # 长度都用波长作单位
    k = 2 * np.pi
    distance = 100
    radius = distance  # 对称共焦腔

    c = np.sqrt(1j / distance)

    for j in range(len(x)):
        for i in range(len(x)):
            # 两镜面之间的近轴光程
            rho = (
                distance
                + (x[j] - x[i]) ** 2 / (2 * distance)
                - (x[j] ** 2 + x[i] ** 2) / (2 * radius)
            )
            u2[j] += (
                u[i]
                * c
                * np.exp(-1j * k * rho)
                * dx
            )

    return u2


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
u = np.ones(101, dtype=complex)
iteration = 1000
gamma_list = []
error_list = []
stable = 0

for n in range(iteration):
    u2 = propagate(u, x)
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


# 镜面倾斜：先用 theta=0 检查往返传播
theta = 0
k = 2 * np.pi
u_tilt = np.ones(101, dtype=complex)
stable = 0

for n in range(iteration):
    # 从正常镜传播到倾斜镜
    u_tilt2 = propagate(u_tilt, x)

    # 倾斜镜反射后产生线性相位
    phase = np.exp(1j * 2 * k * theta * x)
    u_tilt2 = u_tilt2 * phase

    # 再传播回正常镜
    u_tilt2 = propagate(u_tilt2, x)

    mu_tilt = np.vdot(u_tilt, u_tilt2) / np.vdot(u_tilt, u_tilt)
    error = np.linalg.norm(u_tilt2 - mu_tilt * u_tilt) / np.linalg.norm(u_tilt2)
    u_tilt = u_tilt2 / np.max(np.abs(u_tilt2))

    if error < 1e-4:
        stable += 1
    else:
        stable = 0

    if stable >= 10:
        print("倾斜腔收敛迭代次数:", n + 1)
        break

amplitude_tilt = np.abs(u_tilt)
print(
    "theta=0 与基础场 RMS 差:",
    np.sqrt(np.mean((amplitude_tilt - amplitude) ** 2)),
)

plt.figure()
plt.plot(x, amplitude, label="basic")
plt.plot(x, amplitude_tilt, label="round trip")
plt.xlabel("x / lambda")
plt.ylabel("|u|")
plt.legend()
plt.show()
