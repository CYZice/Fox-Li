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


x = np.linspace(-10, 10, 101)
u = np.ones(101, dtype=complex)
iteration = 300
gamma_list = []

for n in range(iteration):
    u2 = propagate(u, x)
    mu = np.mean(u2 / u)
    gamma_list.append(1 / mu)
    u = u2 / np.max(np.abs(u2))

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
plt.plot(range(1, iteration + 1), np.abs(gamma_list))
plt.xlabel("iteration")
plt.ylabel("|gamma|")
plt.show()

plt.figure()
plt.plot(range(1, iteration + 1), np.angle(gamma_list))
plt.xlabel("iteration")
plt.ylabel("arg(gamma)")
plt.show()
