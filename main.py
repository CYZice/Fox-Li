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


x = np.linspace(-25, 25, 101)
u = np.ones(101, dtype=complex)
iteration = 300

for n in range(iteration):
    u2 = propagate(u, x)
    u = u2 / np.max(np.abs(u2))

print(iteration, abs(u[75]))


plt.figure()
plt.plot(x, np.abs(u))
plt.xlabel("x / lambda")
plt.ylabel("|u|")
plt.show()

plt.figure()
plt.plot(x, np.angle(u))
plt.xlabel("x / lambda")
plt.ylabel("phase")
plt.show()
