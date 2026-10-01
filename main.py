import numpy as np
import matplotlib.pyplot as plt


def propagate(u, x):
    u2 = np.zeros(len(x), dtype=complex)
    dx = x[1] - x[0]
    # 长度都用波长作单位
    k = 2 * np.pi

    c = np.sqrt(1j / 100 * np.exp(-1j * k * 100))

    for j in range(len(x)):
        for i in range(len(x)):
            u2[j] += (
                u[i]
                * c
                * np.exp(-1j * k * (x[j] - x[i]) ** 2 / (2 * 100))
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
