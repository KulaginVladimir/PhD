import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt("./temp_pulse.csv", delimiter=";")
data[:, 0] *= 1e-3


def fit_func(t, t1, t2, dt1, delta, dt2):
    return np.piecewise(
        t,
        [t <= t1, (t > t1) & (t <= t2), t > t2],
        [
            lambda t: 0.8 / (1 + np.exp(-(t - 0.5 * t1) / dt1)),
            lambda t: 0.8
            / (1 + np.exp(-(t1 - 0.5 * t1) / dt1))
            * (1 + delta * (t - t1) / (t2 - t1)),
            lambda t: 0.8
            / (1 + np.exp(-(t1 - 0.5 * t1) / dt1))
            * (1 + delta)
            * np.exp(-(t - t2) / dt2),
        ],
    )


# popt, pcov = curve_fit(fit_func, data[:,0], data[:, 1], p0 = [1e-4, 1e-3, 2e-6, 30, 2e-4]) # for 10 ms
popt, pcov = curve_fit(
    fit_func, data[:, 0], data[:, 1], p0=[1e-4, 1e-3, 2e-6, 0.1, 2e-6]
)  # for 1 ms


print(popt)

fig = plt.figure(figsize=(5, 4))
plt.plot(data[:, 0], data[:, 1], label="Experiment")

t = np.linspace(-1e-3, 2e-3, 200)
plt.plot(t, fit_func(t, *popt), label="Fit")

plt.show()


times = np.linspace(0, 10e-3, 10000)

int = np.trapz(fit_func(times, *popt), x=times)
print(f"Integral = {int:.7f}")
int1 = np.trapz(fit_func(times, *popt) / int, x=times)
print(f"Integral = {int1:.7f}")
