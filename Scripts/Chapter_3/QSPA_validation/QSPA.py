import festim as F
import numpy as np
import matplotlib.pyplot as plt
from sub_functions.materials import W
from sub_functions.custom_classes import CustomHeatSolver, CustomHeatSource
import sympy as sp
import fenics as f

alpha_W = 0.4
sigma = 5.67e-8
T0 = 300


def q_heat(t):
    E0 = 0.7e6

    t1 = 1.73891412e-05
    t2 = 6.01227240e-04
    dt1 = 3.31805703e-06
    delta = 3.53224724e-01
    dt2 = 2.55841858e-04
    norm = 0.0007774

    f1 = lambda t: 0.8 / (1 + sp.exp(-(t - 0.5 * t1) / dt1))
    f2 = lambda t: 1 + delta * (t - t1) / (t2 - t1)
    f3 = lambda t: sp.exp(-(t - t2) / dt2)

    return (
        E0
        * sp.Piecewise(
            (f1(t), t <= t1),
            (f1(t1) * f2(t), (t > t1) & (t <= t2)),
            (f1(t1) * f2(t2) * f3(t), True),
        )
        / norm
    )


def q_rad(T, mobile):
    return -alpha_W * sigma * (T**4 - T0**4)


def heat_loss(T):
    return -1.2e3 * W().thermal_cond_function(T) * (T - T0)


def recombination_Og(T, mobile):
    Kr_0 = 3e-25  # T^0.5*m^4/s
    Er = 1.03  # 2.06
    Elim = 0.2
    return -Kr_0 / (T ** (1 / 2)) * f.exp(2 * (Er - Elim) / F.k_B / T) * mobile**2


model = F.Simulation()

vertices = np.concatenate(
    [
        np.linspace(0, 1e-6, 500),
        np.linspace(1e-6, 10e-6, 1000),
        np.linspace(10e-6, 100e-6, 1000),
        np.linspace(100e-6, 2e-3, 500),
    ]
)

model.mesh = F.MeshFromVertices(vertices=vertices)

mat = W()

model.materials = F.Material(
    id=1,
    D_0=mat.D_0,
    E_D=mat.E_diff,
    thermal_cond=mat.thermal_cond_function,
    rho=mat.rho,
    heat_capacity=mat.heat_capacity_function,
    Q=mat.heat_of_transport_function,
)

trap1 = F.Trap(
    k_0=mat.nu_D / mat.n_IS,
    E_k=mat.E_diff,
    p_0=mat.nu_0,
    E_p=1.0,
    density=1e-5 * mat.n_mat,
    materials=model.materials[0],
)

trap2 = F.Trap(
    k_0=mat.nu_D / mat.n_IS,
    E_k=mat.E_diff,
    p_0=mat.nu_0,
    E_p=1.5,
    density=1e-5 * mat.n_mat,
    materials=model.materials[0],
)

model.traps = [trap1, trap2]

model.sources = [
    CustomHeatSource(function=heat_loss, field="T", volume=1),
    F.ImplantationFlux(
        flux=q_heat(F.t) / 7 / 5 / 1.6e-19, imp_depth=2.6e-9, width=1.2e-9, volume=1
    ),
]

model.boundary_conditions = [
    F.DirichletBC(surfaces=[1, 2], value=0, field=0),
    # F.CustomFlux(function=recombination_Og, field=0, surfaces=1),
    F.FluxBC(surfaces=1, value=q_heat(F.t), field="T"),
    F.CustomFlux(surfaces=[1, 2], function=q_rad, field="T"),
]

model.T = CustomHeatSolver(
    transient=True,
    initial_condition=T0,
    absolute_tolerance=1.0,
    relative_tolerance=1e-3,
)

model.dt = F.Stepsize(
    initial_value=1e-6,
    max_stepsize=lambda t: 1e-5 if t < 5e-3 else 25,
    dt_min=1e-8,
    stepsize_change_ratio=1.05,
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    transient=True,
    final_time=1e4,
    maximum_iterations=60,
    soret=True,
)

derived_quantities = F.DerivedQuantities(
    [
        F.TotalVolume(field="retention", volume=1),
        F.TotalSurface(field="T", surface=1),
        F.TotalSurface(field="T", surface=2),
    ],
    filename="./test.csv",
)

TXT = [
    F.TXTExport(
        field="retention", filename="./retention.txt", times=[1e2], write_at_last=True
    )
]

model.exports = [derived_quantities] + TXT


model.initialise()
model.run()


t = np.array(derived_quantities.t)
retention = np.array(derived_quantities[0].data)
T_front = np.array(derived_quantities[1].data)
T_back = np.array(derived_quantities[2].data)

exp = np.loadtxt("./T_back_exp.csv", delimiter=",", skiprows=1)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

axs[0].plot(t, T_front, label="front")
axs[0].plot(t, T_back, label="back")

axs[0].scatter(exp[:, 0], exp[:, 1], s=5, zorder=0)


axs[0].set_xlim(1e-6, 1e3)
axs[0].set_xscale("log")
axs[0].set_xlabel("t, s")
axs[0].set_ylabel("T, K")
axs[0].legend()


"""plt.plot(t, retention / 2e-3 / mat.n_mat)
plt.xlim(0, 0.1)
plt.show()"""

axs[1].plot(TXT[0].data[:, 0] / 1e-6, TXT[0].data[:, -1] / mat.n_mat)
# plt.xscale("log")

axs[1].set_yscale("log")
axs[1].set_ylim(1e-7, 1e-4)
axs[1].set_xlim(-1, 100)
axs[1].set_xlabel("x, um")
axs[1].set_ylabel(r"Retention, m$^{-3}$")

plt.tight_layout()
plt.show()

print(np.trapz(TXT[0].data[:, -1], x=TXT[0].data[:, 0]))
