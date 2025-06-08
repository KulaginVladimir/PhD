import festim as F
import numpy as np
import sympy as sp
import properties
import sys

E0 = float(sys.argv[1])
duration = sys.argv[2]
a = float(sys.argv[3])

if duration == "1ms":
    final_time = 1e-1
    max_stepsize = lambda t: 2.5e-5 if t < 2e-3 else 1e-3
    pulse_duration = 978e-6
elif duration == "250us":
    final_time = 1e-2
    max_stepsize = lambda t: 2.5e-6 if t < 5e-4 else 5e-4
    pulse_duration = 220e-6


def pulse(r, t):
    width = 1e-3
    sigma_r = width / 4

    return (
        a
        * E0
        * sp.exp(-(r**2) / 2 / sigma_r**2)
        / 2
        / np.pi
        / sigma_r**2
        / pulse_duration
        * sp.Piecewise((1, t <= pulse_duration), (0, True))
    )


def rad(T, _):
    return -5.670374419e-8 * (T**4 - 300**4)


w_atom_density = 6.31e28  # atom/m3
D0_W = 1.93e-7 / np.sqrt(2)
Ed_W = 0.2


# Define Simulation object
model = F.Simulation(log_level=40)

model.mesh = F.MeshFromXDMF(
    volume_file="../mesh_LID2D/mesh.xdmf",
    boundary_file="../mesh_LID2D/mf.xdmf",
    type="cylindrical",
)

# Define material properties
tungsten = F.Material(
    id=2,
    D_0=D0_W,
    E_D=Ed_W,
    rho=properties.rho_W,
    thermal_cond=properties.thermal_cond_function_W,
    heat_capacity=properties.heat_capacity_function_W,
    Q=properties.heat_of_transport_function_W,
)
copper = F.Material(
    id=1,
    D_0=0,
    E_D=0,
    rho=1,
    heat_capacity=properties.rhoCp_Cu,
    thermal_cond=properties.thermal_cond_Cu,
    Q=0,
)

model.materials = F.Materials([tungsten, copper])

n1 = 0.02894656 * w_atom_density
E_p1 = 1.1081157
n2 = 0.01790 * w_atom_density
E_p2 = 1.27906163
n3 = 0.01967725 * w_atom_density
E_p3 = 1.53602897
n4 = 0.00600191 * w_atom_density
E_p4 = 1.81760592

trap_1 = F.Trap(
    k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
    E_k=Ed_W,
    p_0=1e13,
    E_p=E_p1,
    density=n1,
    materials=tungsten,
)

trap_2 = F.Trap(
    k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
    E_k=Ed_W,
    p_0=1e13,
    E_p=E_p2,
    density=n2,
    materials=tungsten,
)

trap_3 = F.Trap(
    k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
    E_k=Ed_W,
    p_0=1e13,
    E_p=E_p3,
    density=n3,
    materials=tungsten,
)

trap_4 = F.Trap(
    k_0=D0_W / (1.1e-10**2 * 6 * w_atom_density),
    E_k=Ed_W,
    p_0=1e13,
    E_p=E_p4,
    density=n4,
    materials=tungsten,
)

model.traps = [trap_1, trap_2, trap_3, trap_4]

model.initial_conditions = [
    F.InitialCondition(field="1", value=n1 * sp.Piecewise((1, F.y > 6e-3), (0, True))),
    F.InitialCondition(field="2", value=n2 * sp.Piecewise((1, F.y > 6e-3), (0, True))),
    F.InitialCondition(field="3", value=n3 * sp.Piecewise((1, F.y > 6e-3), (0, True))),
    F.InitialCondition(field="4", value=n4 * sp.Piecewise((1, F.y > 6e-3), (0, True))),
]

# Set boundary conditions
model.boundary_conditions = [
    F.FluxBC(surfaces=6, value=pulse(F.x, F.t), field="T"),
    # F.CustomFlux(surfaces=[3, 6], field="T", function=rad),
    F.DirichletBC(surfaces=6, value=0, field=0),
]

# Define the material temperature evolution
model.T = F.HeatTransferProblem(
    initial_condition=300,
    absolute_tolerance=1e-1,
    relative_tolerance=1e-4,
    maximum_iterations=30,
    linear_solver="mumps",
)

# Define the simulation settings
model.dt = F.Stepsize(
    initial_value=5e-7,
    stepsize_change_ratio=1.1,
    max_stepsize=max_stepsize,
    dt_min=1e-8,
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=final_time,
    soret=True,
    maximum_iterations=30,
    traps_element_type="DG",
    linear_solver="mumps",
)

# Define the exports
derived_quantities = F.DerivedQuantities(
    [
        F.SurfaceFluxCylindrical(field="solute", surface=6),
    ],
    show_units=True,
    filename=f"../out_data/results_{duration}_{E0:.3f}J_{a:.2f}/derived_quantities.csv",
    nb_iterations_between_compute=1,
)

XDMF = [
    F.XDMFExport(
        field="retention",
        filename=f"../out_data/results_{duration}_{E0:.3f}J_{a:.2f}/retention.xdmf",
        checkpoint=True,
        label="retention",
        mode="last",
    ),
]

model.exports = [derived_quantities] + XDMF
model.initialise()
model.run()

print("\n DONE")
