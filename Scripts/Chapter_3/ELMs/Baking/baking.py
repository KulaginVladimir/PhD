import festim as F
import numpy as np
import sys

sigma = 5.67e-8
alpha_W = 0.4
T0 = 373.0
T_baking = 220 + 273  # 220 C

eta_tr = float(sys.argv[1])
Edt = float(sys.argv[2])
results_folder = f"./results/"

baking_time = 3600 * 24 * 25

w_atom_density = 6.31e28  # atom/m3

my_model = F.Simulation(log_level=40)

r = 1.0188
N = 750
width = 0.1e-9

mesh = [0]
for i in range(1, N):
    mesh.append(mesh[-1] + width * r ** (i - 1))
mesh[-1] = 6e-3
vertices = np.array(mesh)

my_model.mesh = F.MeshFromVertices(vertices=vertices)

tungsten = F.Material(id=1, D_0=1.93e-7 / np.sqrt(2), E_D=0.2)

my_model.materials = tungsten

trap = F.Trap(
    k_0=1.93e-7 / (1.1e-10**2 * 6 * w_atom_density) / np.sqrt(2),
    E_k=0.2,
    p_0=1e13,
    E_p=Edt,
    density=eta_tr * w_atom_density,
    materials=tungsten,
)

my_model.traps = [trap]

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1, 2], value=0, field=0),
]

my_model.initial_conditions = [
    F.InitialCondition(
        field=0,
        value="./results/exp_mobile_re.xdmf",
        label="mobile_concentration",
        time_step=-1,
    ),
    F.InitialCondition(
        field="1",
        value="./results/exp_trapped_re.xdmf",
        label="trap_1_concentration",
        time_step=-1,
    ),
]

my_model.T = F.Temperature(value=T_baking)

my_model.dt = F.Stepsize(
    initial_value=1e-6, stepsize_change_ratio=1.1, max_stepsize=100, dt_min=1e-8
)

absolute_tolerance_c = 1e10
relative_tolerance_c = 1e-8
max_iter = 500

my_model.settings = F.Settings(
    absolute_tolerance=absolute_tolerance_c,
    relative_tolerance=relative_tolerance_c,
    final_time=baking_time,
    maximum_iterations=max_iter,
)

derived_quantities = [
    F.DerivedQuantities(
        [
            F.TotalSurface(field="T", surface=1),
            F.TotalVolume(field="retention", volume=1),
            F.TotalVolume(field="solute", volume=1),
            F.TotalVolume(field="1", volume=1),
        ],
        nb_iterations_between_compute=1,
        show_units=True,
        filename="./results/derived_quantities_baking.csv",
    )
]


my_model.exports = derived_quantities

my_model.initialise()
my_model.run()
