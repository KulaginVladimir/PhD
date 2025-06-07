#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --array=1-21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=300:00:00

export DIJITSO_CACHE_DIR=./cache
export OMP_NUM_THREADS=1 
config=./config_stat.txt

time=1000
f_ELM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
q_stat=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
E_ELM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
Edt=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
eta_tr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)

filename=STAT_${SLURM_ARRAY_JOB_ID}_${time}s_${q_stat}MW_${Edt}eV_${eta_tr}
dir=$PWD
mkdir ${filename}

cat > ./${filename}/${filename}.py << EOL
import festim as F
import fenics as f
import numpy as np
import sympy as sp
from scipy import special

f_ELM = ${f_ELM}

stat_params = {
    1e6:  [51, 6.85e-1, 4.84e-1, 2.64e-9, 1.42e-9],
    5e6:  [75, 6.66e-1, 4.61e-1, 3.27e-9, 1.76e-9],
    10e6: [97, 6.54e-1, 4.47e-1, 3.76e-9, 2.03e-9]
}

q_stat = ${q_stat}
E_stat = stat_params[q_stat][0]
q = 1.6e-19
r_stat = stat_params[q_stat][1]
r_en_stat = stat_params[q_stat][2]
Te = E_stat / 4.85
htf = (4.85 * (1-r_en_stat) + 2.15) * Te + 13.6
Gamma_stat = q_stat / htf / q
X_stat = stat_params[q_stat][3]
sigma_stat = stat_params[q_stat][4]

sigma = 5.67e-8
alpha_W = 0.4
T0 = 373.0
w_atom_density = 6.31e28  # atom/m3
eta_tr = ${eta_tr}
Edt = ${Edt}

def thermal_cond_function(T):
    return 149.441-45.466e-3*T+13.193e-6*T**2-1.484e-9*T**3+3.866e6/(T+1e-4)**2
def heat_capacity_function(T):
    return (21.868372+8.068661e-3*T-3.756196e-6*T**2+1.075862e-9*T**3+1.406637e4/(T+1.)**2) / 183.84e-3   
def heat_of_transport(T):
    return -0.0045 * F.R * T**2

def q_tot(T, mobile):
    return q_stat-alpha_W * sigma * (T ** 4 - T0 ** 4)
    
def cooling(T, mobile):
    if q_stat == 10e6:
        params = {# "freq": q_eqv * (a + b*x+ c*exp(-d*x))
            0:   [1.296e-1, 6.510e-3, 1.254e-1, np.log(0.87984)],
            10:  [1.245e-1, 4.510e-3, 1.176e-1, np.log(0.9187)],
            20:  [1.227e-1, 4.020e-3, 1.178e-1, np.log(0.92199)],
            30:  [1.215e-1, 3.720e-3, 1.170e-1, np.log(0.9265)],
            40:  [1.207e-1, 3.490e-3, 1.163e-1, np.log(0.93021)],
            50:  [1.200e-1, 3.320e-3, 1.157e-1, np.log(0.93327)],
            60:  [1.195e-1, 3.180e-3, 1.151e-1, np.log(0.93582)],
            70:  [1.190e-1, 3.060e-3, 1.147e-1, np.log(0.93807)],
            80:  [1.186e-1, 2.950e-3, 1.143e-1, np.log(0.94002)],
            90:  [1.182e-1, 2.860e-3, 1.139e-1, np.log(0.94175)],
            100: [1.179e-1, 2.780e-3, 1.136e-1, np.log(0.94331)]}
    elif q_stat == 5e6:
        params = {# "freq": q_eqv * (a + b*x+ c*exp(-d*x))
            0:   [1.388e-1, 1.287e-2, 1.336e-1, np.log(7.922e-1)],
            10:  [1.305e-1, 6.830e-3, 1.196e-1, np.log(8.852e-1)],
            20:  [1.281e-1, 5.770e-3, 1.215e-1, np.log(8.949e-1)],
            30:  [1.264e-1, 5.160e-3, 1.206e-1, np.log(9.033e-1)],
            40:  [1.252e-1, 4.740e-3, 1.197e-1, np.log(9.097e-1)],
            50:  [1.242e-1, 4.430e-3, 1.188e-1, np.log(9.147e-1)],
            60:  [1.234e-1, 4.180e-3, 1.182e-1, np.log(9.188e-1)],
            70:  [1.227e-1, 3.970e-3, 1.175e-1, np.log(9.223e-1)],
            80:  [1.221e-1, 3.800e-3, 1.170e-1, np.log(9.253e-1)],
            90:  [1.215e-1, 3.650e-3, 1.165e-1, np.log(9.278e-1)],
            100: [1.211e-1, 3.520e-3, 1.161e-1, np.log(9.300e-1)]}
    elif q_stat == 1e6:
        params = {# "freq": q_eqv * (a + b*x+ c*exp(-d*x))
            0:   [1.506e-1, 6.331e-2, 1.449e-1, np.log(3.523e-1)],
            10:  [1.352e-1, 1.171e-2, 1.219e-1, np.log(7.864e-1)],
            20:  [1.340e-1, 8.860e-3, 1.260e-1, np.log(8.476e-1)],
            30:  [1.318e-1, 7.500e-3, 1.252e-1, np.log(8.672e-1)],
            40:  [1.301e-1, 6.640e-3, 1.241e-1, np.log(8.799e-1)],
            50:  [1.287e-1, 6.040e-3, 1.230e-1, np.log(8.889e-1)],
            60:  [1.276e-1, 5.580e-3, 1.221e-1, np.log(8.960e-1)],
            70:  [1.266e-1, 5.220e-3, 1.213e-1, np.log(9.016e-1)],
            80:  [1.258e-1, 4.930e-3, 1.205e-1, np.log(9.064e-1)],
            90:  [1.250e-1, 4.680e-3, 1.198e-1, np.log(9.104e-1)],
            100: [1.243e-1, 4.470e-3, 1.192e-1, np.log(9.138e-1)]}
    return -(q_stat) * (params[f_ELM][0] + params[f_ELM][1] * (T-T0) - params[f_ELM][2] * f.exp(params[f_ELM][3]*(T-T0)))

def norm_flux(X, sigma):
    L = 6e-3
    return 2 / (special.erf((L-X)/np.sqrt(2)/sigma) + special.erf((X)/np.sqrt(2)/sigma))

my_model = F.Simulation(log_level = 40)

r = 1.0188
N = 750
width = 0.1e-9

mesh = [0]
for i in range(1, N):
    mesh.append(mesh[-1] + width*r**(i-1))
mesh[-1] = 6e-3   
vertices = np.array(mesh)

my_model.mesh = F.MeshFromVertices(vertices=vertices)

tungsten = F.Material(id=1, 
                    D_0=1.93e-7/np.sqrt(2), 
                    E_D=0.2,
                    thermal_cond=thermal_cond_function, 
                    heat_capacity=heat_capacity_function,
                    rho=19250,
                    heat_transport=heat_of_transport)

my_model.materials = tungsten

trap = F.Trap(
        k_0=1.93e-7/(1.1e-10**2*6*w_atom_density)/np.sqrt(2),
        E_k=0.2,
        p_0=1e13,
        E_p=Edt,
        density=eta_tr*w_atom_density,
        materials=tungsten
    )

my_model.traps = [trap]

stat_source = F.ImplantationFlux(
    flux=Gamma_stat*(1-r_stat)*norm_flux(X_stat, sigma_stat),  # H/m2/s
    imp_depth=X_stat,  # m
    width=sigma_stat,  # m
    volume=1
)


my_model.sources = [stat_source]

my_model.boundary_conditions = [
    F.CustomFlux(function=q_tot, field="T", surfaces=1),
    F.CustomFlux(function=cooling, field="T", surfaces=2),
    F.DirichletBC(surfaces=[1,2], value=0, field=0)
]

absolute_tolerance_T = 1.0
relative_tolerance_T = 1e-5
absolute_tolerance_c = 1e10
relative_tolerance_c = 1e-8
max_iter = 100

my_model.T = F.HeatTransferProblem(
    absolute_tolerance=absolute_tolerance_T,
    relative_tolerance=relative_tolerance_T,
    initial_value=373,
    maximum_iterations=max_iter,
)

my_model.dt = F.Stepsize(
    initial_value=1e-7,
    stepsize_change_ratio=1.025,
    t_stop = 1,
    stepsize_stop_max = 1,
    dt_min=1e-7
)

my_model.settings = F.Settings(
    absolute_tolerance=absolute_tolerance_c,
    relative_tolerance=relative_tolerance_c,
    final_time=${time},
    soret=True,
    maximum_iterations=max_iter,
)

n_exp = "last"
results_folder = "$PWD/${filename}/results/"
XDMF = [F.XDMFExport(
        field="solute",
        filename=results_folder + "/mobile.xdmf",
        checkpoint=False,  # needed in 1D
        mode=n_exp,
        ), 
        F.XDMFExport(
        field="1",
        filename=results_folder + "/trapped.xdmf",
        checkpoint=False,  # needed in 1D
        mode=n_exp,
        ),
        F.XDMFExport(
        field="retention",
        filename=results_folder + "/ret.xdmf",
        checkpoint=False,  # needed in 1D
        mode=n_exp,
        ),
        F.XDMFExport(
        field="T",
        filename=results_folder + "/T.xdmf",
        checkpoint=False,  # needed in 1D
        mode=n_exp,
        )]

#######################################RECYCLING COEFFICIENT
class CustomDerivedQuantities(F.DerivedQuantities):
    def compute(self, t):
        # TODO need to support for soret flag in surface flux
        row = [t]
        for quantity in self.derived_quantities:
            if isinstance(quantity, (F.MaximumVolume, F.MinimumVolume)):
                value = quantity.compute(self.volume_markers)
            elif isinstance(quantity, RecyclingCoefficient):
                value = quantity.compute(t)
            else:
                value = quantity.compute()
            quantity.data.append(value)
            quantity.t.append(t)
            row.append(value)
        self.data.append(row)
        self.t.append(t)

class RecyclingCoefficient(F.HydrogenFlux):
    def __init__(self, surface) -> None:
        super().__init__(surface)
        self.title = "Recycling coefficient surface {}".format(surface)

    def compute(self, time):
        des_flux = np.abs(super().compute())
        impl_flux = 0
        t = sp.Symbol("t")
        for source in my_model.sources:
            if isinstance(source.flux, sp.Expr):
                impl_flux += source.flux.subs(t,time)
            else:
                impl_flux += source.flux
        return des_flux / impl_flux
#######################################RECYCLING COEFFICIENT

derived_quantities = [F.DerivedQuantities([F.HydrogenFlux(surface=1), 
                                          F.TotalSurface(field='T', surface=1), 
                                          F.TotalVolume(field='retention', volume=1),
                                          F.TotalVolume(field='solute', volume=1),
                                          F.TotalVolume(field="1", volume=1)],
                                          #RecyclingCoefficient(surface=1)],
                                          nb_iterations_between_compute = 10,
                                          filename=results_folder + "derived_quantities.csv")]

my_model.exports = derived_quantities + XDMF 

my_model.initialise()
my_model.run()
EOL


cd ./${filename}
mkdir ./results
mpirun -np 1 python3 ./${filename}.py
