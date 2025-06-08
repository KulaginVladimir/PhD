import festim as F
import fenics as f
import numpy as np
import sympy as sp
import sub_functions.heat_pulses as heat_pulses
from sub_functions.surface_kinetics import CustomSurfaceKinetics
from sub_functions.materials import W, Be

L_tr = 10e-6  # trap layer, m
trap_distribution = sp.Piecewise((1, F.x < L_tr), (0, True))
T0 = 300  # initial temperature, K


class LID_simulation:
    def __init__(
        self,
        E_laser,
        duration,
        material_name,
        E_dt,
        eta_tr,
        phi,
        output_folder,
        output_name,
        is_txt=False,
        soret=True,
        alpha=1,
    ):
        self.E_laser = E_laser
        self.duration = duration
        self.material_name = material_name
        self.E_dt = E_dt
        self.eta_tr = eta_tr
        self.phi = phi
        self.output_folder = output_folder
        self.output_name = output_name
        self.is_txt = is_txt
        self.soret = soret
        self.alpha = alpha

        if duration == "ns":
            self.q_heat = lambda t: E_laser * heat_pulses.gauss_heat(t, 40e-9, 25e-9)

            self.final_time = 1000e-9
            self.step_size = F.Stepsize(
                initial_value=1e-10,
                stepsize_change_ratio=1.05,
                max_stepsize=lambda t: 0.2e-9 if t < 100e-9 else 1e-7,
                dt_min=1e-11,
            )

            self.vertices = np.concatenate(
                [
                    np.linspace(0, 1e-6, 1000),
                    np.linspace(1e-6, 10e-6, 750),
                    np.linspace(10e-6, 100e-6, 500),
                ]
            )

        elif duration == "us":
            self.q_heat = lambda t: E_laser * heat_pulses.trapezoid_heat(
                t, 2.5e-6, 9.5e-6, 0, 0.5e-6, 0.5e-6
            )

            self.final_time = 100e-6
            self.step_size = F.Stepsize(
                initial_value=1e-8,
                stepsize_change_ratio=1.05,
                max_stepsize=lambda t: 2.5e-8 if t < 20e-6 else 5e-7,
                dt_min=1e-11,
            )

            self.vertices = np.concatenate(
                [
                    np.linspace(0, 10e-6, 1000),
                    np.linspace(10e-6, 50e-6, 1000),
                    np.linspace(50e-6, 0.5e-3, 1000),
                ]
            )

        elif duration == "ms":
            self.q_heat = lambda t: E_laser * heat_pulses.trapezoid_heat(
                t, 0.25e-3, 4.95e-3, 0.3, 0.05e-3, 0.05e-3
            )

            self.final_time = 20e-3
            self.step_size = F.Stepsize(
                initial_value=1e-6,
                stepsize_change_ratio=1.05,
                max_stepsize=lambda t: 5e-6 if t < 10e-3 else 5e-5,
                dt_min=1e-11,
            )

            self.vertices = np.concatenate(
                [
                    np.linspace(0, 10e-6, 1000),
                    np.linspace(10e-6, 1e-4, 1000),
                    np.linspace(1e-4, 20e-3, 500),
                ]
            )

        if material_name == "Be":
            self.material = Be()
        elif material_name == "W":
            self.material = W()
        else:
            raise ValueError(f"Material {material_name} is not defined")

    def exp_type(self, x):
        if isinstance(x, np.ndarray):
            return np.exp(x)
        else:
            return f.exp(x)

    def check_Qc(self, Qc, cs):
        if callable(Qc):
            return Qc(cs)
        else:
            return Qc

    def k_sb(self, T, cs, cm):
        mat = self.material
        Q_c = self.check_Qc(mat.Q_c, cs)

        return mat.nu_0 * self.exp_type(-(mat.E_s - Q_c) / F.k_B / T)

    def k_bs(self, T, cs, cm):
        mat = self.material

        return (
            mat.nu_D * mat.lambda_abs * self.exp_type(-(mat.E_s - mat.Q_s) / F.k_B / T)
        )

    def J_a(self, T, cs):
        mat = self.material
        Q_c = self.check_Qc(mat.Q_c, cs)
        k_a = mat.nu_0 * self.exp_type(-(mat.E_d - Q_c) / F.k_B / T)

        return cs * k_a

    def J_m_s(self, T, cs):
        mat = self.material
        Q_c = self.check_Qc(mat.Q_c, cs)
        k_m_s = mat.nu_0 / mat.n_surf * self.exp_type(-2 * (mat.E_c - Q_c) / F.k_B / T)

        return 2 * cs**2 * k_m_s

    def J_m_sb(self, T, cs, cm):
        mat = self.material
        Q_c = self.check_Qc(mat.Q_c, cs)
        k_m_sb = (
            mat.nu_D
            / mat.n_IS
            * self.exp_type(-(mat.E_c - Q_c + mat.E_s - mat.Q_s) / F.k_B / T)
        )

        return cs * cm * k_m_sb

    def J_m_b(self, T, cm):
        mat = self.material

        k_m_b = (
            mat.nu_D
            / mat.n_IS
            * mat.lambda_abs
            * self.exp_type(-2 * (mat.E_s - mat.Q_s) / F.k_B / T)
        )

        return 2 * cm**2 * k_m_b

    def J_vs(self, T, cs, cm):
        return -self.J_m_s(T, cs) - self.J_a(
            T, cs
        )  # - self.J_m_s(T, cs) - self.J_m_sb(T, cs, cm)

    def J_vb(self, T, cs, cm):
        return 0  # -self.J_m_b(T, cm) - self.J_m_sb(T, cs, cm)

    def run(self):
        LID_model = F.Simulation(log_level=40)

        LID_model.mesh = F.MeshFromVertices(self.vertices)

        mod_thermal_cond = lambda T: self.material.thermal_cond_function(T) * self.alpha

        material = F.Material(
            id=1,
            D_0=self.material.D_0,
            E_D=self.material.E_diff,
            thermal_cond=mod_thermal_cond,
            heat_capacity=self.material.heat_capacity_function,
            rho=self.material.rho,
            Q=self.material.heat_of_transport_function,
        )

        LID_model.materials = [material]

        LID_model.traps = [
            F.Trap(
                k_0=self.material.nu_D / self.material.n_IS,
                E_k=self.material.E_diff,
                p_0=self.material.nu_0,
                E_p=self.E_dt,
                density=self.eta_tr * self.material.n_mat * trap_distribution,
                materials=material,
            )
        ]

        LID_model.T = F.HeatTransferProblem(
            transient=True,
            initial_condition=T0,
            absolute_tolerance=1.0,
            relative_tolerance=1e-3,
        )

        LID_model.boundary_conditions = [
            F.DirichletBC(field=0, value=0, surfaces=2),
            F.DirichletBC(field="T", value=T0, surfaces=2),
            F.FluxBC(field="T", value=self.q_heat(F.t), surfaces=1),
            CustomSurfaceKinetics(
                k_bs=self.k_bs,
                k_sb=self.k_sb,
                lambda_IS=self.material.lambda_IS,
                n_surf=self.material.n_surf,
                n_IS=self.material.n_IS,
                J_vs=self.J_vs,
                J_vb=self.J_vb,
                initial_condition=0,
                surfaces=1,
            ),
        ]

        LID_model.initial_conditions = [
            F.InitialCondition(
                field="1",
                value=self.phi * self.eta_tr * self.material.n_mat * trap_distribution,
            )
        ]

        LID_model.dt = self.step_size

        LID_model.settings = F.Settings(
            absolute_tolerance=1e16,
            relative_tolerance=1e-6,
            final_time=self.final_time,
            soret=self.soret,
            traps_element_type="DG",
        )

        derived_quantities = F.DerivedQuantities(
            [
                F.AdsorbedHydrogen(surface=1),
                F.HydrogenFlux(surface=1),
                F.TotalSurface(field="T", surface=1),
                F.PointValue(field=0, x=0),
            ],
            show_units=True,
            filename=f"{self.output_folder}/data_{self.output_name}.csv",
        )

        TXT = [
            F.TXTExport(
                field="retention",
                filter=True,
                write_at_last=True,
                filename=f"{self.output_folder}/ret_{self.output_name}.txt",
            ),
            F.TXTExport(
                field="1",
                filter=True,
                write_at_last=True,
                filename=f"{self.output_folder}/trapped_{self.output_name}.txt",
            ),
            F.TXTExport(
                field="solute",
                filter=True,
                write_at_last=True,
                filename=f"{self.output_folder}/solute_{self.output_name}.txt",
            ),
        ]

        LID_model.exports = [derived_quantities]

        if self.is_txt:
            LID_model.exports += TXT

        LID_model.initialise()
        LID_model.run()

        return derived_quantities
