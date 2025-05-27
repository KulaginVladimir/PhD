import fenics as f
import numpy as np
from festim import k_B


class Material:
    def __init__(
        self,
        n_mat: float,
        rho: float,
        n_surf: float,
        n_IS: float,
        lambda_IS: float,
        lambda_abs: float,
        D_0: float,
        E_diff: float,
    ):
        """Base class for material properties

        :param n_mat: concentration of host atoms, m^-3
        :type n_mat: float
        :param rho: material density, kg m^-3
        :type rho: float
        :param n_surf: concentration of adsorption sites, m^-2
        :type n_surf: float
        :param n_IS: _description_
        :type n_IS: float
        :param lambda_IS: concentration of interstitial sites, m^-3
        :type lambda_IS: float
        :param lambda_abs: distance between interstitial sites, m^-3
        :type lambda_abs: float
        :param D_0: diffusion pre-factor, m^2 s^-1
        :type D_0: float
        :param E_diff: diffusion activation energy, eV
        :type E_diff: float
        """

        self.n_mat = n_mat
        self.rho = rho
        self.n_surf = n_surf
        self.n_IS = n_IS
        self.lambda_IS = lambda_IS
        self.lambda_abs = lambda_abs
        self.D_0 = D_0
        self.E_diff = E_diff

        self.E_d = 2.26  # molecule dissociation energy per atom, eV
        self.nu_0 = 1e13  # Debye frequency, s^-1
        self.nu_D = D_0 / lambda_IS**2  # attempt frequency in a diffusion process, s^-1


class W(Material):
    def __init__(self):
        n_mat = 6.31e28  # W bulk concentration, m^-3
        rho = 19250.0  # Tungsten density
        n_surf = 1.26 * n_mat ** (2 / 3)  # concentration of adsorption sites, m^-2
        n_IS = 6 * n_mat  # concentration of IS, m^-3
        lambda_IS = 110e-12  # distance between IS sites, m
        lambda_abs = (
            n_surf / n_IS
        )  # distance between adsorption and interstitial sites, m
        D_0 = 1.93e-7 / np.sqrt(2)  # diffusivity pre-factor, m^2 s^-1
        E_diff = 0.2  # diffusion activation energy, eV

        super().__init__(n_mat, rho, n_surf, n_IS, lambda_IS, lambda_abs, D_0, E_diff)

        self.Q_s = 1.04  # Solution energy, eV
        self.E_c = 0  # Chemisorption activation energy, eV
        self.E_s = self.Q_s + E_diff  # Solution activation energy, eV
        self.M = 183.84e-3

    def thermal_cond_function(self, T):
        # Temperature dependence of the W thermal conductivity in W/m/K
        if not isinstance(T, float):
            if any(T.vector().get_local() == 0):
                return 149.441 - 45.466e-3 * T + 13.193e-6 * T**2 - 1.484e-9 * T**3
        else:
            return (
                149.441
                - 45.466e-3 * T
                + 13.193e-6 * T**2
                - 1.484e-9 * T**3
                + 3.866e6 / (T + 1.0) ** 2
            )

    def heat_capacity_function(self, T):
        # Temperature dependence of the W volumetric heat capacity in J/kg
        if not isinstance(T, float):
            if any(T.vector().get_local() == 0):
                return (
                    21.868372
                    + 8.068661e-3 * T
                    - 3.756196e-6 * T**2
                    + 1.075862e-9 * T**3
                ) / self.M
        else:
            Cp_below_3080 = (
                21.868372
                + 8.068661e-3 * T
                - 3.756196e-6 * T**2
                + 1.075862e-9 * T**3
                + 1.406637e4 / (T + 1.0) ** 2
            ) / self.M
            Cp_above_3080 = (2.022 + 1.315e-2 * T) / self.M
            return f.conditional(T <= 3080, Cp_below_3080, Cp_above_3080)

    def heat_of_transport_function(self, T):
        # Temperature dependence of the Soret coefficient in eV
        return -0.0045 * k_B * T**2

    def Q_c(self, cs):
        theta = cs / self.n_surf

        if isinstance(cs, np.ndarray):
            return -(0.071 + 0.647 / (1 + np.exp((theta - 1) / 0.05)))
        else:
            return -(0.071 + 0.647 / (1 + f.exp((theta - 1) / 0.05)))


class Be(Material):
    def __init__(self):
        n_mat = 1.23e29  # Be bulk concentration, m^-3
        rho = 1850  # Be density, kg m^-3
        n_surf = n_mat ** (2 / 3)  # concentration of adsorption sites, m^-2
        n_IS = 6 * n_mat  # concentration of IS, m^-3
        lambda_IS = 3.154e-10  # distance between IS sites, m
        lambda_abs = (
            n_surf / n_IS
        )  # distance between adsorption and interstitial sites, m
        D_0 = 1.4e-6 / np.sqrt(2)  # diffusivity pre-factor, m^2 s^-1
        E_diff = 0.38  # diffusion activation energy, eV

        super().__init__(n_mat, rho, n_surf, n_IS, lambda_IS, lambda_abs, D_0, E_diff)

        self.Q_s = 1.58  # Solution energy, eV
        self.E_c = 0  # Chemisorption activation energy, eV
        self.E_s = self.Q_s + E_diff  # Solution activation energy, eV
        self.Q_c = -0.5  # heat of chemisorption, eV
        self.Tm = 1560  # melting temperature, K
        self.Tt = 1543  # alpha -> beta transition temperature, K
        self.M = 9.012182e-3

    def thermal_cond_function(self, T):
        # Temperature dependence of the Be thermal conductivity in W/m/K
        if not isinstance(T, float):
            if any(T.vector().get_local() == 0):
                return 148.8912 - 76.3780e-3 * T + 12.0174e-6 * T**2
        else:

            kappa_bm = 148.8912 - 76.3780e-3 * T + 12.0174e-6 * T**2 + 6.5407e6 / T**2

            kappa_am = 84.59 + 54.22e-3 * (T - self.Tm)

            kappa = f.conditional(T < self.Tm, kappa_bm, kappa_am)
            return kappa

    def heat_capacity_function(self, T):
        # Temperature dependence of the Be volumetric heat capacity in J/kg
        if not isinstance(T, float):
            if any(T.vector().get_local() == 0):
                return (21.205 + 5.694e-3 * T + 0.962e-6 * T**2) / self.M
        else:

            Cp_bt = (
                21.205 + 5.694e-3 * T + 0.962e-6 * T**2 - 0.5874e6 / T**2
            )  # before transition
            Cp_at = 30  # before melting
            Cp_am = 25.4345 + 2.150e-3 * T  # after melting

            Cp1 = f.conditional(T < self.Tm, Cp_at, Cp_am)
            Cp = f.conditional(T < self.Tt, Cp_bt, Cp1)

            return Cp / self.M

    def heat_of_transport_function(self, T):
        # Temperature dependence of the Soret coefficient in eV
        return 0
