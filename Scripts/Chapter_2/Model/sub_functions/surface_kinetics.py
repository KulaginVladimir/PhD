import festim as F


class CustomSurfaceKinetics(F.SurfaceKinetics):
    def __init__(
        self,
        k_sb,
        k_bs,
        lambda_IS,
        n_surf,
        n_IS,
        J_vs,
        J_vb,
        surfaces,
        initial_condition,
        **prms
    ):
        super().__init__(
            k_sb,
            k_bs,
            lambda_IS,
            n_surf,
            n_IS,
            J_vs,
            surfaces,
            initial_condition,
            **prms
        )
        self.J_vb = J_vb

    def create_form(self, solute, solute_prev, solute_test_function, T, ds, dt):
        """
        Creates the general form associated with the surface species

        Args:
            solute (fenics.Function or ufl.Indexed): mobile solution for "current"
                timestep
            solute_prev (fenics.Function or ufl.Indexed): mobile solution for
                "previous" timestep
            solute_test_function (fenics.TestFunction or ufl.Indexed): mobile test function
            T (festim.Temperature): the temperature of the simulation
            ds (fenics.Measure): the ds measure of the sim
            dt (festim.Stepsize): the step-size
        """

        lambda_IS = self.lambda_IS
        n_surf = self.n_surf
        n_IS = self.n_IS
        self.form = 0

        for i, surf in enumerate(self.surfaces):

            J_vs = self.J_vs
            if callable(J_vs):
                J_vs = J_vs(T.T, self.solutions[i], solute, **self.prms)
            J_vb = self.J_vb
            if callable(J_vs):
                J_vb = J_vb(T.T, self.solutions[i], solute, **self.prms)
            k_sb = self.k_sb
            if callable(k_sb):
                k_sb = k_sb(T.T, self.solutions[i], solute, **self.prms)
            k_bs = self.k_bs
            if callable(k_bs):
                k_bs = k_bs(T.T, self.solutions[i], solute, **self.prms)

            J_sb = k_sb * self.solutions[i] * (1 - solute / n_IS)
            J_bs = k_bs * solute * (1 - self.solutions[i] / n_surf)

            if dt is not None:
                # Surface concentration form
                self.form += (
                    (self.solutions[i] - self.previous_solutions[i])
                    / dt.value
                    * self.test_functions[i]
                    * ds(surf)
                )
                # Flux to solute species
                self.form += (
                    lambda_IS
                    * (solute - solute_prev)
                    / dt.value
                    * solute_test_function
                    * ds(surf)
                )

            self.form += -(J_vs + J_bs - J_sb) * self.test_functions[i] * ds(surf)
            self.form += (J_bs - J_sb + J_vb) * solute_test_function * ds(surf)

        self.sub_expressions += [expression for expression in self.prms.values()]
