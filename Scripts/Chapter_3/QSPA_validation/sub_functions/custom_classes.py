import festim as F
import fenics as f


class CustomHeatSource(F.Source):
    def __init__(self, function, volume, field="all") -> None:
        self.function = function
        super().__init__(value=None, volume=volume, field=field)

    def form(self, T):
        return self.function(T)


class CustomHeatSolver(F.HeatTransferProblem):
    def define_variational_problem(self, materials, mesh, dt=None):
        """Create a variational form for heat transfer problem

        Args:
            materials (F.Materials): the materials.
            mesh (F.Mesh): the mesh.
            dt (F.Stepsize, optional): the stepsize. Only needed if
                self.transient is True. Defaults to None.
        """
        F.festim_print("Defining variational problem heat transfers")

        T, T_n = self.T, self.T_n
        v_T = self.v_T

        self.F = 0
        for mat in materials:
            thermal_cond = mat.thermal_cond
            if callable(thermal_cond):  # if thermal_cond is a function
                thermal_cond = thermal_cond(T)

            subdomains = mat.id  # list of subdomains with this material
            if type(subdomains) is not list:
                subdomains = [subdomains]  # make sure subdomains is a list
            if self.transient:
                cp = mat.heat_capacity
                rho = mat.rho
                if callable(cp):  # if cp or rho are functions, apply T
                    cp = cp(T)
                if callable(rho):
                    rho = rho(T)
                # Transien term
                for vol in subdomains:
                    self.F += rho * cp * (T - T_n) / dt.value * v_T * mesh.dx(vol)
            # Diffusion term
            for vol in subdomains:
                if mesh.type == "cartesian":
                    self.F += f.dot(thermal_cond * f.grad(T), f.grad(v_T)) * mesh.dx(
                        vol
                    )
                elif mesh.type == "cylindrical":
                    r = f.SpatialCoordinate(mesh.mesh)[0]
                    self.F += (
                        r
                        * f.dot(thermal_cond * f.grad(T), f.grad(v_T / r))
                        * mesh.dx(vol)
                    )
                elif mesh.type == "spherical":
                    r = f.SpatialCoordinate(mesh.mesh)[0]
                    self.F += (
                        thermal_cond
                        * r
                        * r
                        * f.dot(f.grad(T), f.grad(v_T / r / r))
                        * mesh.dx(vol)
                    )
        # source term
        for source in self.sources:
            if type(source.volume) is list:
                volumes = source.volume
            else:
                volumes = [source.volume]

            if isinstance(source, CustomHeatSource):
                source.value = source.form(T)

            for volume in volumes:
                self.F += -source.value * v_T * mesh.dx(volume)

            if isinstance(source.value, (f.Expression, f.UserExpression)):
                self.sub_expressions.append(source.value)

        # Boundary conditions
        for bc in self.boundary_conditions:
            if isinstance(bc, F.FluxBC):
                bc.create_form(self.T, solute=None)

                # TODO: maybe that's not necessary
                self.sub_expressions += bc.sub_expressions

                for surf in bc.surfaces:
                    self.F += -bc.form * self.v_T * mesh.ds(surf)
