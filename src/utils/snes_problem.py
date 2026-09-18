import dolfin as df


class SNESProblem:
    def __init__(self, F, u, bcs):
        # PETSc SNES callback adapter: exposes F(u)=0 and J(u)=dF/du for a DOLFIN residual form.
        V = u.function_space()
        du = df.TrialFunction(V)
        self.L = F
        self.a = df.derivative(F, u, du)
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, F):
        x = df.PETScVector(x)
        F = df.PETScVector(F)

        # SNES provides `x`; copy it into the DOLFIN Function to assemble on the distributed mesh.
        x.vec().copy(self.u.vector().vec())
        self.u.vector().apply("")

        df.assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)
            # Apply BCs both to (F,x) and the Function vector to keep ghosted entries consistent.
            bc.apply(F, self.u.vector())

    def J(self, snes, x, J, P):
        J = df.PETScMatrix(J)
        # Jacobian assembly assumes `self.u` matches the SNES iterate.
        x.copy(self.u.vector().vec())
        self.u.vector().apply("")

        df.assemble(self.a, tensor=J)
        for bc in self.bcs:
            bc.apply(J)
