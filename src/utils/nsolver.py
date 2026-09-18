import dolfin as df
import math
import sys
from .snes_problem import SNESProblem


def _log(comm=None):
    """The FE logger for this solve. Built per call (Loggers are stateless and cheap) so the
    rank gate always follows the communicator the caller is actually solving on."""
    from .log_mpi import get_logger
    return get_logger("fe", comm)


class NSolver(object):
    def __init__(self, params):
        self.parameters = params
        self.isfirstiteration = False
        # Robustness records of the most recent line-search solve (D1 instrumentation).
        self.last_solve_iters = 0
        self.last_solve_alpha = 1.0

        Ftotal = params["F"]
        w = params["w"]
        bcs = params["boundary_conditions"]
        Jac = params["Jacobian"]

        self.problem = df.NonlinearVariationalProblem(
            Ftotal,
            w,
            bcs=bcs,
            J=Jac,
            form_compiler_parameters={"representation": "uflacs"},
        )
        self.nsolver = df.NonlinearVariationalSolver(self.problem)
        self.nsolver.parameters["nonlinear_solver"] = "newton"
        self.newton_params = self.nsolver.parameters["newton_solver"]
        self.problem_snes = SNESProblem(Ftotal, w, bcs)
        # Reuse a dedicated update vector to avoid per-iteration allocations in the manual solver branch.
        self._dww = w.copy(deepcopy=True)

    @staticmethod
    def default_parameters():
        return {"rel_tol": 1e-7, "abs_tol": 1e-7, "max_iter": 200}

    def _newton_linesearch_type0(self, abs_tol, rel_tol, maxiter, is_root, mode=0,
                                 alpha_min=2.0 ** -8):
        """Globalized (backtracking line-search) Newton for the Type-0 path. Identical to
        plain Newton on well-behaved steps (alpha=1 is tried and accepted first); only
        when the full step yields a non-finite or non-decreasing residual does it damp
        alpha (Armijo sufficient-decrease) -- exactly the overshoot protection the stock
        DOLFIN Newton lacks. Raises if no finite descent step exists or it does not
        converge, so a genuine failure is never silently masked. Per-iteration progress
        prints only at mode > 1 (the default mode=1 keeps this primary path quiet)."""
        Jac = self.parameters["Jacobian"]
        Ftotal = self.parameters["F"]
        w = self.parameters["w"]
        bcs = self.parameters["boundary_conditions"]
        uflp = {"representation": "uflacs"}
        alpha_min = float(alpha_min)
        if not (0.0 < alpha_min <= 1.0):
            raise ValueError("newton_alpha_min must satisfy 0 < value <= 1")

        B = df.assemble(Ftotal, form_compiler_parameters=uflp)
        for bc in bcs:
            bc.apply(B)
        res0 = B.norm("l2")
        if not math.isfinite(res0):
            raise RuntimeError("damped Newton: non-finite residual at the warm-start state")
        res = res0
        dww = self._dww
        it = 0
        alpha_used_min = 1.0
        while res > abs_tol and (res / res0) > rel_tol and it < maxiter:
            it += 1
            A, b = df.assemble_system(Jac, -Ftotal, bcs, form_compiler_parameters=uflp)
            df.solve(A, dww.vector(), b, "mumps")
            w_backup = w.vector().get_local()
            dloc = dww.vector().get_local()
            alpha = 1.0
            accepted = False
            n_nonfinite = 0
            n_trials = 0
            last_trial_res = float("nan")
            while alpha >= alpha_min:
                n_trials += 1
                w.vector().set_local(w_backup + alpha * dloc)
                w.vector().apply("insert")
                Bt = df.assemble(Ftotal, form_compiler_parameters=uflp)
                for bc in bcs:
                    bc.apply(Bt)
                rt = Bt.norm("l2")
                # accept the first step that is finite AND reduces the residual (Armijo)
                if math.isfinite(rt) and rt < (1.0 - 1.0e-4 * alpha) * res:
                    res = rt
                    accepted = True
                    break
                # Failure-class record (D1): distinguish a NON-FINITE trial residual (the
                # update left the admissible cone -- e.g. J<=0 under the isochoric split)
                # from a merely NON-DECREASING one (a genuine no-solution plateau). The two
                # demand opposite remedies, and the message alone could not tell them apart.
                if not math.isfinite(rt):
                    n_nonfinite += 1
                last_trial_res = rt
                alpha *= 0.5
            if not accepted:
                # Is the floor itself the binding constraint? One extra probe far below it
                # (failure path only -- never on the hot path) distinguishes "no solution in
                # this direction" from "a residual-decreasing step exists, but alpha_min sat
                # above it". MEASURED on the isochoric-split HO path: the exact Newton
                # direction had ||d|| ~ 5e2 off a 59 Pa ground tangent and was descent-monotone
                # only for alpha < 1e-3, i.e. one halving below the 2^-8 default -- and step
                # halving cannot fix that, because the floor is on alpha, not on the increment.
                probe_alpha = alpha_min / 64.0
                w.vector().set_local(w_backup + probe_alpha * dloc)
                w.vector().apply("insert")
                Bp = df.assemble(Ftotal, form_compiler_parameters=uflp)
                for bc in bcs:
                    bc.apply(Bp)
                probe_res = Bp.norm("l2")
                # A MEANINGFUL decrease only. MEASURED on the LVW isovolumic ceiling re-run
                # (alpha_min 2^-14, 46 firings): probe_res equalled res to 3 decimals every
                # time -- a < 5e-4 relative decrease at alpha ~ 1e-6 is not a descent window,
                # and deepening the floor 64x did not move a single fold. Require the probe
                # to beat the residual by a FIXED 1e-3 margin (the Armijo test above collapses
                # to 'any decrease' at alpha ~ 1e-6, which is exactly the false positive).
                floor_bound = math.isfinite(probe_res) and probe_res < (1.0 - 1.0e-3) * res
                w.vector().set_local(w_backup)
                w.vector().apply("insert")
                self.last_solve_iters = it
                self.last_solve_alpha = alpha * 2.0
                if floor_bound:
                    _log(w.function_space().mesh().mpi_comm()).warn(
                        "solve",
                        "line search hit its damping FLOOR while a residual-decreasing step "
                        "still existed below it -- this failure is the floor, not the physics; "
                        "deepen SimDet['newton_alpha_min']",
                        solver="ls-newton", alpha_min="%.2e" % alpha_min,
                        probe_alpha="%.2e" % probe_alpha, res="%.6e" % res,
                        probe_res="%.6e" % probe_res, iter=it)
                # NOTE: the leading text is a parsing contract (config/*_signatures.json,
                # scripts/summarize_ed_es_validation.py) -- fields are APPENDED, never reworded.
                raise RuntimeError(
                    "damped Newton: no finite descent step (alpha < %.1e) at res=%.3e"
                    " last_trial_res=%.3e nonfinite_trials=%d/%d iter=%d"
                    % (alpha_min, res, last_trial_res, n_nonfinite, n_trials, it)
                )
            if mode > 1:
                # Per-Newton-iteration detail, opt-in at mode > 1: this is the one path that can
                # emit per iteration, so it stays behind an explicit request for that detail.
                _log(w.function_space().mesh().mpi_comm()).info(
                    "solve", "line-search Newton iteration",
                    solver="ls-newton", iter=it, alpha=alpha, res=res, rel=res / res0)
            alpha_used_min = min(alpha_used_min, alpha)
        # Robustness margin of the completed solve (acceptance criterion 2): how many Newton
        # iterations it needed and how far the line search had to damp. Read by callers that
        # report the worst case over a continuation ramp.
        self.last_solve_iters = it
        self.last_solve_alpha = alpha_used_min
        if not (res <= abs_tol or (res / res0) <= rel_tol):
            raise RuntimeError(
                "damped Newton did not converge (res=%.3e rel=%.3e after %d iters)"
                % (res, res / res0, it)
            )

    def solvenonlinear(self):
        params = self.parameters
        abs_tol = params.get("abs_tol", self.default_parameters()["abs_tol"])
        rel_tol = params.get("rel_tol", self.default_parameters()["rel_tol"])
        maxiter = params.get("max_iter", self.default_parameters()["max_iter"])
        mode = params.get("mode", 0)

        Jac = params["Jacobian"]
        Ftotal = params["F"]
        w = params["w"]
        bcs = params["boundary_conditions"]
        solvertype = params["Type"]
        comm = w.function_space().mesh().mpi_comm()
        is_root = df.MPI.rank(comm) == 0
        # Quiet DOLFIN's per-solve chatter ("Solving nonlinear variational problem." plus the
        # full "Newton iteration N: r (abs)=..." report) on the live raw-log stream: on a healthy
        # solve it emits ~5 lines PER Newton solve and there are hundreds of solves, which floods
        # the Raw log and buries the real stage progress. WARNING keeps genuine solver warnings
        # and errors visible, and non-convergence still RAISES (error_on_nonconvergence) -> the
        # job fails with a reason, so nothing is masked. Shared by every LV-FE solve
        # (heart, unload, calibrate). The single configuration point now lives in log_mpi.
        from .log_mpi import configure_dolfin_logging
        configure_dolfin_logging(comm)

        if solvertype == 0:
            # Standard FEniCS Newton solve using a direct linear solver (robust but memory-heavy in 3D).
            self.newton_params["linear_solver"] = "mumps"
            self.newton_params["absolute_tolerance"] = abs_tol
            self.newton_params["relative_tolerance"] = rel_tol
            self.newton_params["maximum_iterations"] = maxiter
            self.newton_params["report"] = False  # suppress per-iteration Newton report on the raw-log stream
            # Opt-in MUMPS null-pivot detection (SimDet["mumps_null_pivot"]). At the
            # late-systolic over-ejection limit point the equal-order P1P1 saddle-point
            # tangent develops a (near-)zero pivot -> stock MUMPS aborts the
            # factorization with DIVERGED_PC_FAILED (residual norm 0.0, 0 iterations)
            # BEFORE any Newton step. ICNTL(24)=1 lets MUMPS detect and regularize the
            # null pivot so the factorization completes; Newton's residual gate
            # (abs_tol/rel_tol) is UNCHANGED, so a genuine fold (no real solution) still
            # fails to converge and RAISES -- this crosses a marginally-singular tangent
            # without masking a true singularity. CNTL(3) sets the null-pivot threshold.
            # Off by default -> stock MUMPS (production paths unchanged).
            self._mumps_null_pivot = bool(params.get("mumps_null_pivot", False))
            if self._mumps_null_pivot:
                # Announce ONCE per process (hundreds of solves per cycle would otherwise
                # flood the raw log -- same anti-spam rationale as the Newton report above).
                from heartlog import announce_once
                if announce_once("nsolver:mumps_null_pivot"):
                    _log(comm).warn(
                        "solve",
                        "MUMPS null-pivot detection ENGAGED (SimDet['mumps_null_pivot']=True) -> "
                        "ICNTL(24)=1 to regularize the near-singular P1P1 saddle-point pivot at "
                        "the systolic limit point. Newton residual tolerance is unchanged; a true "
                        "fold still raises.",
                        icntl_24=1, cntl_3=float(params.get("mumps_cntl_3", -1.0e-8)),
                    )
                df.PETScOptions.set("mat_mumps_icntl_24", 1)
                df.PETScOptions.set("mat_mumps_cntl_3", float(params.get("mumps_cntl_3", -1.0e-8)))
            try:
                if params.get("newton_linesearch", False):
                    # PRIMARY globalized (backtracking line-search) Newton, opt-in via
                    # SimDet["newton_linesearch"]. The stock DOLFIN Newton takes FULL steps
                    # (no line search), so in stiff regimes -- e.g. the isovolumic active
                    # twitch near a fold -- it overshoots from a finite state into a
                    # non-finite (nan) residual (iter 0 finite, iter 1 nan) and cannot
                    # recover. The line search tries alpha=1 first, so well-behaved steps are
                    # identical to plain Newton; it damps alpha ONLY when the full step is
                    # non-finite or non-decreasing. Converges to the true solution or raises
                    # -- never masks. Stock Newton remains the default everywhere the flag is
                    # unset (production FCH/LVW unchanged).
                    self._newton_linesearch_type0(
                        abs_tol, rel_tol, maxiter, is_root, mode,
                        alpha_min=float(params.get("newton_alpha_min", 2.0 ** -8)),
                    )
                else:
                    self.nsolver.solve()
            finally:
                if self._mumps_null_pivot:
                    # PETSc's option database is process-global. Clear these even when the
                    # nonlinear solve raises, otherwise a failed ED/ES retry silently changes
                    # every later MUMPS solve in the same process.
                    for _k in ("mat_mumps_icntl_24", "mat_mumps_cntl_3"):
                        df.PETScOptions.clear(_k)

        elif solvertype == 1:
            # PETSc SNES path with explicit KSP/PC options (fieldsplit for saddle-point systems).
            import petsc4py

            petsc4py.init(sys.argv)
            from petsc4py import PETSc

            b = df.PETScVector()
            J_mat = df.PETScMatrix()
            opts = PETSc.Options()
            opts.setValue("snes_type", "vinewtonssls")
            opts.setValue("ksp_type", "bcgs")
            opts.setValue("pc_type", "fieldsplit")
            opts.setValue("pc_fieldsplit_type", "additive")
            opts.setValue("pc_fieldsplit_detect_saddle_point", True)
            opts.setValue("fieldsplit_0_ksp_type", "preonly")
            opts.setValue("fieldsplit_0_pc_type", "lu")
            opts.setValue("fieldsplit_1_ksp_type", "preonly")
            opts.setValue("fieldsplit_1_pc_type", "bjacobi")
            opts.setValue("snes_linesearch_type", "basic")
            # SNES/KSP monitors were hardcoded on here and rank-unguarded, so every rank wrote
            # per-iteration residuals straight to PETSC_COMM_WORLD's stdout -- thousands of
            # unprefixed lines interleaved from N ranks. Now opt-in, and announced when engaged.
            from .log_mpi import configure_petsc_monitors
            configure_petsc_monitors(bool(params.get("petsc_monitors", False)), _log(comm))

            self.snes = PETSc.SNES().create(df.MPI.comm_world)
            self.snes.setFromOptions()
            self.snes.setFunction(self.problem_snes.F, b.vec())
            self.snes.setJacobian(self.problem_snes.J, J_mat.mat())
            self.snes.solve(None, self.problem_snes.u.vector().vec())

        else:
            # Manual Picard/Newton-like loop: assemble (J, -F) each iter and apply an additive update to `w`.
            it = 0
            if not self.isfirstiteration:
                A, b = df.assemble_system(
                    Jac,
                    -Ftotal,
                    bcs,
                    form_compiler_parameters={"representation": "uflacs"},
                )
                resid0 = b.norm("l2")
                rel_res = b.norm("l2") / resid0
                res = resid0
                if mode > 0:
                    _log(comm).info("solve", "Newton iteration",
                                    solver="manual", iter=it, res=res, rel=rel_res)
                solve(A, w.vector(), b)
                it += 1
                self.isfirstiteration = True

            B = df.assemble(Ftotal, form_compiler_parameters={"representation": "uflacs"})
            for bc in bcs:
                bc.apply(B)

            res = B.norm("l2")
            resid0 = res
            rel_res = 1.0

            if mode > 0:
                _log(comm).info("solve", "Newton iteration",
                                solver="manual", iter=it, res=res, rel=rel_res)

            dww = self._dww
            dww.vector()[:] = 0.0

            while (rel_res > rel_tol and res > abs_tol) and it < maxiter:
                it += 1
                A, b = df.assemble_system(
                    Jac,
                    -Ftotal,
                    bcs,
                    form_compiler_parameters={"representation": "uflacs"},
                )
                solve(A, dww.vector(), b)
                w.vector().axpy(1.0, dww.vector())

                B = df.assemble(
                    Ftotal, form_compiler_parameters={"representation": "uflacs"}
                )
                for bc in bcs:
                    bc.apply(B)

                res = B.norm("l2")
                rel_res = res / resid0

                if mode > 0:
                    _log(comm).info("solve", "Newton iteration",
                                    solver="manual", iter=it, res=res, rel=rel_res)

            if (rel_res > rel_tol and res > abs_tol) or math.isnan(res):
                raise RuntimeError("Failed Convergence")
