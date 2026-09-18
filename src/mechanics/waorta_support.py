import dolfin as df
import numpy as np


def dirichlet_bcs_waorta(W, facetboundaries, sim_det, zero_vec):
    """
    Return default Dirichlet BCs for waorta runs.

    The aortic ring is clamped unless spring BCs are enabled with `mv_aorta`,
    in which case the variational spring form replaces the hard constraint.
    """
    aorta_ring = sim_det["aorta_ring"]
    springbc = bool(sim_det.get("springbc"))
    if springbc and sim_det.get("mv_aorta"):
        return []
    return [df.DirichletBC(W.sub(0), zero_vec, facetboundaries, aorta_ring)]


def build_passive_form_waorta(
    add_passive_region,
    sim_det,
    dx_me,
    w_me,
    wtest_me,
    WpRub_me,
    WpAorta_me,
    rubber_region_sef=None,
):
    """
    Assemble passive energy contributions for waorta configurations:
      - split atria vs ventricles via region ids and per-chamber scaling
      - optional rubber/aorta/interface regions using specialized SEFs

    ``rubber_region_sef`` (optional): ``{region_id: SEF}`` overriding ``WpRub_me`` for
    individual ``rubber_region`` ids -- the aortic-valve cap (3) under its own law while
    the mitral cap (5) and basal ring (4) keep the shared plug form byte-for-byte. An id
    that is not in ``rubber_region`` is refused: a silent no-op override would leave a
    note believing the cap law is on when no cell carries it.
    """
    F1 = None
    overrides = {int(k): v for k, v in (rubber_region_sef or {}).items()}
    if overrides:
        rubber_ids = {int(r) for r in sim_det.get("rubber_region", [])}
        missing = sorted(set(overrides) - rubber_ids)
        if missing:
            raise ValueError(
                "rubber_region_sef overrides region(s) %s that are not in rubber_region=%s"
                % (missing, sorted(rubber_ids)))

    if sim_det.get("active_region", []):
        # Use configurable region ids for atria/ventricles.
        la_rid = sim_det.get("la_rid", 11)
        ra_rid = sim_det.get("ra_rid", 8)

        for regionid in sim_det["active_region"]:
            if regionid in (la_rid, ra_rid):
                F1 = add_passive_region(
                    F1,
                    dx_me(int(regionid)),
                    float(sim_det["atrial_passive_coeff"]),
                )
            else:
                F1 = add_passive_region(
                    F1,
                    dx_me(int(regionid)),
                    float(sim_det["ventricular_passive_coeff"]),
                )

        # Keep legacy behavior: waorta configs are expected to define `rubber_region` when `active_region` is set.
        for regionid in sim_det["rubber_region"]:
            W_region = overrides.get(int(regionid), WpRub_me)
            rubber_form = df.derivative(W_region, w_me, wtest_me) * dx_me(int(regionid))
            F1 = rubber_form if F1 is None else (F1 + rubber_form)

    if sim_det.get("aorta_region", []):
        for regionid in sim_det["aorta_region"]:
            aorta_form = df.derivative(WpAorta_me, w_me, wtest_me) * dx_me(int(regionid))
            F1 = aorta_form if F1 is None else (F1 + aorta_form)

    if sim_det.get("interface_region", []):
        for regionid in sim_det["interface_region"]:
            interface_form = df.derivative(WpAorta_me, w_me, wtest_me) * dx_me(int(regionid))
            F1 = interface_form if F1 is None else (F1 + interface_form)

    return F1


def build_stabilization_mu_waorta(QDG, uflforms, matid_me, sim_det, region_mu=None):
    """
    Build per-cell shear modulus used by P1P1 pressure stabilization for waorta meshes.

    Region masks are based on `matid_me` values and the configured region id lists.
    ``region_mu`` (optional): ``{region_id: shear-modulus expression}`` applied LAST, so a
    region carrying its own law (the aortic-valve cap) is stabilised with that law's
    modulus rather than the shared rubber one.
    """
    F_mu = df.Function(QDG)  # DG0

    mu_main = df.project(uflforms.shear_main(), QDG)
    mu_aorta = df.project(uflforms.shear_aorta(), QDG)
    mu_rubber = df.project(uflforms.shear_rubber(), QDG)

    mu_main_local = mu_main.vector().get_local()
    mu_aorta_local = mu_aorta.vector().get_local()
    mu_rubber_local = mu_rubber.vector().get_local()

    # Work on local arrays to avoid repeated vector assembly calls.
    F_mu_local = F_mu.vector().get_local()
    mf_matid = df.MeshFunction("size_t", QDG.mesh(), QDG.mesh().topology().dim())
    mf_matid.set_all(0)
    mf_matid.array()[:] = matid_me.array()

    aortic_regions = list(
        set(sim_det.get("aorta_region", [])) | set(sim_det.get("interface_region", []))
    )
    rubber_regions = list(set(sim_det.get("av_region", [])))

    mask_aorta = np.isin(mf_matid.array(), aortic_regions)
    mask_rubber = np.isin(mf_matid.array(), rubber_regions)
    mask_main = ~(mask_aorta | mask_rubber)

    F_mu_local[mask_main] = mu_main_local[mask_main]
    F_mu_local[mask_aorta] = mu_aorta_local[mask_aorta]
    F_mu_local[mask_rubber] = mu_rubber_local[mask_rubber]
    for rid, mu_expr in (region_mu or {}).items():
        mask_r = mf_matid.array() == int(rid)
        mu_r_local = df.project(mu_expr, QDG).vector().get_local()
        F_mu_local[mask_r] = mu_r_local[mask_r]

    F_mu.vector().set_local(F_mu_local)
    F_mu.vector().apply("insert")
    return F_mu


def maybe_add_aorta_pressure_term(Fp_me, uflforms, sim_det):
    """Optionally include aortic cavity pressure in p-control mode."""
    if sim_det.get("aorta_pres"):
        return Fp_me + uflforms.Aortacavitypres()
    return Fp_me


def lv_volume_waorta(uflforms):
    """LV cavity volume specialization for waorta topology."""
    return uflforms.LVcavityvol_waorta()


def laplace_waorta(uflforms):
    """Laplace solve specialization for waorta meshes (if enabled)."""
    return uflforms.solveLaplaceEquation_waorta()
