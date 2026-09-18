import dolfin as df
import ufl
from collections.abc import Iterable as IterableABC
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple


MeasureFactory = Callable[[int], df.Measure]
TopSpringGetter = Callable[[], Tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]]


class SpringBCContext:
    """
    Container for the fields required to build the spring/dashpot UFL forms.

    Only geometric information and unknowns live here; configuration values
    (boundary ids, scaling factors, etc.) stay inside SimDet, which keeps the
    builder general while avoiding direct coupling with MEmodel3_light.
    """

    def __init__(
        self,
        displacement,
        previous_displacement,
        test_function,
        measure,
        current_normal,
        reference_normal,
        identity,
        ed_displacement=None,
        top_spring_getter=None,
    ):
        self.displacement = displacement
        self.previous_displacement = previous_displacement
        self.test_function = test_function
        self.measure = measure
        self.current_normal = current_normal
        self.reference_normal = reference_normal
        self.identity = identity
        self.ed_displacement = ed_displacement
        self.top_spring_getter = top_spring_getter


def build_spring_form(
    option: Optional[str],
    ctx: SpringBCContext,
    sim_options: Dict,
) -> Optional[ufl.core.expr.Expr]:
    """
    Build the requested spring/dashpot form and return the UFL expression.

    Parameters
    ----------
    option:
        Name of the spring boundary option ("waorta", "fch", "lv", "biv").
    ctx:
        SpringBCContext with the finite-element fields and measures.
    sim_options:
        Mechanics configuration (SimDet) dictionary.
    """

    if not option:
        return None

    option = option.lower()
    builder_map = {
        "waorta": _build_waorta_form,
        "fch": _build_fch_form,
        "lv": _build_lv_form,
        "biv": _build_biv_form,
    }

    builder = builder_map.get(option)
    if builder is None:
        raise ValueError(f"Unknown spring boundary option '{option}'.")

    return builder(ctx, sim_options)


def _spring_surface(
    measure: Optional[df.Measure],
    normal_proj,
    k_pair: Sequence,
    c_pair: Sequence,
    disp_expr,
    vel_expr,
    v_test,
    identity,
    tangential_proj=None,
):
    if measure is None:
        return None
    tangential_proj = tangential_proj or (identity - normal_proj)

    # Separate normal/tangential stiffness and damping reduces rigid-mode drift while avoiding over-constraint.
    form = df.inner(normal_proj * (k_pair[0] * disp_expr + c_pair[0] * vel_expr), v_test) * measure
    form += df.inner(tangential_proj * (k_pair[1] * disp_expr + c_pair[1] * vel_expr), v_test) * measure
    return form


def _flatten_ids(values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, IterableABC) and not isinstance(value, (str, bytes)):
            for nested in _flatten_ids(value):
                yield nested
        else:
            yield value


def _sum_measures(measure_factory: MeasureFactory, ids: Iterable[Optional[int]]):
    # Accept nested lists/tuples and ignore invalid entries to keep SimDet configuration flexible.
    valid_ids = []
    for idx in _flatten_ids(ids):
        try:
            valid_ids.append(int(idx))
        except (TypeError, ValueError):
            continue
    if not valid_ids:
        return None
    measure = measure_factory(valid_ids[0])
    for idx in valid_ids[1:]:
        measure += measure_factory(idx)
    return measure


def _ensure_pair(values, default: Optional[Sequence] = None):
    if isinstance(values, (list, tuple)):
        if len(values) == 2:
            return values[0], values[1]
    if default is not None and hasattr(default, "__len__") and len(default) == 2:
        return default[0], default[1]
    return values, values


def _spring_forms_for_surfaces(ctx, surfaces, c_pair, delta, rel):
    form = None
    for spec in surfaces:
        measure_ids = spec.get("ids")
        use_sum = spec.get("use_sum", True)
        measure = (
            _sum_measures(ctx.measure, measure_ids) if use_sum else ctx.measure(measure_ids)
        )
        if measure is None:
            continue
        normal_proj = spec.get("normal") or df.outer(ctx.current_normal, ctx.current_normal)
        tangential_proj = spec.get("tangential")
        k_pair = spec.get("k_pair")
        surf_form = _spring_surface(
            measure,
            normal_proj,
            k_pair,
            c_pair,
            delta,
            rel,
            ctx.test_function,
            ctx.identity,
            tangential_proj=tangential_proj,
        )
        if surf_form is not None:
            form = surf_form if form is None else form + surf_form
    return form or 0


def _build_waorta_form(ctx: SpringBCContext, sim_options: Dict):
    if ctx.ed_displacement is None:
        raise ValueError("waorta spring option requires ED displacement.")

    k_pair = _ensure_pair(sim_options.get("springparam", [2.0e3, 2.0e3]))
    c_pair = _ensure_pair(sim_options.get("dashpotparam", [2.0e2, 2.0e1]))

    # Anchoring reference of the epicardial (pericardial) Robin support. The legacy
    # anchor is the END-DIASTOLIC state: the support is traction-free at ED and charges
    # the myocardium for every excursion away from its ED SHAPE, which is what produces
    # the measured 3.27x (Ecc) / 2.00x (Ell) endo-to-epi peak-strain gradient and the
    # 4.6x under-twist of the coupled 8159 run (campaign doc section 8.3). The opt-in
    # `spring_ref_unloaded` anchors it on the UNLOADED reference instead (delta = u), so
    # the support no longer resists departure from ED. `u_me_ED` is zero until the driver
    # assigns it after the loading phase, so the two agree during preload and differ only
    # in the coupled loop. Default off => byte-identical legacy behaviour.
    if sim_options.get("spring_ref_unloaded"):
        delta = ctx.displacement
        rel = ctx.displacement - ctx.previous_displacement
    else:
        delta = ctx.displacement - ctx.ed_displacement
        rel = ctx.displacement - ctx.previous_displacement - ctx.ed_displacement
    normal_proj = df.outer(ctx.current_normal, ctx.current_normal)

    coeff = list(k_pair)
    # The epicardium (`epiid`) and the apex (`apxid`) share one stiffness unless the
    # OPTIONAL `springparam_apex` splits them. The apex is the anchor the coupled run
    # pushes AGAINST (measured: the apex RISES 2.03 mm where the study has it stationary),
    # so it is worth grading on its own. Absent => the single summed surface, i.e.
    # byte-identical to the legacy path.
    apex_override = sim_options.get("springparam_apex")
    if apex_override is None:
        surfaces = [
            {"ids": [sim_options.get("epiid"), sim_options.get("apxid")], "normal": normal_proj, "k_pair": coeff},
        ]
    else:
        ka = _ensure_pair(apex_override)
        surfaces = [
            {"ids": [sim_options.get("epiid")], "normal": normal_proj, "k_pair": coeff},
            {"ids": [sim_options.get("apxid")], "normal": normal_proj,
             "k_pair": [float(ka[0]), float(ka[1])]},
        ]

    if sim_options.get("mv_aorta"):
        mv_ring = sim_options.get("aorta_ring")
        if mv_ring is not None:
            # Mobile annulus: the aortic cut-rim Robin spring that REPLACES the hard
            # `u=0` clamp (waorta_support.dirichlet_bcs_waorta drops the Dirichlet
            # whenever springbc and mv_aorta are both set, so the two never both act).
            # Its stiffness has historically been hard-wired to springparam/5; the
            # OPTIONAL `springparam_aorta_ring` makes it an independent, absolute
            # [k_n,k_t] so the basal/annular constraint can be graded without moving
            # the epicardial spring with it. Absent => springparam/5 exactly, i.e.
            # byte-identical to the legacy path.
            ring_override = sim_options.get("springparam_aorta_ring")
            if ring_override is None:
                ring_coeff = [coeff[0] / 5.0, coeff[1] / 5.0]
            else:
                kr = _ensure_pair(ring_override)
                ring_coeff = [float(kr[0]), float(kr[1])]
            surfaces.append({"ids": mv_ring, "use_sum": False, "normal": normal_proj, "k_pair": ring_coeff})

    if sim_options.get("spring_on_aorta"):
        wall_id = sim_options.get("aorta_ext_wall")
        if wall_id is not None:
            # OPTIONAL `springparam_aorta_wall` makes the perivascular tether an independent,
            # absolute [k_n,k_t] (Pa/cm) so the aortic support can be graded without moving
            # the epicardial spring with it. Absent => the full epicardial pair, exactly as
            # before (byte-identical; asserted in tests/test_waorta_spring_ring.py).
            wall_override = sim_options.get("springparam_aorta_wall")
            wall_k = k_pair if wall_override is None else [float(x) for x in _ensure_pair(wall_override)]
            surfaces.append({"ids": wall_id, "use_sum": False, "normal": normal_proj, "k_pair": wall_k})

    return _spring_forms_for_surfaces(ctx, surfaces, c_pair, delta, rel)


def _resolve_fch_region_kpairs(sim_options: Dict):
    """Resolve the per-region (ventricular-epi / atrial / aorta) spring stiffness
    pairs for the four-chamber Robin support, with strict backward compatibility.

    The legacy single-``springparam`` path applied ONE [k_n,k_t] to the ventricular
    epicardium (epi+apex) and the atrial wall, and 1.5x that to the aorta surface.
    To support PER-REGION calibration of the pericardial spring (ventricular
    epicardium vs atrial wall vs aorta — the chambers are NOT facet-separable beyond
    that, since LV/RV share epiid=9 and LA/RA share atrialid=10), three OPTIONAL
    SimDet keys override that single value:

      - ``springparam_epi``     -> ventricular epicardium (facets epiid + apxid)
      - ``springparam_atrial``  -> atrial wall (facet atrialid; only when ``fch_fe``)
      - ``springparam_aorta``   -> aorta surface (facet aortaid)

    A single ``springparam_by_region`` dict ({"epi":[...], "atrial":[...],
    "aorta":[...]}) is an alternative grouping; the explicit per-region keys win.
    Any region without an override falls back to the single ``springparam`` (the
    aorta to 1.5x the resolved epicardial pair), so when NONE of the new keys are
    present the assembled form is byte-identical to the legacy single-spring path.
    """
    base_k = list(_ensure_pair(sim_options.get("springparam", [2.0e3, 2.0e3])))
    by_region = sim_options.get("springparam_by_region") or {}

    def region_k(name, fallback):
        val = sim_options.get("springparam_%s" % name)
        if val is None:
            val = by_region.get(name)
        if val is None:
            return list(fallback)
        kp = _ensure_pair(val)
        return [kp[0], kp[1]]

    epi_k = region_k("epi", base_k)
    atrial_k = region_k("atrial", base_k)
    # Aorta inherits 1.5x the resolved epicardial pair unless explicitly overridden
    # (== the legacy default when epi itself is not overridden).
    aorta_k = region_k("aorta", [1.5 * epi_k[0], 1.5 * epi_k[1]])
    # Aorta TRUNCATED cut-rim (facet ``aorta_wall``): a MODEST omni-directional (normal +
    # tangential) Robin tether for the elastic constraint of the cut-away mediastinal tissue,
    # replacing the legacy hard ``u=0`` Dirichlet there. Opt-in via ``aorta_cutface_robin``;
    # default ~3 kPa/mm (between the repo's 3 kPa/mm aorta surface and Strocchi 2020's 10 kPa/mm
    # vessel-rim spring) -- enough to remove the aorta rigid modes without biasing deformation.
    cutface_k = region_k("aorta_cutface", [3.0e3, 3.0e3])
    return epi_k, atrial_k, aorta_k, cutface_k


def _build_fch_form(ctx: SpringBCContext, sim_options: Dict):
    epi_k, atrial_k, aorta_k, cutface_k = _resolve_fch_region_kpairs(sim_options)
    c_pair = _ensure_pair(sim_options.get("dashpotparam", [2.0e2, 2.0e1]))

    delta = ctx.displacement
    rel = ctx.displacement - ctx.previous_displacement

    normal_proj = df.outer(ctx.current_normal, ctx.current_normal)
    surfaces = [
        {"ids": [sim_options.get("epiid"), sim_options.get("apxid")], "normal": normal_proj, "k_pair": epi_k},
    ]
    aorta_id = sim_options.get("aortaid")
    if aorta_id is not None:
        surfaces.append(
            {
                "ids": aorta_id,
                "use_sum": False,
                "normal": df.outer(ctx.reference_normal, ctx.reference_normal),
                "k_pair": aorta_k,
            }
        )

    if sim_options.get("fch_fe"):
        atrial_id = sim_options.get("atrialid")
        if atrial_id is not None:
            tangential = ctx.identity - df.outer(ctx.current_normal, ctx.current_normal)
            surfaces.append(
                {
                    "ids": atrial_id,
                    "use_sum": False,
                    "normal": df.outer(ctx.reference_normal, ctx.reference_normal),
                    "k_pair": atrial_k,
                    "tangential": tangential,
                }
            )

    # Truncated aorta cut-rim (facet ``aorta_wall``): opt-in MODEST omni-directional Robin tether
    # (normal + explicit tangential, reference-normal), replacing the legacy hard ``u=0`` Dirichlet
    # (which `_default_dirichlet_bcs` drops when this flag is set, so they never both act). Removes
    # the aorta rigid modes / improves conditioning without the over-stiffness of a clamp.
    if sim_options.get("aorta_cutface_robin"):
        cutface_id = sim_options.get("aorta_wall")
        if cutface_id is not None:
            cf_tangential = ctx.identity - df.outer(ctx.reference_normal, ctx.reference_normal)
            surfaces.append(
                {
                    "ids": cutface_id,
                    "use_sum": False,
                    "normal": df.outer(ctx.reference_normal, ctx.reference_normal),
                    "k_pair": cutface_k,
                    "tangential": cf_tangential,
                }
            )

    return _spring_forms_for_surfaces(ctx, surfaces, c_pair, delta, rel)


def _build_lv_form(ctx: SpringBCContext, sim_options: Dict):
    if ctx.ed_displacement is None:
        raise ValueError("lv spring option requires ED displacement.")

    k_pair = _ensure_pair(sim_options.get("springparam", [2.0e3, 2.0e3]))
    c_pair = _ensure_pair(sim_options.get("dashpotparam", [2.0e2, 2.0e1]))

    if sim_options.get("spring_unloaded_reference", False):
        # Unloaded-referenced spring (always active at the operating point):
        # measure total displacement from the reference mesh so the spring stays
        # active at the ED operating point instead of relaxing there (delta->0)
        # once `u_ED` is reassigned to the loaded state at the closed-loop start.
        delta = ctx.displacement
        rel = ctx.displacement - ctx.previous_displacement
    else:
        # Default (ED-referenced): spring vanishes at the ED operating point.
        delta = ctx.displacement - ctx.ed_displacement
        rel = ctx.displacement - ctx.previous_displacement - ctx.ed_displacement
    normal_proj = df.outer(ctx.reference_normal, ctx.reference_normal)

    coeff = list(k_pair)
    # Spring surface ids. Default = epi + apex + base. An opt-in `spring_facetids_lv`
    # lets a caller free the base (drop `topid`) while keeping the epicardial spring —
    # the physiological "pericardium constrains epi, base free" configuration.
    # `springparam`=[k_n,k_t] is the only effective epicardial stiffness.
    spring_ids = sim_options.get(
        "spring_facetids_lv",
        [sim_options.get("epiid"), sim_options.get("apxid"), sim_options.get("topid")],
    )
    surfaces = [
        {
            "ids": spring_ids,
            "normal": normal_proj,
            "k_pair": coeff,
        }
    ]

    # Optional fractional spring at the base when the base is otherwise free:
    # applies `spring_top_fraction`*[k_n,k_t] at `topid` only if `topid` is not
    # already in the main spring surface (avoids double-counting).
    top_fraction = sim_options.get("spring_top_fraction")
    topid_cfg = sim_options.get("topid")
    if top_fraction and topid_cfg is not None and int(topid_cfg) not in set(_flatten_ids(spring_ids)):
        frac = float(top_fraction)
        surfaces.append(
            {
                "ids": topid_cfg,
                "use_sum": False,
                "normal": normal_proj,
                "k_pair": [coeff[0] * frac, coeff[1] * frac],
            }
        )

    # Optional explicit basal spring with ABSOLUTE [k_n_base, k_t_base] at `topid`
    # when the base is otherwise free (topid not in the main spring surface).
    # Use [k_n_base, 0] for a light NORMAL-ONLY basal support: it resists basal
    # eversion/buckling at high cavity pressure (which otherwise causes the
    # volume-control unloading solve to lose ellipticity ~5 mmHg) without clamping
    # in-plane (radial/circumferential) motion. Distinct from the epicardial spring
    # and absolute (no scaling), so its stiffness is exactly
    # [k_n_base, k_t_base]. Applied consistently in unloading + calibration + sim.
    basal = sim_options.get("spring_basal")
    if basal is not None and topid_cfg is not None and int(topid_cfg) not in set(_flatten_ids(spring_ids)):
        bk = _ensure_pair(basal)
        surfaces.append(
            {
                "ids": topid_cfg,
                "use_sum": False,
                "normal": normal_proj,
                "k_pair": [bk[0], bk[1]],
            }
        )

    if sim_options.get("spring_atbase") and ctx.top_spring_getter:
        topid = sim_options.get("topid")
        if topid is None:
            raise ValueError("spring_atbase requires 'topid' in SimDet.")
        a_, b_ = ctx.top_spring_getter()
        form = _spring_forms_for_surfaces(ctx, surfaces, c_pair, delta, rel)
        form -= a_ * df.inner(b_, ctx.test_function) * ctx.measure(topid)
        return form

    return _spring_forms_for_surfaces(ctx, surfaces, c_pair, delta, rel)


def _build_biv_form(ctx: SpringBCContext, sim_options: Dict):
    k_spring = sim_options.get("springparam", [])
    c_damping = sim_options.get("dashpotparam", [])
    spr_facetids = sim_options.get("springfacets", [])

    if not (k_spring and c_damping and spr_facetids):
        return None

    # Per-surface spring/dashpot lists are zipped; callers must keep lengths aligned.
    normal_proj = df.outer(ctx.reference_normal, ctx.reference_normal)
    form = 0
    for k_s, c_d, spr_f in zip(k_spring, c_damping, spr_facetids):
        k_pair = _ensure_pair(k_s)
        c_pair = _ensure_pair(c_d)
        surf_form = _spring_surface(
            ctx.measure(spr_f),
            normal_proj,
            k_pair,
            c_pair,
            ctx.displacement,
            ctx.displacement - ctx.previous_displacement,
            ctx.test_function,
            ctx.identity,
        )
        if surf_form is not None:
            form += surf_form

    return form
