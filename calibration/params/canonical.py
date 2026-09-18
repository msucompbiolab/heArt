"""Single authoritative source for every LV and FCH calibration parameter.

Every parameter the calibration / simulation pipelines use is defined here ONCE. All
consumers — the legacy FE configs (`orchestrate/lv_baseline_config.py` for LV,
`orchestrate/base_config.py` for FCH), the active/passive stages, the policy/schedule,
and the FE emitters (via the stage adapters' CLI) — read their values from here.
Nothing else defines, defaults, or overrides these values; there is no second place
to look.

Groups (siblings):
- `LV_*` / `*_PROTOCOL` / `LOADING` / `PRESSURE_WAVEFORM` — the LV pipeline.
- `FCH_ACTIVE_TMAX` / `FCH_PASSIVE_CPARAM` / `FCH_VOLUMES` / `FCH_PROTOCOL` — the FCH
  per-chamber twitch/passive SAMPLING band the FE surrogate emitters use.
- `FCH_BASE_*` — the FCH four-chamber CLOSED-LOOP OPERATING point that
  `orchestrate/base_config.py` reads (mirrors how `lv_baseline_config` reads `LV_*`):
  per-chamber absolute Cparam/Tmax, the fallback Guccione/Burkhoff substrate, the
  mechanics support spring, the mesh scale, the cycle scalars, and the region/facet
  ids. `apply_fch_material(SimDet, ...)` writes this substrate onto a SimDet (the FCH
  analogue of `lv_spring_strain.apply_material_baseline`).

Each entry is a record:
    {"value": <v>, "units": <str>, "tunable": <bool>, "range": [lo, hi] | None,
     "note": <str>}

- `tunable=True`  : the calibration search/fit is allowed to vary this; `range`
                    bounds the search. The value here is the starting point/default.
- `tunable=False` : fixed model input; the calibration never changes it.

Use `values(GROUP)` to get a plain `{name: value}` dict for a consumer, or
`flat()` / `provenance()` for the full audit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# --- LV passive material: reduced Guccione W = (C/2)(exp(QQ)-1) -----------------
LV_PASSIVE: Dict[str, Dict[str, Any]] = {
    "Cparam": {"value": 100.0, "units": "Pa", "tunable": False, "range": [20.0, 400.0],
               "note": "overall EDPVR stiffness scale — FIXED from the Klotz MFF unload (LV-only mode); not searched"},
    "bff":    {"value": 29.0,  "units": "1",  "tunable": False, "range": None,
               "note": "fiber-direction exponent"},
    "bfx":    {"value": 13.3,  "units": "1",  "tunable": False, "range": None,
               "note": "fiber-sheet shear exponent"},
    "bxx":    {"value": 26.6,  "units": "1",  "tunable": False, "range": None,
               "note": "cross-fiber exponent"},
    "Kappa":  {"value": 1.0e5, "units": "Pa", "tunable": False, "range": None,
               "note": "incompressibility penalty"},
}

# --- Holzapfel-Ogden 2009 (10.1098/rsta.2009.0091) orthotropic passive law ------
# SOTA alternative to Guccione, with a STIFF fiber term (a_f) decoupled from the soft bulk (a)
# — this decoupling is what lets active-strain develop physiological systolic tension (the
# single soft Guccione Cparam starves it). OPT-IN: selected via SimDet Passive model =
# "HolzapfelOgden" (default stays Guccione). a-coefficients converted kPa->Pa (repo Pa
# convention); b's dimensionless. Values are the validated class/benchmark defaults
# (HolzapfelOgden uniaxial matched the analytic benchmark to 0.000%). CAVEAT: the isotropic
# exponent b=0.023 is the heArt_py3 class/benchmark default; the canonical HO-2009 Table-1
# human value is widely cited as b~8.023 — VERIFY against the paper (institutional gateway)
# and CALIBRATE the exponents to the case03 EDPVR before physiological use (a Stage-C /
# grounding-dossier item, not an implementation blocker).
HO_PASSIVE: Dict[str, Dict[str, Any]] = {
    "a":    {"value": 59.0,    "units": "Pa", "tunable": True,  "range": [10.0, 500.0],
             "note": "isotropic ground-state stiffness (HO 2009 a=0.059 kPa)"},
    "b":    {"value": 8.023,   "units": "1",  "tunable": True,  "range": [0.01, 12.0],
             "note": "isotropic exponent (HO-2009 Table 1 b=8.023; heArt_py3 default had a dropped-digit 0.023)"},
    "a_f":  {"value": 18472.0, "units": "Pa", "tunable": True,  "range": [2000.0, 40000.0],
             "note": "FIBER stiffness (HO 2009 a_f=18.472 kPa) — carries active-strain tension"},
    "b_f":  {"value": 16.026,  "units": "1",  "tunable": True,  "range": [2.0, 30.0],
             "note": "fiber exponent"},
    "a_s":  {"value": 2481.0,  "units": "Pa", "tunable": True,  "range": [200.0, 8000.0],
             "note": "sheet stiffness (HO 2009 a_s=2.481 kPa)"},
    "b_s":  {"value": 11.120,  "units": "1",  "tunable": True,  "range": [2.0, 25.0],
             "note": "sheet exponent"},
    "a_fs": {"value": 216.0,   "units": "Pa", "tunable": True,  "range": [20.0, 1000.0],
             "note": "fiber-sheet cross stiffness (HO 2009 a_fs=0.216 kPa)"},
    "b_fs": {"value": 11.436,  "units": "1",  "tunable": True,  "range": [2.0, 25.0],
             "note": "fiber-sheet exponent"},
    "Kappa":{"value": 1.0e5,   "units": "Pa", "tunable": False, "range": None,
             "note": "incompressibility penalty (shared with Guccione path)"},
}

# The four HO a-coefficients are the ONE scalar stiffness DOF the HO paths fit. HO's strain energy
# is LINEAR in them, so scaling all four scales the passive stress uniformly (leaving the b
# exponents, and hence the EDPVR shape, untouched). Both HO consumers use this single definition:
#   * `unloading lv --params-json {"unload.passive_model":"HolzapfelOgden"}` fits the scale to the
#     Klotz V0 (see unloading/CLAUDE.md -- MEASURED not to be reachable at canonical exponents);
#   * the ED/ES stage fits it so the ED OPERATING POINT is reproduced on whatever reference it was
#     handed, which is the minimum consistency requirement when the reference was unloaded under a
#     DIFFERENT law (Guccione) -- inflating that reference with HO must still deliver the same
#     (EDV, EDP). See research_dossier ed_es_active_strain_ho@lv.
HO_A_KEYS = ("a", "a_f", "a_s", "a_fs")
HO_B_KEYS = ("b", "b_f", "b_s", "b_fs")


def ho_passive_block(scale: float = 1.0) -> Dict[str, float]:
    """Canonical Holzapfel-Ogden passive params with all four a-coefficients multiplied by
    `scale` (the 1-DOF stiffness DOF) and the b-exponents at their canonical values."""
    ho = values(HO_PASSIVE)
    block = {k: float(ho[k]) * float(scale) for k in HO_A_KEYS}
    block.update({k: float(ho[k]) for k in HO_B_KEYS})
    return block

# --- LV active material: Burkhoff time-varying elastance ------------------------
# PHYSIOLOGICAL-PLAUSIBILITY band for a FITTED LV Tmax, i.e. the range within which a converged
# ED/ES single-beat fit is accepted as physically meaningful. RECALIBRATED 2026-07 from [60,150] kPa
# on cohort + literature evidence; grounded in research_dossier ed_es_tmax_plausibility_band@lv.
#
# It is a VALIDITY band (does this fit describe human myocardium at some contractile state?), NOT a
# diagnostic band (is this heart normal?). The distinction is the whole point: contractility is the
# quantity being MEASURED, so a band narrow enough to encode "normal" rejects exactly the reduced-
# contractility patients the pipeline exists to characterize.
#
# Literature, both in THIS regime (patient-specific/idealized human LV FE, Guccione-family active
# stress with fiber-length dependence, closed-loop 0D, single-beat-EDPVR unloaded reference):
#   * Shavik 2021 (10.1007/s12265-021-10130-y) Table 6 -- active tension Tref: NORMAL 130 kPa;
#     HF cases 109 / 100 / 91 / 70 / 52 kPa, i.e. down to 40% of normal. Tier 2, VERIFIED.
#   * Mojumder 2023 (10.1038/s41598-023-28037-w) -- peak myofiber tension fitted to clinical data:
#     60 kPa obstructive HCM, 242 kPa non-obstructive HCM, 375 kPa control; disease at 16% of the
#     control value. Tier 2, VERIFIED.
# So the literature spans 52-375 kPa across contractile states, and the old 60 kPa floor excluded
# even Shavik's own 52 kPa HF case while the old 150 kPa ceiling excluded two of Mojumder's three
# subjects. The 20 kPa floor is an INFERENCE (labeled as such): 20/130 ~ 15% of Shavik's normal,
# matching the 16% disease scaling Mojumder measured, and this cohort's lowest fits sit at
# EF 3.5-12% -- sicker than either paper's patients.
TMAX_PLAUSIBLE_LO_PA = 20000.0
TMAX_PLAUSIBLE_HI_PA = 400000.0

LV_ACTIVE: Dict[str, Dict[str, Any]] = {
    "Tmax":   {"value": 100000.0, "units": "Pa", "tunable": False,
               "range": [TMAX_PLAUSIBLE_LO_PA, TMAX_PLAUSIBLE_HI_PA],
               "note": "contractility — FIXED from the ED/ES single-beat (LV-only mode); not searched. "
                       "100 kPa is the operator-validated full-cycle default. RANGE RECALIBRATED "
                       "2026-07 to [20,400] kPa (was [60,150]) -- see TMAX_PLAUSIBLE_LO_PA above for "
                       "the Shavik-2021 / Mojumder-2023 citations. Measured on the 178-case DICOM "
                       "cohort: Tmax tracks contractile state (Pearson r(EF,Tmax)=+0.79, n=86; median "
                       "by EF bin 50/51/59/70/88/128 kPa from EF<20% to EF>60%), so a narrow band is "
                       "the wrong instrument -- all 22 sub-60-kPa fits were low-EF patients (8/9 at "
                       "EF<20%) and none exceeded 150 kPa. FE-conditioning caveat retained: coarse/"
                       "degenerate meshes lose ellipticity earlier and cannot reach a high Tmax, which "
                       "is a SOLVER limit reported as ACTIVE_STRESS_CEILING, not a plausibility limit"},
    "l0":     {"value": 1.5,  "units": "um", "tunable": False, "range": None,
               "note": "slack sarcomere length; dead-zone threshold lambda_f = l0/lr"},
    "lr":     {"value": 1.85, "units": "um", "tunable": False, "range": None},
    "t0":     {"value": 275.0, "units": "ms", "tunable": False, "range": None, "note": "time-to-peak"},
    "t_trans":{"value": 300.0, "units": "ms", "tunable": False, "range": None},
    "tau":    {"value": 25.0,  "units": "ms", "tunable": False, "range": None, "note": "relaxation"},
    "B":      {"value": 4.75,  "units": "1",  "tunable": False, "range": None},
    "Ca0":    {"value": 4.35,  "units": "1",  "tunable": False, "range": None},
    "Ca0max": {"value": 4.35,  "units": "1",  "tunable": False, "range": None},
    "l_range_buffer": {"value": 0.002, "units": "um", "tunable": False, "range": None,
                       "note": "softplus buffer around the length-tension dead zone"},
}

# --- LV mechanics support: "pericardial sliding" spring BC ----------------------
# Epicardium: absolute [k_n, k_t]; base: tangential-only (normal DOF free, preserves
# longitudinal shortening / MAPSE). springparam is the only effective stiffness.
LV_SUPPORT: Dict[str, Dict[str, Any]] = {
    "springparam":  {"value": [4000.0, 2000.0], "units": "Pa/mm", "tunable": True,
                     "range": [[2000.0, 16000.0], [1000.0, 8000.0]],
                     "note": "epicardial [k_n, k_t]. HALVED 2026-06 from [8000,4000] (operator): a softer "
                             "pericardial tether deflates the unloaded reference further (V0 92.4->82.4 mL @ EDP=12) "
                             "in the Robin-support range. NB previously a spring_tune-CALIBRATED value -- the LV "
                             "strain/twist calibration + forward physiology should be re-validated at the new k."},
    "spring_basal": {"value": [0.0, 1000.0], "units": "Pa/mm", "tunable": True,
                     "range": [[0.0, 0.0], [0.0, 6000.0]],
                     "note": "base [k_n, k_t]; k_n pinned 0 (free long-axis), tangential-only. HALVED 2026-06 "
                             "from [0,2000] (operator), consistent with the epicardial halving."},
    "dashpotparam": {"value": [200.0, 20.0], "units": "Pa.ms/mm", "tunable": False, "range": None,
                     "note": "[c_n, c_t]; negligible quasi-static"},
    "spring_unloaded_reference": {"value": True, "units": "bool", "tunable": False, "range": None,
                     "note": "spring active at the ED operating point (measured from reference mesh)"},
}

# --- Active-twitch calibration protocol: V0->EDV isovolumic sweep ---------------
ACTIVE_PROTOCOL: Dict[str, Dict[str, Any]] = {
    "num_volumes":         {"value": 6, "units": "count", "tunable": False, "range": [3, 12],
                            "note": "evenly spaced volumes over [V0, EDV]"},
    "peak_activation_cap": {"value": 0.95, "units": "fraction", "tunable": False, "range": [0.3, 0.99],
                            "note": "activate-to-peak FE-solve cap"},
    "continuation_dt_ms":  {"value": 4.0, "units": "ms", "tunable": False, "range": [0.5, 20.0],
                            "note": "twitch continuation step; also sets the time-bisection floor (0.1x). "
                                    "4 ms reaches full activation here; coarser limit-points early"},
    "viscous_eta":         {"value": 500.0, "units": "Pa.ms", "tunable": False, "range": [0.0, 5000.0],
                            "note": "Kelvin-Voigt regularization removing the late-systolic zero pivot"},
    "activation_time_ms":  {"value": 10.0, "units": "ms", "tunable": False, "range": None,
                            "note": "homogeneous twitch onset"},
    "fe_jobs":             {"value": 1, "units": "count", "tunable": False, "range": [1, 64],
                            "note": "parallel volume-sweep workers; 1 = serial (unchanged). >1 fans "
                                    "each volume to an independent np=1 worker process, then aggregates"},
    "sample_mode":         {"value": "isovolumic", "units": "enum", "tunable": False, "range": None,
                            "note": "active-twitch FE sampling mode: 'isovolumic' (default; the "
                                    "volume-driven V0->EDV secant-held sweep above) or "
                                    "'pressure_waveform' (the pressure-driven variably-afterloaded "
                                    "beat family in PRESSURE_WAVEFORM -- no inner pressure root-find). "
                                    "Both fit the SAME 4-param ESPVR surrogate from the SAME CSV schema"},
}

# --- Pressure-driven active-twitch sampling (alternative to the isovolumic sweep) -
# Optional mode (ACTIVE_PROTOCOL.sample_mode == "pressure_waveform"): instead of
# holding prescribed volumes via a secant pressure root-find, prescribe a family of LV
# pressure waveforms P(t) (one per afterload level) and let cavity volume EVOLVE -- a
# single Newton solve per step, NO inner root-find. Each beat shares the same activation
# time-course and differs only in its plateau (afterload) pressure; the family is the
# classic variably-afterloaded ESPVR construction (spreads end-systolic volume across the
# operating range so Ees/Ecurv/V0_active stay identifiable). Afterloads are fractions of
# P_iso_peak (the cavity pressure that holds EDV at the capped peak activation, probed
# once). Viscosity note: the Kelvin-Voigt term is 2*eta*(E-E_old)/dt with E_old the
# (never-updated) unloaded reference and dt fixed at SimDet["dt"] -- a pseudo-elastic
# anchor independent of the march step, treated identically to the isovolumic path, so
# P_active = P_total - P_passive(V) is on the same footing (no quasi-static knob needed).
PRESSURE_WAVEFORM: Dict[str, Dict[str, Any]] = {
    "num_afterloads":      {"value": 6, "units": "count", "tunable": False, "range": [3, 12],
                            "note": "number of afterload (plateau-pressure) beats in the family"},
    "afterload_fractions": {"value": "0.50,0.60,0.70,0.80,0.90,0.95", "units": "fraction",
                            "tunable": False, "range": None,
                            "note": "plateau pressures as fractions of P_iso_peak. Biased HIGH "
                                    "(near-isovolumic) on purpose: only there is end-systolic volume "
                                    "afterload-SENSITIVE (so Ees/Ecurv are identifiable). Far below "
                                    "P_iso_peak the LV over-ejects to the length-tension dead-zone "
                                    "volume regardless of afterload (Ves clusters -> uninformative). "
                                    "Empty -> evenly spaced num_afterloads over [floor, 0.95]"},
    "afterload_floor_frac":{"value": 0.50, "units": "fraction", "tunable": False, "range": [0.05, 0.95],
                            "note": "minimum afterload fraction; below it the LV over-ejects into the "
                                    "dead-zone collapse (afterload-insensitive Ves, passive-table "
                                    "clamping, eventual ellipticity-loss fold) -- never prescribe lower"},
    "march_dt_ms":         {"value": 25.0, "units": "ms", "tunable": False, "range": [2.0, 100.0],
                            "note": "activation-clock step on the rise (sampling density + Newton "
                                    "step). Adaptive time-bisection retries a failed step down to "
                                    "0.1x. NOT a viscous lever (see group note)"},
    "march_dp_mmhg":       {"value": 1.0, "units": "mmHg", "tunable": False, "range": [0.1, 10.0],
                            "note": "pressure-continuation sub-step within each activation step "
                                    "(Newton robustness reaching the envelope pressure)"},
    "passive_suction_floor_mmhg": {"value": -10.0, "units": "mmHg", "tunable": False, "range": [-40.0, 0.0],
                            "note": "lower pressure bound of the active-off passive P(V) table so it "
                                    "covers ejection volumes below V0 (P_active baseline source)"},
    "march_through_relaxation": {"value": False, "units": "bool", "tunable": False, "range": None,
                            "note": "also march past peak into relaxation (waveform-shape validation "
                                    "only; the 4 ESPVR params need only the rise-to-peak)"},
    "fe_jobs":             {"value": 1, "units": "count", "tunable": False, "range": [1, 64],
                            "note": "parallel workers; 1 = serial. >1 fans each afterload beat to an "
                                    "independent np=1 worker process, then aggregates (mirrors the "
                                    "isovolumic fan-out)"},
}

# --- Loading / cycle -------------------------------------------------------------
LOADING: Dict[str, Dict[str, Any]] = {
    "EDP_mmhg":      {"value": 12.0, "units": "mmHg", "tunable": False, "range": [2.0, 30.0],
                      "note": "the single end-diastolic pressure: defines EDV / the V0->EDV active "
                              "sweep top AND the closed-loop preload. SET 2026-06 to 12 (was 10) for ONE "
                              "consistent EDP across unload+loading+ed_es (the unload EDP MUST match the loading "
                              "EDP or the recovered reference won't reproduce ED); 12 mmHg = upper-normal LV EDP."},
    "heartbeat_ms":  {"value": 800.0, "units": "ms", "tunable": False, "range": None},
}

# --- FCH (four-chamber) per-chamber materials + protocol -------------------------
# Absolute per-chamber materials (no scales); the single source mirroring the FCH
# emitters (fe_{active_twitch,passive_pv}_fch_legacy DEFAULT_TMAX_PA / DEFAULT_CPARAM).
# LV/RV are ventricular (Burkhoff ventricular timing); LA/RA are atrial (atrial timing).
FCH_ACTIVE_TMAX: Dict[str, Dict[str, Any]] = {
    "lv": {"value": 480000.0, "units": "Pa", "tunable": False, "range": [200000.0, 700000.0], "note": "LV contractility — FIXED from the ED/ES single-beat (derived_tmax.json); not searched"},
    "rv": {"value": 400000.0, "units": "Pa", "tunable": True, "range": [150000.0, 600000.0], "note": "RV (thin wall -> higher Tmax per unit pressure)"},
    "la": {"value": 200000.0, "units": "Pa", "tunable": True, "range": [50000.0, 350000.0], "note": "LA (small a-wave)"},
    "ra": {"value": 200000.0, "units": "Pa", "tunable": True, "range": [50000.0, 350000.0], "note": "RA"},
}
FCH_PASSIVE_CPARAM: Dict[str, Dict[str, Any]] = {
    "lv": {"value": 130.0, "units": "Pa", "tunable": False, "range": [40.0, 400.0], "note": "LV Guccione stiffness — FIXED from the Klotz MFF unload (unload sidecar); not searched"},
    "rv": {"value": 325.0, "units": "Pa", "tunable": True, "range": [80.0, 600.0], "note": "RV stiffer"},
    "la": {"value": 90.0, "units": "Pa", "tunable": True, "range": [30.0, 300.0], "note": "LA reservoir compliance"},
    "ra": {"value": 90.0, "units": "Pa", "tunable": True, "range": [30.0, 300.0], "note": "RA"},
}
FCH_VOLUMES: Dict[str, Dict[str, Any]] = {
    "lv": {"value": "70,95,120,145,170,195", "units": "mL", "tunable": False, "range": None, "note": "per-chamber V0->EDV twitch band"},
    "rv": {"value": "46,68,90,112,134,156", "units": "mL", "tunable": False, "range": None},
    "la": {"value": "32,40,48,56,64", "units": "mL", "tunable": False, "range": None},
    "ra": {"value": "18,25,32,39,46", "units": "mL", "tunable": False, "range": None},
}
FCH_PROTOCOL: Dict[str, Dict[str, Any]] = {
    "peak_activation_cap":   {"value": 0.9, "units": "fraction", "tunable": False, "range": [0.3, 0.99],
                              "note": "FCH activate-to-peak FE-solve cap (near-peak RV/atrial isovolumic is hardest)"},
    "continuation_dt_ms":    {"value": 4.0, "units": "ms", "tunable": False, "range": [0.5, 20.0]},
    "viscous_eta":           {"value": 500.0, "units": "Pa.ms", "tunable": False, "range": [0.0, 5000.0]},
    "hold_pressure_mmhg":    {"value": 2.0, "units": "mmHg", "tunable": False, "range": [0.0, 10.0],
                              "note": "small diastolic pressure on the 3 inactive chambers (in-situ coupling)"},
    "preload_pressure_mmhg": {"value": 10.0, "units": "mmHg", "tunable": False, "range": [2.0, 30.0]},
    "activation_time_ms":    {"value": 10.0, "units": "ms", "tunable": False, "range": None},
    "fe_jobs":               {"value": 1, "units": "count", "tunable": False, "range": [1, 64],
                              "note": "parallel FE workers (active twitch: chamber x volume; "
                                      "passive PV: per-chamber); 1 = serial"},
}

# --- FCH (four-chamber) CLOSED-LOOP baseline ------------------------------------
# THE single source for orchestrate/base_config (mirrors lv_baseline_config reading
# from the LV_* groups). These are the four-chamber CLOSED-LOOP OPERATING values:
# the default full-cycle run (demo/case24 -> run_light) and every FCH unloading /
# simulation / spring consumer that reads base_config. Each value equals the
# historical base_config literal EXACTLY (behavior-preserving migration).
#
# Relationship to the FCH_* groups above: FCH_ACTIVE_TMAX / FCH_PASSIVE_CPARAM /
# FCH_VOLUMES / FCH_PROTOCOL are the per-chamber TWITCH/PASSIVE *sampling band* the
# FE surrogate emitters use; the groups below are the closed-loop *operating point*.
# Per-chamber Tmax and the LV/LA/RA passive Cparam agree across both contexts; the
# RV passive Cparam deliberately DIFFERS -- operating 40 Pa (soft thin RV free wall,
# unloaded-reference convention) vs sampling 325 Pa. They are kept as separate
# records so neither context silently changes the other; reconciling them is an
# operator decision (it re-fits the surrogate). Within each context a region resolves
# to exactly ONE absolute Cparam/Tmax from here.

# Base ("fallback") Guccione substrate + per-chamber ABSOLUTE passive Cparam.
# ``Cparam`` is the SINGLE DOCUMENTED FALLBACK coefficient: a mesh region with no
# per-chamber ``Cparam_{ch}`` override resolves to it. The Guccione SEF is linear in
# Cparam, so a chamber's absolute value enters as the scale Cparam_region/Cparam
# applied to the one SEF built at the fallback -- no region reads a bare "base".
FCH_BASE_PASSIVE: Dict[str, Dict[str, Any]] = {
    "Cparam":    {"value": 100.0, "units": "Pa", "tunable": False, "range": None,
                  "note": "fallback Guccione stiffness: a region without a per-chamber "
                          "Cparam_{ch} override resolves to this (LV-consistent substrate)"},
    "bff":       {"value": 29.0,  "units": "1",  "tunable": False, "range": None,
                  "note": "fiber-direction exponent (global; not per-chamber -- sits inside exp(Q))"},
    "bfx":       {"value": 13.3,  "units": "1",  "tunable": False, "range": None, "note": "fiber-sheet shear exponent"},
    "bxx":       {"value": 26.6,  "units": "1",  "tunable": False, "range": None, "note": "cross-fiber exponent"},
    "eta":       {"value": 50.0,  "units": "Pa.ms", "tunable": False, "range": None,
                  "note": "passive Kelvin-Voigt viscosity (engaged only when _passive_viscous_)"},
    "Kappa":     {"value": 1.0e5, "units": "Pa", "tunable": False, "range": None, "note": "incompressibility penalty"},
    "Cparam_lv": {"value": 130.0, "units": "Pa", "tunable": True, "range": [40.0, 400.0], "note": "LV operating Guccione stiffness"},
    "Cparam_rv": {"value": 40.0,  "units": "Pa", "tunable": True, "range": [20.0, 400.0],
                  "note": "RV operating stiffness: soft thin RV free wall (more compliant than LV "
                          "with the unloaded reference). Operating value; the twitch SAMPLING band "
                          "(FCH_PASSIVE_CPARAM.rv) is a separate 325 Pa record"},
    "Cparam_la": {"value": 90.0,  "units": "Pa", "tunable": True, "range": [30.0, 300.0], "note": "LA reservoir compliance"},
    "Cparam_ra": {"value": 90.0,  "units": "Pa", "tunable": True, "range": [30.0, 300.0], "note": "RA"},
}

# Base ("fallback") Burkhoff time-varying active material + per-chamber ABSOLUTE Tmax.
# ``Tmax`` is the SINGLE DOCUMENTED FALLBACK contractility (a region without a
# per-chamber ``Tmax_{ch}`` override resolves to it). The Burkhoff active stress is
# linear in Tmax, so a chamber's absolute value enters as the scale Tmax_region/Tmax
# on the base form -- the same single-coefficient convention as the passive group.
# Ventricular (LV/RV) and atrial (LA/RA) share these timings except the atrial
# overrides (*_atr / tdelay_atr). Plain int values are intentional (matched to the
# historical base_config literal types).
FCH_BASE_ACTIVE: Dict[str, Dict[str, Any]] = {
    "Tmax":       {"value": 1250000.0, "units": "Pa", "tunable": False, "range": None,
                   "note": "fallback contractility (== legacy contRactility); the per-chamber "
                           "Tmax_{ch} below are the effective absolute contractilities"},
    "tau":        {"value": 35,    "units": "ms", "tunable": False, "range": None,
                   "note": "ventricular relaxation time constant (clinical isovolumic-relaxation tau ~30-45 ms)"},
    "tau_atr":    {"value": 24,    "units": "ms", "tunable": False, "range": None, "note": "atrial relaxation (faster than ventricle)"},
    "t_trans":    {"value": 290,   "units": "ms", "tunable": False, "range": None},
    "t_trans_atr":{"value": 105.0, "units": "ms", "tunable": False, "range": None},
    "B":          {"value": 5.0,   "units": "1",  "tunable": False, "range": None},
    "t0":         {"value": 260,   "units": "ms", "tunable": False, "range": None, "note": "time-to-peak"},
    "t0_atr":     {"value": 70.0,  "units": "ms", "tunable": False, "range": None},
    "tdelay_atr": {"value": 660.0, "units": "ms", "tunable": False, "range": None, "note": "atrial activation delay (closed-loop)"},
    "l0":         {"value": 1.50,  "units": "um", "tunable": False, "range": None,
                   "note": "length-tension slack length (low end of physiological range)"},
    "Ca0":        {"value": 4.35,  "units": "1",  "tunable": False, "range": None},
    "Ca0max":     {"value": 4.35,  "units": "1",  "tunable": False, "range": None},
    "lr":         {"value": 1.85,  "units": "um", "tunable": False, "range": None},
    "Tmax_lv":    {"value": 480000.0, "units": "Pa", "tunable": True, "range": [200000.0, 700000.0], "note": "LV contractility"},
    "Tmax_rv":    {"value": 400000.0, "units": "Pa", "tunable": True, "range": [150000.0, 600000.0], "note": "RV (thin wall -> higher Tmax per unit pressure)"},
    "Tmax_la":    {"value": 200000.0, "units": "Pa", "tunable": True, "range": [50000.0, 350000.0], "note": "LA"},
    "Tmax_ra":    {"value": 200000.0, "units": "Pa", "tunable": True, "range": [50000.0, 350000.0], "note": "RA"},
}

# Mechanics support (pericardial spring) -- per-region absolute [k_n, k_t]
# from the WS3 four-chamber spring calibration (passive kinematics sweep, best converged
# config: softer epi both scores best and fully inflates). springparam_{epi,atrial} are the
# effective per-surface stiffness (read by spring_bc_forms._build_fch_form); the single
# `springparam` is the legacy fallback for any surface without a per-region override.
FCH_BASE_SUPPORT: Dict[str, Dict[str, Any]] = {
    "springparam_epi":    {"value": [4000.0, 4000.0], "units": "Pa/mm", "tunable": True,
                           "range": [[2000.0, 32000.0], [1000.0, 4000.0]],
                           "note": "ventricular epicardium [k_n, k_t]; WS3-validated"},
    "springparam_atrial": {"value": [2000.0, 1000.0], "units": "Pa/mm", "tunable": True,
                           "range": [[1000.0, 4000.0], [500.0, 2000.0]],
                           "note": "atrial wall [k_n, k_t]; WS3-validated"},
    "springparam":  {"value": [4000.0, 4000.0], "units": "Pa/mm", "tunable": True,
                     "range": [[2000.0, 16000.0], [1000.0, 8000.0]],
                     "note": "legacy single-surface fallback [k_n, k_t] (= the epi pair)"},
    "dashpotparam": {"value": [200.0, 20.0], "units": "Pa.ms/mm", "tunable": False, "range": None,
                     "note": "[c_n, c_t]; negligible in the quasi-static sweep"},
}

# Mesh + cycle scalars. The canonical FCH mesh is the big-RV Strocchi remesh
# (meshes/fch_remesh/fch_clregion_whole.hdf5), used at scale 0.1 (mm->cm so cavity volumes
# are mL): LV 169.6 / RV 234.2 / LA 89.3 / RA 192.5 mL, RV/LV 1.38.
FCH_BASE_GEOMETRY: Dict[str, Dict[str, Any]] = {
    "mesh_scale_fch": {"value": 1.0e-1, "units": "1", "tunable": False, "range": [5.0e-2, 1.0e-1],
                       "note": "FCH mechanics mesh scale; Strocchi remesh is mm -> 0.1 gives mL; "
                               "EF/ratio scale-invariant, sets absolute EDV magnitude"},
    "mesh_subdir":    {"value": "meshes/fch_remesh", "units": "path", "tunable": False, "range": None,
                       "note": "repo-relative dir holding the canonical FCH mesh (+ _refine)"},
    "mesh_basename":  {"value": "fch_clregion_whole", "units": "name", "tunable": False, "range": None,
                       "note": "HDF5 basename (group fch_clregion); EP reads {casename}_refine.hdf5"},
}
FCH_BASE_CYCLE: Dict[str, Dict[str, Any]] = {
    "HeartBeatLength": {"value": 800.0, "units": "ms", "tunable": False, "range": None},
    "dt":              {"value": 0.5,   "units": "ms", "tunable": False, "range": None, "note": "closed-loop time step"},
    "writeStep":       {"value": 40.0,  "units": "step", "tunable": False, "range": None},
    "nLoadSteps":      {"value": 32,    "units": "count", "tunable": False, "range": None, "note": "diastolic loading increments"},
}

# Mesh region / facet ids (mesh-specific contract; fixed model inputs, not tunable).
FCH_BASE_REGIONS: Dict[str, Dict[str, Any]] = {
    "aorta_wall": {"value": 7,  "units": "id", "tunable": False, "range": None},
    "LVendoid":   {"value": 1,  "units": "id", "tunable": False, "range": None},
    "RVendoid":   {"value": 6,  "units": "id", "tunable": False, "range": None},
    "LAendoid":   {"value": 2,  "units": "id", "tunable": False, "range": None},
    "RAendoid":   {"value": 3,  "units": "id", "tunable": False, "range": None},
    "epiid":      {"value": 9,  "units": "id", "tunable": False, "range": None},
    "atrialid":   {"value": 10, "units": "id", "tunable": False, "range": None},
    "apxid":      {"value": 11, "units": "id", "tunable": False, "range": None},
    "aortaid":    {"value": 8,  "units": "id", "tunable": False, "range": None},
    "lv_rid":     {"value": 10, "units": "id", "tunable": False, "range": None, "note": "LV myocardial matid"},
    "rv_rid":     {"value": 9,  "units": "id", "tunable": False, "range": None, "note": "RV myocardial matid"},
    "la_rid":     {"value": 11, "units": "id", "tunable": False, "range": None, "note": "LA wall matid (== LAendoid-bounded wall)"},
    "ra_rid":     {"value": 8,  "units": "id", "tunable": False, "range": None, "note": "RA wall matid"},
    "active_region": {"value": [10, 9, 8, 11], "units": "id", "tunable": False, "range": None,
                      "note": "matids carrying active stress (la_rid/rv_rid/ra_rid/apxid order)"},
}

# FCH zero-pressure-reference (unloading) semantics. The bundled FCH mesh is a
# VENTRICULAR-ED segmentation: at ventricular ED the ventricles are at max volume
# (correctly unloaded from their EDP) but the atria are at ATRIAL END-SYSTOLE (near
# their minimum volume, still actively contracting) -- NOT a passive diastolic
# maximum. Unloading the atria from a positive atrial "EDP" therefore over-deflates
# the recovered atrial reference (V0 too small -> EDPVR too stiff -> impaired
# reservoir -> reduced ventricular preload). When ``atria_unstressed_ed`` is True the
# atria are treated as their OWN near-unstressed reference at the segmentation frame:
# the FCH unloader drives the atrial cavity pressures to ~0 (ventricles still unload
# from their true EDP), the forward loading phase holds the atria at ~0 (so the ED
# reference atrial shape == the unloaded shape and the round-trip closes), and the
# atrial passive V0 fit is anchored to the recovered unloaded cavity volume (clamped
# into the literature bands below). Default False == byte-identical legacy behavior.
FCH_BASE_UNLOAD: Dict[str, Dict[str, Any]] = {
    "atria_unstressed_ed": {"value": False, "units": "bool", "tunable": False, "range": None,
                            "note": "treat LA/RA as their own near-unstressed reference at the "
                                    "ventricular-ED frame (unload atria from a small near-unstressed "
                                    "pressure, hold loading atria at the same, anchor atrial V0 to "
                                    "unloaded cavity volume). Default False = legacy atrial-EDP unload "
                                    "(byte-identical)"},
    "atria_unstressed_pressure_mmhg": {"value": 4.0, "units": "mmHg", "tunable": False, "range": [3.0, 6.0],
                            "note": "small regularizing transmural pressure the atria unload from (and "
                                    "load to) under atria_unstressed_ed. NOT 0 and NOT <3: a thin atrial "
                                    "wall at low transmural pressure loses ellipticity on the evolving "
                                    "Sellier backward reference and the FE solve diverges -- HPCC sweep "
                                    "(jobs 10067755/10072702/10074123, big-RV remesh) DIVERGED at 0/1/2 "
                                    "mmHg even with lv-first-fine staging; 3 and 4 converge. 4.0 is the "
                                    "validated default: it corrects the LA over-deflation (LA 8->4 mmHg, "
                                    "V0 56->74 mL / -37%->-17%) while leaving RV+RA V0 invariant vs the "
                                    "legacy 8/4 unload (LV V0 shifts -4.8%, a real left AV-plane coupling). "
                                    "Going below 4 to also un-deflate RA excites the trans-septal RV "
                                    "near-indeterminacy (RV V0 227->211 at 3 mmHg) -- avoid on this mesh"},
    "atrial_v0_band_la":   {"value": [20.0, 40.0], "units": "mL", "tunable": False, "range": None,
                            "note": "physiological LA unstressed-volume band (anchor clamp + gate A1)"},
    "atrial_v0_band_ra":   {"value": [15.0, 30.0], "units": "mL", "tunable": False, "range": None,
                            "note": "physiological RA unstressed-volume band (anchor clamp + gate A1)"},
}

ED_ES: Dict[str, Dict[str, Any]] = {
    # Simplified ED/ES wall-stress -> contractility (Tmax) estimator (calibration/stages/ed_es_stress.py).
    # edv_ml/esv_ml are ECHO-derived (injected from the 2Decho4DReconstruction frame_phase_manifest.json:
    # ED=max-volume frame, ES=min-volume frame); the rest are levers/assumptions. The estimate is
    # traceable to these echo observables (recorded in the summary provenance).
    "edv_ml": {"value": 0.0, "units": "mL", "tunable": False, "range": [0.0, 500.0],
               "note": "echo end-diastolic volume (injected; 0 = required, fail loudly)"},
    "esv_ml": {"value": 0.0, "units": "mL", "tunable": False, "range": [0.0, 500.0],
               "note": "echo end-systolic volume (injected; 0 = required, fail loudly)"},
    "full_edv_ml": {
        "value": 0.0, "units": "mL", "tunable": False, "range": [0.0, 500.0],
        "note": "full-reconstruction EDV denominator for trimmed-LV ESV ratio mapping; "
                "0 = ESV is already in the consumed mesh's volume domain",
    },
    "sbp_mmhg": {"value": 120.0, "units": "mmHg", "tunable": False, "range": [60.0, 250.0],
                 "note": "brachial systolic cuff pressure (non-invasive)"},
    "esp_fraction": {"value": 0.9, "units": "1", "tunable": False, "range": [0.8, 1.0],
                     "note": "ESP = esp_fraction*SBP (Chen 2001 single-beat Ees: Pes ~= 0.9*SBP)"},
    "edp_mmhg": {"value": 12.0, "units": "mmHg", "tunable": False, "range": [2.0, 25.0],
                 "note": "assumed end-diastolic pressure (the one non-echo scalar; can be set from echo E/e'). "
                         "SET 2026-06 to 12 (was 8) to match the unload/loading EDP (one consistent EDP); upper-normal."},
    "tmax_lo": {"value": 60000.0, "units": "Pa", "tunable": False, "range": [1000.0, 1000000.0],
                "note": "Backward-compatible name for the adaptive Tmax SEED. The default 60 kPa "
                        "is a representative low historical seed; its measured pressure residual "
                        "chooses the direction and magnitude of the second probe."},
    "tmax_hi": {"value": 150000.0, "units": "Pa", "tunable": False, "range": [1000.0, 1000000.0],
                "note": "Backward-compatible legacy high value. The default 150 kPa is NOT probed "
                        "unconditionally; the adaptive search uses the 400 kPa validated coverage rail "
                        "and establishes a bracket from direction-oriented incremental probes."},
    "pressure_tol_mmhg": {"value": 1.0, "units": "mmHg", "tunable": False, "range": [0.1, 5.0],
                          "note": "ES holding-pressure match tolerance for the Tmax bisection"},
}

GROUPS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "lv_passive": LV_PASSIVE,
    "lv_active": LV_ACTIVE,
    "lv_support": LV_SUPPORT,
    "active_protocol": ACTIVE_PROTOCOL,
    "pressure_waveform": PRESSURE_WAVEFORM,
    "loading": LOADING,
    "fch_active_tmax": FCH_ACTIVE_TMAX,
    "fch_passive_cparam": FCH_PASSIVE_CPARAM,
    "fch_volumes": FCH_VOLUMES,
    "fch_protocol": FCH_PROTOCOL,
    "fch_base_passive": FCH_BASE_PASSIVE,
    "fch_base_active": FCH_BASE_ACTIVE,
    "fch_base_support": FCH_BASE_SUPPORT,
    "fch_base_geometry": FCH_BASE_GEOMETRY,
    "fch_base_cycle": FCH_BASE_CYCLE,
    "fch_base_regions": FCH_BASE_REGIONS,
    "fch_base_unload": FCH_BASE_UNLOAD,
    "ed_es": ED_ES,
}


def values(group: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Plain {name: value} dict for a consumer (e.g. values(LV_PASSIVE))."""
    return {name: rec["value"] for name, rec in group.items()}


def value(group: Dict[str, Dict[str, Any]], name: str) -> Any:
    """Single value lookup (raises KeyError if the parameter is not defined here)."""
    return group[name]["value"]


def flat() -> Dict[str, Any]:
    """Flattened {group.name: value} for logging / run manifests."""
    return {f"{g}.{n}": rec["value"] for g, grp in GROUPS.items() for n, rec in grp.items()}


def tunable() -> Dict[str, Dict[str, Any]]:
    """Every tunable parameter with its range, for the calibration search/fit."""
    return {f"{g}.{n}": {"value": rec["value"], "range": rec["range"]}
            for g, grp in GROUPS.items() for n, rec in grp.items() if rec.get("tunable")}


def provenance() -> Dict[str, Dict[str, Any]]:
    """Full record per parameter (value/units/tunable/range/note) for the audit."""
    return {f"{g}.{n}": dict(rec) for g, grp in GROUPS.items() for n, rec in grp.items()}


def apply_fch_material(SimDet, *, cparam_lv=None, cparam_rv=None, cparam_la=None,
                       cparam_ra=None, tmax_lv=None, tmax_rv=None, tmax_la=None,
                       tmax_ra=None):
    """Write the canonical FCH closed-loop myocardial substrate onto ``SimDet`` from the
    single source (FCH_BASE_PASSIVE / FCH_BASE_ACTIVE). Mirrors
    ``lv_spring_strain.apply_material_baseline`` so every FCH consumer (unloading /
    twitch / spring / simulation) sets materials from ONE place instead of re-deriving
    literals. Sets the fallback Guccione/Burkhoff substrate + the per-chamber ABSOLUTE
    ``Cparam_{ch}``/``Tmax_{ch}``; explicit kwargs override the per-chamber defaults.
    ``orchestrate.base_config`` produces an IDENTICAL material block; this re-applies it
    onto an existing SimDet (e.g. after other overrides). Returns ``SimDet``."""
    p = values(FCH_BASE_PASSIVE)
    a = values(FCH_BASE_ACTIVE)
    g = SimDet.setdefault("GiccioneParams", {})
    pp = g.setdefault("Passive params", {})
    pp["Cparam"] = p["Cparam"]
    pp["bff"] = p["bff"]
    pp["bfx"] = p["bfx"]
    pp["bxx"] = p["bxx"]
    pp["eta"] = p["eta"]
    g["Kappa"] = p["Kappa"]
    ap = g.setdefault("Active params", {})
    for k in ("Tmax", "tau", "tau_atr", "t_trans", "t_trans_atr", "B", "t0",
              "t0_atr", "tdelay_atr", "l0", "Ca0", "Ca0max", "lr"):
        ap[k] = a[k]
    src = {"Cparam_lv": p["Cparam_lv"], "Cparam_rv": p["Cparam_rv"],
           "Cparam_la": p["Cparam_la"], "Cparam_ra": p["Cparam_ra"],
           "Tmax_lv": a["Tmax_lv"], "Tmax_rv": a["Tmax_rv"],
           "Tmax_la": a["Tmax_la"], "Tmax_ra": a["Tmax_ra"]}
    over = {"Cparam_lv": cparam_lv, "Cparam_rv": cparam_rv, "Cparam_la": cparam_la,
            "Cparam_ra": cparam_ra, "Tmax_lv": tmax_lv, "Tmax_rv": tmax_rv,
            "Tmax_la": tmax_la, "Tmax_ra": tmax_ra}
    for k, ov in over.items():
        SimDet[k] = float(ov) if ov is not None else src[k]
    return SimDet


# --- Runtime-editable controls + the SINGLE override channel ---------------------
# The physically meaningful calibration knobs a user may override at runtime. Each
# maps to a canonical (group, key[, list-index]) so an edit writes back to the one
# source. `id` is the stable key used in the optional override JSON. Spring stiffness
# is exposed as four scalar controls mapping into the [k_n, k_t] lists.
#   ref = (group_dict, key, index_or_None)
EDITABLE: Dict[str, Dict[str, Any]] = {
    # active material
    "active.Tmax":    {"ref": (LV_ACTIVE, "Tmax", None),    "group": "active",  "label": "Contractility Tmax"},
    "active.t0":      {"ref": (LV_ACTIVE, "t0", None),      "group": "active",  "label": "Time-to-peak t0"},
    "active.t_trans": {"ref": (LV_ACTIVE, "t_trans", None), "group": "active",  "label": "Decay-onset t_trans"},
    "active.tau":     {"ref": (LV_ACTIVE, "tau", None),     "group": "active",  "label": "Relaxation tau"},
    # passive material
    "passive.Cparam": {"ref": (LV_PASSIVE, "Cparam", None), "group": "passive", "label": "Stiffness Cparam"},
    "passive.bff":    {"ref": (LV_PASSIVE, "bff", None),    "group": "passive", "label": "Fiber exponent bff"},
    "passive.bfx":    {"ref": (LV_PASSIVE, "bfx", None),    "group": "passive", "label": "Fiber-sheet exponent bfx"},
    "passive.bxx":    {"ref": (LV_PASSIVE, "bxx", None),    "group": "passive", "label": "Cross-fiber exponent bxx"},
    # four-chamber PER-CHAMBER materials (the FCH override channel): absolute per-chamber passive
    # stiffness Cparam_{ch} and active contractility Tmax_{ch}. Range-checked against the canonical
    # FCH group ranges. FIXED-INPUT architecture: LV is upstream-sourced (tunable=False) — Cparam_lv
    # from the Klotz MFF unload, Tmax_lv from the ED/ES single-beat — and these LV keys remain here
    # ONLY so the pipeline can INJECT those fixed upstream values via the override channel (they are
    # NOT search dimensions). RV/atria stay tunable (searched by loop_search).
    "material.Cparam_lv": {"ref": (FCH_PASSIVE_CPARAM, "lv", None), "group": "material", "label": "LV passive Cparam"},
    "material.Cparam_rv": {"ref": (FCH_PASSIVE_CPARAM, "rv", None), "group": "material", "label": "RV passive Cparam"},
    "material.Cparam_la": {"ref": (FCH_PASSIVE_CPARAM, "la", None), "group": "material", "label": "LA passive Cparam"},
    "material.Cparam_ra": {"ref": (FCH_PASSIVE_CPARAM, "ra", None), "group": "material", "label": "RA passive Cparam"},
    "material.Tmax_lv":   {"ref": (FCH_ACTIVE_TMAX, "lv", None),    "group": "material", "label": "LV active Tmax"},
    "material.Tmax_rv":   {"ref": (FCH_ACTIVE_TMAX, "rv", None),    "group": "material", "label": "RV active Tmax"},
    "material.Tmax_la":   {"ref": (FCH_ACTIVE_TMAX, "la", None),    "group": "material", "label": "LA active Tmax"},
    "material.Tmax_ra":   {"ref": (FCH_ACTIVE_TMAX, "ra", None),    "group": "material", "label": "RA active Tmax"},
    # spring / contact support (epi + basal, normal + tangential)
    "support.epi_k_normal":      {"ref": (LV_SUPPORT, "springparam", 0),  "group": "support", "label": "Epi normal stiffness k_n"},
    "support.epi_k_tangential":  {"ref": (LV_SUPPORT, "springparam", 1),  "group": "support", "label": "Epi tangential stiffness k_t"},
    "support.basal_k_normal":    {"ref": (LV_SUPPORT, "spring_basal", 0), "group": "support", "label": "Basal normal stiffness k_n"},
    "support.basal_k_tangential":{"ref": (LV_SUPPORT, "spring_basal", 1), "group": "support", "label": "Basal tangential stiffness k_t"},
    # loading
    "loading.edp_mmhg": {"ref": (LOADING, "EDP_mmhg", None), "group": "loading", "label": "End-diastolic pressure EDP"},
    # active protocol / solver knobs
    "protocol.continuation_dt_ms":  {"ref": (ACTIVE_PROTOCOL, "continuation_dt_ms", None),  "group": "protocol", "label": "Twitch continuation dt"},
    "protocol.viscous_eta":         {"ref": (ACTIVE_PROTOCOL, "viscous_eta", None),         "group": "protocol", "label": "Viscous regularization eta"},
    "protocol.num_volumes":         {"ref": (ACTIVE_PROTOCOL, "num_volumes", None),         "group": "protocol", "label": "V0->EDV sweep volume count"},
    "protocol.peak_activation_cap": {"ref": (ACTIVE_PROTOCOL, "peak_activation_cap", None), "group": "protocol", "label": "Activate-to-peak cap"},
}

# Single override channel: an optional JSON file path. Set by the CLI (`--params-json`)
# for the whole run AND exported to subprocesses via this env var, so every consumer
# that imports `canonical` (in-process or in a spawned emitter/fit/search) overlays
# the SAME overrides. There is no other override mechanism.
OVERRIDE_ENV = "CALIBRATION_PARAMS_OVERRIDE"


def _ref_get(grp: Dict[str, Any], key: str, idx):
    rec = grp[key]
    val, rng = rec["value"], rec.get("range")
    if idx is not None:
        val = val[idx]
        rng = rng[idx] if isinstance(rng, (list, tuple)) and rng and isinstance(rng[0], (list, tuple)) else rng
    return rec, val, rng


def editable_schema() -> list:
    """The runtime-editable control set for the override JSON (id/group/label/units/
    value/range). This is the authoritative list of what a user may override."""
    out = []
    for pid, meta in EDITABLE.items():
        grp, key, idx = meta["ref"]
        rec, val, rng = _ref_get(grp, key, idx)
        out.append({"id": pid, "group": meta["group"], "label": meta["label"],
                    "units": rec.get("units"), "value": val, "range": rng})
    return out


def apply_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay user overrides (id -> value) onto the canonical params in place. Only
    EDITABLE ids are accepted; values are range-checked. Returns the applied map."""
    applied: Dict[str, Any] = {}
    for pid, raw in (overrides or {}).items():
        if pid not in EDITABLE:
            raise KeyError(f"unknown override parameter '{pid}'. Editable ids: {sorted(EDITABLE)}")
        grp, key, idx = EDITABLE[pid]["ref"]
        rec, _cur, rng = _ref_get(grp, key, idx)
        v = float(raw)
        if isinstance(rng, (list, tuple)) and len(rng) == 2 and all(isinstance(x, (int, float)) for x in rng):
            if not (rng[0] <= v <= rng[1]):
                raise ValueError(f"override '{pid}'={v} outside range {rng}")
        if idx is None:
            rec["value"] = v
        else:
            rec["value"][idx] = v
        applied[pid] = v
    return applied


def param_overrides_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the parameter-override map from a unified override file. The file is
    either a flat ``{id: value}`` dict (parameters only) or a sectioned object with
    optional ``parameters`` / ``targets`` keys (targets are consumed by the objective,
    not here). Returns the ``{id: value}`` parameter map."""
    if isinstance(payload, dict) and ("parameters" in payload or "targets" in payload):
        return dict(payload.get("parameters", {}) or {})
    return dict(payload or {})


def _load_env_overrides() -> None:
    """Import-time hook: if the override-file env var points to a JSON, overlay it.
    No-op when unset/missing so default behavior (no override) is unchanged. This is
    how a run's single override propagates to every subprocess that imports canonical."""
    path = os.environ.get(OVERRIDE_ENV)
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        apply_overrides(param_overrides_from_payload(json.loads(p.read_text(encoding="utf-8"))))
    except Exception as exc:  # surface loudly; do not silently ignore a bad override
        print(f"WARNING: failed to apply {OVERRIDE_ENV}={path}: {exc}", file=sys.stderr)


_load_env_overrides()
