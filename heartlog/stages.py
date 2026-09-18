"""Closed, execution-ordered stage vocabulary per log source.

Stage names are enumerated rather than free text so the ``source/stage`` origin column is a
small, stable set a reader learns once, and so a typo surfaces as a warning instead of silently
minting a new origin. ``Logger.stage()`` warns once on an unknown name but still emits it --
narration is never blocked by bookkeeping.
"""
from __future__ import annotations

# Per-source stage tuples, in the order they execute.
STAGES: dict[str, tuple[str, ...]] = {
    # Forward FE run (run_light / run_waorta): construction -> loading -> coupled time loop.
    # "solve" is the generic bucket for narration that predates stage attribution and reaches the
    # logger through `printout` without naming a stage; prefer a specific stage in new code.
    "fe": ("init", "mesh", "loading", "plug_continuation", "coupling", "ep", "output", "finalize", "solve"),
    # Zero-pressure reference recovery (unloading/lv.py, unloading/fch.py).
    "unload": ("precondition", "anchor", "unloading", "mff", "diagnostic", "verify", "export"),
    # ED/ES single-beat driver (calibration/stages/ed_es_stress.py).
    "ed_es": ("input", "mesh_check", "anchor", "ed", "es", "summary"),
    # Parallel sweep drivers (scripts/fch_active_stability_sweep.py and siblings).
    "sweep": ("plan", "warm", "parallel", "collect"),
    # Isovolumic twitch benchmark (twitch_bench/).
    "bench": ("plan", "twitch", "collect"),
    # Autonomous control loops. These narrate their own watch/fix tempo rather than an FE stage.
    "autocore": ("frontier", "ledger", "deploy", "guard", "datalayer"),
    "autocal": ("watch", "dag", "stage", "gate", "fix", "ledger", "finalize", "surrogate_search"),
    "active_lab": ("watch", "tree", "node", "gate", "fix", "ledger", "provenance", "investigate"),
    "unload_lab": ("watch", "tree", "node", "gate", "fix", "ledger", "provenance", "fit"),
    "campaign": ("watch", "oracle", "fix", "ledger"),
    "unload_loop": ("watch", "dag", "stage", "gate", "ledger"),
    "lv_batch": ("manifest", "stage", "run", "aggregate"),
    # Mesh -> neural-network dataset conversion (nnprep/).
    "nnprep": ("convert", "export", "verify", "stats"),
    # Mesh -> GINO inference DAG (nninfer/). One stage name per DAG stage, plus `run` for the
    # whole-DAG driver, so a log line names the stage the gate will report on.
    "nninfer": ("run", "contract", "mesh", "record", "conditions", "embed", "infer",
                "reconstruct", "validate", "gate"),
    # Scientific-writing manuscript pipeline (scribe/): contract -> corpus harvest -> repro
    # check -> evidence extraction -> equation mapping -> outline -> bibliography -> figures ->
    # section drafting -> assembly -> claim audit -> cross-artifact consistency -> submission QC.
    "scribe": ("contract", "corpus", "repro", "evidence", "equations", "outline", "bibliography",
               "figures", "sections", "assemble", "claim_audit", "consistency", "submission_qc"),
    # Read-only mesh/dataset validation (scripts/validate_lvw_mesh.py).
    "mesh_validate": ("inventory", "topology", "markers", "cavity", "fibres", "cross", "report"),
    # LV+aorta (LVW) mesh rebuild (remesh_lvw/). One stage name per pipeline stage.
    "remesh_lvw": ("surfaces", "graft", "shells", "basal", "aorta", "transplant",
                   "volmesh", "label", "fiber", "write", "verify"),
    # Experimental swine hemodynamics extraction (postprocessing/extract_swine_hemodynamics.py).
    "swine_extract": ("load", "beats", "metrics", "write"),
    # LVW 0D PV-loop Monte-Carlo fit (calibration/stages/lvw_loop_fit.py).
    "lvw_fit": ("plan", "search", "select", "notes", "finalize"),
    # Read-only transmural-resolution / fibre-physiology audit (scripts/audit_lvw_fibers.py).
    "fiber_audit": ("load", "laplace", "transmural", "basis", "helix", "sheet", "report"),
    # Regional contractility: zone-marker construction (calibration/regions.py).
    "regions": ("read", "partition", "write", "verify"),
    # Dang 2005 akinesis-map driver (scripts/dang2005_akinesis_map.py): one grid cell per
    # (C_infarct, %Tmax_infarct), plus the pre-grid feasibility probe.
    "dang": ("plan", "probe", "cell", "collect"),
    # Figure-6 four-chamber event-state snapshots (fch_events/): resolve the named events
    # from the 0D PV trace, measure the fibre triad, then one isolated FE equilibrium per event.
    "fch_events": ("events", "basis", "build", "continuation", "solve", "export", "aggregate"),
    # ED-state manifest generator (scripts/make_fch_ed_manifest.py): read the mesh, measure the
    # cavities twice (DOLFIN + dolfin-free), cross-check, write, round-trip through the contract.
    "ed_manifest": ("read", "measure", "crosscheck", "write", "verify"),
    # Calibration-targets generators (scripts/make_fch_targets_manuscript.py): read inputs,
    # derive the bands, write, verify the file loads through the objective.
    "targets": ("read", "derive", "write", "verify"),
    # The logger's own diagnostics (emission failures). Never used by callers.
    "heartlog": ("emit",),
}


def is_known(source: str, stage: str) -> bool:
    """True if ``stage`` belongs to ``source``'s declared vocabulary."""
    return stage in STAGES.get(source, ())
