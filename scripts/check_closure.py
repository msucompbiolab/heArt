#!/usr/bin/env python3
"""Assert the package is exactly the import closure of the run -- no more, no less.

Two failure modes this guards against:

  * something needed was dropped -- an import that resolves to nothing and is NOT one of
    the capabilities removed on purpose. That is a run that dies after the JIT compile;
  * something unneeded crept back in -- a .py file no import reaches.

The four modules below were removed because this run cannot reach them: the configuration
selects Guccione passive and Time-varying active, and kappa = 0. Their import sites sit
inside branches in files that are byte-locked by code_provenance, so the statements stay
and are expected to dangle. Reaching one raises ImportError, which is the correct outcome:
this package reproduces one run and is not a general solver.

Needs only the standard library. Exit 0 when the closure is exact.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "demo/fch_baseline.py"

REMOVED = {
    "src.mechanics.holzapfelogden": "alternative passive law (config selects Guccione)",
    "src.mechanics.GuccioneAct": "alternative active law (config selects Time-varying)",
    "src.mechanics.transverse_config": "cross-fibre activation (kappa = 0 here)",
    "src.mechanics.rigid_support": "automatic rigid-body removal (not used by this arm)",
}


def module_candidates(dotted):
    parts = [p for p in dotted.split(".") if p]
    if parts and parts[0] == "heArt_legacy":
        parts = parts[1:]
    if not parts:
        return []
    stem = "/".join(parts)
    return [f"{stem}.py", f"{stem}/__init__.py"]


def imports_of(path, rel):
    """(module that must resolve, optional submodule candidates) pairs for this file.

    `from a.b import c` is ambiguous: c may be a submodule or just a name defined in a.b.
    Only the BASE (a.b) has to resolve to a file; c is followed when it happens to be a
    submodule and ignored otherwise. Treating every imported name as a module is what makes
    a naive closure walker report half the package as missing.
    """
    out = []
    pkg = list(rel.parts[:-1])
    FIRST_PARTY = ("heArt_legacy", "src", "orchestrate", "calibration", "heartlog", "demo")
    for node in ast.walk(ast.parse(path.read_text(errors="ignore"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                if node.module:
                    base = base + node.module.split(".")
            elif node.module and node.module.split(".")[0] in FIRST_PARTY:
                base = node.module.split(".")
                if base[0] == "heArt_legacy":
                    base = base[1:]
            else:
                continue
            if base:
                subs = [".".join(base + [a.name]) for a in node.names]
                if node.module is None:
                    # `from . import x` / `from .. import x`: every name MUST be a submodule,
                    # so each one has to resolve. This is how a removed module that is only
                    # ever imported that way (rigid_support) stays visible to this check.
                    out.append((".".join(base), []))
                    out.extend((sub, []) for sub in subs)
                else:
                    out.append((".".join(base), subs))
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in FIRST_PARTY:
                    out.append((".".join(parts[1:] if parts[0] == "heArt_legacy" else parts), []))
    return out


def main():
    present = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
               if p.relative_to(ROOT).parts[0] != "scripts"}

    reached, dangling, stack = set(), {}, [Path(ENTRY)]
    while stack:
        rel = stack.pop()
        if str(rel) in reached or str(rel) not in present:
            continue
        reached.add(str(rel))
        # Importing a submodule executes every package __init__ above it.
        parts = rel.parts[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if init in present:
                reached.add(init)
        for base, subs in imports_of(ROOT / rel, rel):
            hit = [c for c in module_candidates(base) if c in present]
            if hit:
                stack.extend(Path(c) for c in hit)
            else:
                dangling.setdefault(base, []).append(str(rel))
            # follow `from pkg import submodule` only where it really is a submodule
            for sub in subs:
                stack.extend(Path(c) for c in module_candidates(sub) if c in present)

    failures = []

    unexpected = {d: w for d, w in dangling.items() if d not in REMOVED}
    for d, where in sorted(unexpected.items()):
        print(f"[FAIL] import resolves to nothing: {d}  (from {', '.join(sorted(set(where)))})")
        failures.append(d)
    for d in sorted(REMOVED):
        if d in dangling:
            print(f"[OK ] removed on purpose: {d:<38} {REMOVED[d]}")
        else:
            print(f"[NOTE] {d} is no longer imported anywhere; drop it from REMOVED")

    unreached = sorted(present - reached)
    for f in unreached:
        print(f"[FAIL] file is in the package but no import reaches it: {f}")
        failures.append(f)

    print(f"\nreached {len(reached)} of {len(present)} python files from {ENTRY}")
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        return 1
    print("Closure is exact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
