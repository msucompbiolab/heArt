# Validation record

How far reproduction has been carried for this package, and what remains open. No values
are restated here — the scripts in `scripts/` print what they compare.

## Status

| check | result |
|---|---|
| `check_closure.py` — the package is exactly the import closure | pass |
| `code_provenance` over the nine trajectory-defining sources | pass |
| mesh checksum | pass |
| checksums of every input and reference artifact | pass |
| serial (np=1) inflation ramp, including unloaded cavity volumes | exact |
| parallel run, unloaded cavity volumes | agrees to round-off, not bitwise |
| parallel run, `output_PV.csv` | agrees to round-off; deviation bounded, not growing |
| output checksums against `reference/MANIFEST.sha256` | in progress |

`verify_outputs.sh` was exercised against the preserved original run, where every numerical
artifact matched, and against a different arm of the same campaign, where none did, so a
pass from it is meaningful.

## Bitwise equality holds in serial, not in parallel

Run in serial, this package reproduces the original result bit for bit. Run across multiple
ranks, the unloaded cavity volumes differ from the original in their final bits.
`scripts/verify_loading.py` reports the magnitude, in absolute, relative and ULP terms.

The difference does not grow. Tracked across the opening cycle, the relative deviation in
both cavity volumes and cavity pressures stays flat at the level it started — the coupled
solve contracts the perturbation rather than amplifying it, so this is a fixed round-off
offset and not the beginning of a divergence. `scripts/verify_trajectory.py` reports the
per-column figures for any prefix.

The cause is reduction order. Floating-point summation is not associative, so a reduction
over a domain-decomposed mesh depends on the order the ranks are visited, and that depends
on the partition. A serial run has no such freedom, which is why it is exact and why the
difference can be attributed to the parallel reduction rather than to the code, the inputs
or the hardware — `code_provenance` covers every source file that can move the trajectory
and matches; the mesh and input checksums match; and the job was pinned to the CPU feature
the original used.

**What this means for the checksum comparison.** `MANIFEST.sha256` is a bitwise instrument,
and a last-bit difference anywhere changes a checksum completely. The HDF5 payloads and CSV
traces of a parallel run will therefore not match it, even though the trajectory agrees to
round-off throughout. Read a mismatch there together with this note; `verify_trajectory.py`
is the instrument that measures how close the agreement actually is.

The claim this package supports is:

> The code, inputs and environment are byte-identical to those that produced the original
> result; a serial run reproduces it bit for bit; a parallel run reproduces it to
> floating-point round-off.

For a bitwise check, run in serial and compare against a serial reference. A
bitwise-reproducible parallel run would need a fixed mesh partition and a deterministic
reduction order, neither of which the original run pinned.
