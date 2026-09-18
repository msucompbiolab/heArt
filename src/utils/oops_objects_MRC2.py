import dolfin as df
import os
import json
import shutil
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from .edgetypebc import *

class SafeH5Wrapper:
    """Best-effort HDF5 wrapper: suppresses repeated write errors."""
    def __init__(self, hdf):
        self.hdf = hdf
        self.failed = False

    def write(self, *args, **kwargs):
        if self.failed:
            return
        try:
            self.hdf.write(*args, **kwargs)
        except Exception as e:
            print("WARNING: HDF5 write skipped:", e)
            self.failed = True

    def close(self):
        try:
            self.hdf.close()
        except Exception:
            pass

def printout(statement, mpi_comm, stage="solve", level=None, **kv):
    """Emit one record on the canonical log.

    Signature and message text are unchanged from the original rank-0 ``print`` helper, because
    the message bodies are a parsing contract for the autonomous loops' failure signatures
    (``Backoff dt reached minimum factor``, ``Loading iteration number = N``, ...). What is added
    is the standard prefix, a real level lifted from any leading severity word, and stage
    attribution.
    """
    from .log_mpi import get_logger, infer_level

    get_logger("fe", mpi_comm).log(
        level or infer_level(statement), stage, str(statement), **kv
    )




def update_mesh(mesh, displacement, boundaries):
    """Return moved mesh plus migrated boundary markers."""
    # Copy+move keeps the original mesh immutable for callers that cache topology/markers.
    new_mesh = df.Mesh(mesh)
    dim = boundaries.dim() if hasattr(boundaries, "dim") else new_mesh.topology().dim() - 1
    new_boundaries = df.MeshFunction("size_t", new_mesh, dim, new_mesh.domains())
    new_boundaries.set_values(boundaries.array())
    df.ALE.move(new_mesh, displacement)
    return new_mesh, new_boundaries


def has_dataset_parallel(dataset, comm, h5_path=None, h5_file=None):
    """MPI-safe dataset existence check to avoid noisy HDF5 diagnostics."""
    if comm is None:
        comm = df.MPI.comm_world
    mpi_comm = comm.tompi4py() if hasattr(comm, "tompi4py") else comm
    try:
        rank = df.MPI.rank(comm)
    except Exception:
        rank = mpi_comm.rank if hasattr(mpi_comm, "rank") else 0
    exists = False
    if rank == 0:
        if h5_path:
            try:
                import h5py
                with h5py.File(h5_path, "r") as h5:
                    exists = dataset in h5
            except Exception:
                exists = h5_file.has_dataset(dataset) if h5_file is not None else False
        elif h5_file is not None:
            exists = h5_file.has_dataset(dataset)
    if hasattr(mpi_comm, "bcast"):
        exists = mpi_comm.bcast(exists, root=0)
    return bool(exists)


def _read_dataset(h5_file, target, dataset, h5_path=None, comm=None):
    """Read dataset into target if available; return bool."""
    if comm is not None or h5_path is not None:
        exists = has_dataset_parallel(dataset, comm, h5_path=h5_path, h5_file=h5_file)
        if exists:
            h5_file.read(target, dataset)
        return exists
    if h5_file.has_dataset(dataset):
        h5_file.read(target, dataset)
        return True
    return False


def _build_vector_space(mesh, spec, default_family="Quadrature", default_degree=0):
    # Accept shorthand like "DG0" or "Quadrature:4" so mesh config can pick storage/FE without code changes.
    family = default_family
    degree = default_degree
    if spec:
        if ":" in spec:
            family, degree_str = spec.split(":", 1)
            degree = int(degree_str)
        else:
            letters = "".join(ch for ch in spec if ch.isalpha())
            digits = "".join(ch for ch in spec if ch.isdigit())
            if letters:
                family = letters
            if digits:
                degree = int(digits)
    kwargs = {}
    if family.lower() == "quadrature":
        kwargs["quad_scheme"] = "default"
    element = df.VectorElement(family, mesh.ufl_cell(), degree, **kwargs)
    if "quad_scheme" in kwargs:
        element._quad_scheme = kwargs["quad_scheme"]
    return df.FunctionSpace(mesh, element), degree


def _create_fiber_spaces(mesh, params, SimDet):
    """Return (fiber_space, storage_space, quadrature_degree) used for fiber loading/projection."""
    quad_degree = SimDet["GiccioneParams"]["deg"]
    eval_spec = SimDet.get("fiber_eval_element") or params.get("fiber_eval_element")
    storage_spec = SimDet.get("fiber_storage_element") or params.get(
        "fiber_storage_element", "DG0"
    )
    # Use a configurable evaluation space while keeping integration quadrature from GiccioneParams.
    if eval_spec:
        fiber_space, _ = _build_vector_space(mesh, eval_spec, "Quadrature", quad_degree)
    else:
        fiber_space, _ = _build_vector_space(
            mesh, f"Quadrature:{quad_degree}", "Quadrature", quad_degree
        )
    storage_space, _ = _build_vector_space(mesh, storage_spec, "DG", 0)
    return fiber_space, storage_space, quad_degree


def _load_fiber_field(h5_file, casename, dataset, storage_space, target_func):
    """Read a fiber-like vector field from HDF5 into the storage space and assign to target."""
    path = f"{casename}/{dataset}"
    temp = df.Function(storage_space)
    h5_file.read(temp, path)
    target_func.assign(temp)


def _get_local_vector_data(vec):
    """Return local numpy view of a DOLFIN vector (works across dolfin versions)."""
    return vec.get_local() if hasattr(vec, "get_local") else vec.array()


def _set_local_vector_data(vec, data):
    """Set local DOLFIN vector data in-place (works across dolfin versions)."""
    if hasattr(vec, "set_local"):
        vec.set_local(data)
    else:
        vec[:] = data
    vec.apply("insert")


def _normalize_vector_function_inplace(func):
    """Normalize a vector df.Function pointwise in its own DOF representation."""
    vec = func.vector()
    local = _get_local_vector_data(vec)
    value_size = func.function_space().ufl_element().value_size()
    if value_size <= 0:
        return
    if local.size % value_size != 0:
        raise ValueError(
            f"Unexpected vector size {local.size} for value_size {value_size}."
        )
    data = local.reshape((-1, value_size))
    norms = np.linalg.norm(data, axis=1)
    norms[norms == 0.0] = 1.0
    data /= norms[:, None]
    _set_local_vector_data(vec, data.reshape((-1,)))


def _orthonormalize_fiber_triad_inplace(f0, s0, n0, comm, tag="mesh", tol=0.02):
    """PERMANENT GUARD: enforce an ORTHONORMAL fibre/sheet/normal frame on the loaded
    triad, in place. The Guccione/Holzapfel anisotropic stiffness is defined in the local
    (f0, s0, n0) frame; if that frame is skewed the directional stiffness (bff/bfx/bxx) is
    wrong. A fibre push-forward that skips re-orthogonalization (f' = F*f normalized, F not
    orthogonal -> f'.s' != 0) produces exactly such a skew (e.g. a deformed/morphed/inflated
    mesh). So at LOAD TIME — seamlessly, for every workflow, regardless of how the mesh was
    produced — if the triad is non-orthonormal beyond `tol` we emit a LOUD WARNING (per the
    repo error-handling rule: never silently degrade) and Gram-Schmidt re-orthonormalize:
    keep the fibre exact, orthogonalize the sheet against it, set normal = fibre x sheet.
    A no-op for an already-orthonormal triad (the validated eF_rb/eS_rb/eN_rb case)."""
    if f0.function_space().ufl_element().value_size() != 3:
        return
    F = _get_local_vector_data(f0.vector()).reshape((-1, 3))
    S = _get_local_vector_data(s0.vector()).reshape((-1, 3))
    local_max = float(np.max(np.abs(np.einsum("ij,ij->i", F, S)))) if F.size else 0.0
    if df.MPI.max(comm, local_max) <= tol:
        return
    printout(
        "WARNING: loaded %s fibers are NON-ORTHONORMAL (max|f.s|=%.3f > %.3f) -- a fiber "
        "transform upstream skipped Gram-Schmidt re-orthogonalization (F*f normalized does "
        "not preserve the frame). Re-orthonormalizing the f/s/n triad in place so the "
        "anisotropic material frame is correct." % (tag, df.MPI.max(comm, local_max), tol),
        comm,
    )
    fn = np.linalg.norm(F, axis=1); fn[fn == 0.0] = 1.0
    fh = F / fn[:, None]
    sp = S - (np.einsum("ij,ij->i", S, fh)[:, None]) * fh
    sn = np.linalg.norm(sp, axis=1); sn[sn == 0.0] = 1.0
    sh = sp / sn[:, None]
    nh = np.cross(fh, sh)
    _set_local_vector_data(f0.vector(), fh.reshape((-1,)))
    _set_local_vector_data(s0.vector(), sh.reshape((-1,)))
    _set_local_vector_data(n0.vector(), nh.reshape((-1,)))


def _update_cavity_control_value(target, value, attr):
    """Set a cavity control scalar for either Constant/Function (`assign`) or legacy Expression fields."""
    if hasattr(target, "assign"):
        target.assign(value)
        return
    if hasattr(target, attr):
        setattr(target, attr, value)
        return
    raise AttributeError(f"Unsupported cavity control type for {attr}.")


def _validate_mesh_marker(marker, expected_dim, name):
    """Raise if a MeshFunction has the wrong topological dimension."""
    if marker is None:
        raise ValueError(f"{name} is required but missing.")
    if marker.dim() != expected_dim:
        raise ValueError(
            f"{name} has dim={marker.dim()} but expected {expected_dim} for this mesh."
        )


def _marker_ids_present(marker, required_ids, comm):
    """MPI-safe: return the subset of ``required_ids`` actually present anywhere in a
    MeshFunction's marker array (a cell/facet carries that id on at least one rank).
    Uses dolfin's MPI.max reduction (no mpi4py op needed)."""
    local = set(int(v) for v in marker.array())
    present = set()
    for rid in required_ids:
        has_local = 1 if int(rid) in local else 0
        if df.MPI.max(comm, has_local) > 0:
            present.add(int(rid))
    return present


def _validate_fch_region_contract(matid, facetboundaries, SimDet, params, comm, mesh_file=""):
    """Fail loudly if the FCH mesh does not carry the region/facet marker ids the
    solver keys off (the contract is a FIXED record in canonical FCH_BASE_REGIONS, not
    queried from the mesh -- a regenerated mesh with different ids would otherwise be
    read SILENTLY wrong). Checks the four myocardial matid regions and the four cavity
    + epicardial facet ids; raises listing any missing id. Per the repo's
    non-negotiable error-handling principle (surface, don't assume)."""
    def _id(key, default):
        v = SimDet.get(key, params.get(key, default))
        return int(v)

    req_regions = {
        "lv_rid": _id("lv_rid", 10), "rv_rid": _id("rv_rid", 9),
        "la_rid": _id("la_rid", 11), "ra_rid": _id("ra_rid", 8),
    }
    req_facets = {
        "LVendoid": _id("LVendoid", 1), "RVendoid": _id("RVendoid", 6),
        "LAendoid": _id("LAendoid", 2), "RAendoid": _id("RAendoid", 3),
        "epiid": _id("epiid", 9),
    }
    have_regions = _marker_ids_present(matid, req_regions.values(), comm)
    have_facets = _marker_ids_present(facetboundaries, req_facets.values(), comm)
    missing = [f"{k}={v} (matid)" for k, v in req_regions.items() if v not in have_regions]
    missing += [f"{k}={v} (facet)" for k, v in req_facets.items() if v not in have_facets]
    if missing:
        raise RuntimeError(
            f"FCH mesh '{mesh_file}' is missing required marker ids: {missing}. "
            f"The region/facet contract (canonical FCH_BASE_REGIONS) is not satisfied "
            f"-- the mesh would be read with wrong region->material/cavity wiring. "
            f"Regenerate the mesh with the canonical ids or fix the *_rid/*endoid keys."
        )
    printout(
        f"FCH region/facet contract OK: matid{sorted(req_regions.values())} + "
        f"facets{sorted(req_facets.values())} all present.",
        comm,
    )


def defCPP_LBBB_Matprop(mesh, mId, meshname="CRT27_AS_smooth_fine"):
    # Build a DG0 Function instead of a JIT-compiled Expression for LBBB tagging.
    marker_set = {14, 15, 16, 24, 25, 26, 27}
    V = df.FunctionSpace(mesh, "DG", 0)
    kappa = df.Function(V)
    local = kappa.vector().get_local()
    dm = V.dofmap()
    marker_values = mId.array()
    for cell in df.cells(mesh):
        dof = dm.cell_dofs(cell.index())[0]
        local[dof] = 0.0 if marker_values[cell.index()] in marker_set else 1.0
    kappa.vector().set_local(local)
    kappa.vector().apply("insert")

    return kappa





class Constant_definitions(object):
    """Container for FitzHugh–Nagumo constants; adjust by overriding `parameters` after init."""

    def __init__(self):
        self.parameters = self.default_parameters()

    def default_parameters(self):
        d1 = 0.2  # diffusion constant for action potential 0.2 in Nash Panfilov
        return {
            "alpha": df.Constant(0.01),  # 0.1 in 'a' in NashPanfilov, 0.01 in Gok_Kuhl
            "gamma": df.Constant(0.002),  # 0.01 'epsilon' in NasPanlov, 0.002 in Gok_Kuhl
            "b": df.Constant(0.15),  # 0.1 in Gok_Kuhl
            "c": df.Constant(8),  # 'k' is NashPanfilov, 8 in Gok_Kuhl
            "mu1": df.Constant(0.2),  # 0.12 in NP ,# 0.2 in Gok_Kuhl
            "mu2": df.Constant(0.3),
            "D1": df.Constant(
                ((d1, "0.0", "0.0"), ("0.0", d1, "0.0"), ("0.0", "0.0", d1))
            ),
            "B": df.Constant((0.0, 0.0, 0.0)),
            "T": df.Constant((0.0, 0.0, 0.0)),
        }




class biventricle_mesh(object):
    """
    Mechanics mesh container for BiV runs: loads geometry/markers/fibers and exposes dx/ds measures.
    """

    def default_parameters(self):
        # Defaults mirror historical CRT27 datasets; override via SimDet to point at other meshes.
        return {
            "directory": "../CRT27/",
            "casename": "CRT27",
            "fibre_quad_degree": 4,
            "outputfolder": "../Outputs/",
            "topid": 4,
            "LVendoid": 2,
            "RVendoid": 3,
            "epiid": 1,
        }

    def update_parameters(self, params):
        self.parameters.update(params)

    def __init__(self, params, SimDet):
        self.mesh = df.Mesh()
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        params = self.parameters

        meshfilename = os.path.join(params["directory"], params["casename"] + ".hdf5")
        f = df.HDF5File(df.MPI.comm_world, meshfilename, "r")
        f.read(self.mesh, params["casename"], False)
        self.facetboundaries = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        facet_dataset = SimDet.get(
            "facetboundaries_dataset", params.get("facetboundaries_dataset", "facetboundaries")
        )
        f.read(self.facetboundaries, f"{params['casename']}/{facet_dataset}")

        self.edgeboundaries = df.MeshFunction("size_t", self.mesh, 1)
        edgeboundary_path = params["casename"] + "/edgeboundaries"
        if has_dataset_parallel(
            edgeboundary_path,
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            try:
                f.read(self.edgeboundaries, edgeboundary_path)
            except (MemoryError, RuntimeError, ValueError) as exc:
                self.edgeboundaries.set_all(0)
                printout(f"WARNING: No edge boundaries read ({exc})", self.mesh.mpi_comm())
        else:
            self.edgeboundaries.set_all(0)
            printout(
                "WARNING: No edge boundaries read (dataset missing)",
                self.mesh.mpi_comm(),
            )

        self.fiberFS, storage_fs, deg = _create_fiber_spaces(self.mesh, params, SimDet)

        self.f0 = df.Function(self.fiberFS)
        self.s0 = df.Function(self.fiberFS)
        self.n0 = df.Function(self.fiberFS)

        fiber_suffix = "_proj_DTI" if SimDet.get("DTI_ME", False)  else ""
        for func, name in (
            (self.f0, f"eF{fiber_suffix}"),
            (self.s0, f"eS{fiber_suffix}"),
            (self.n0, f"eN{fiber_suffix}"),
        ):
            _load_fiber_field(f, params["casename"], name, storage_fs, func)

        _normalize_vector_function_inplace(self.f0)
        _normalize_vector_function_inplace(self.s0)
        _normalize_vector_function_inplace(self.n0)

        self.matid = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim(), self.mesh.domains()
        )
        matid_dataset = SimDet.get(
            "matid_dataset", params.get("matid_dataset", "matid1")
        )
        if not _read_dataset(
            f,
            self.matid,
            f"{params['casename']}/{matid_dataset}",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.matid.set_all(0)

        self.AHAid = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim(), self.mesh.domains()
        )
        if not _read_dataset(
            f,
            self.AHAid,
            params["casename"] + "/AHAid",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.AHAid.set_all(0)

        EpiBCid = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1, self.mesh.domains()
        )
        if not _read_dataset(
            f,
            EpiBCid,
            params["casename"] + "/EpiBCid_Corr",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            EpiBCid.set_all(0)

        self.EpiBCid_me = EpiBCid

        f.close()

        _validate_mesh_marker(self.matid, self.mesh.topology().dim(), "matid")
        _validate_mesh_marker(self.facetboundaries, self.mesh.topology().dim() - 1, "facetboundaries")
        _validate_mesh_marker(self.EpiBCid_me, self.mesh.topology().dim() - 1, "EpiBCid")

        self.topid = params["topid"]
        self.LVendoid = params["LVendoid"]
        self.RVendoid = params["RVendoid"]
        self.epiid = params["epiid"]

        self.dx = df.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=self.matid,
            metadata={"quadrature_degree": deg},
        )
        self.ds = df.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facetboundaries,
        )

        comm = self.mesh.mpi_comm()
        printout(f"df.Mesh size is : {self.mesh.num_cells():.6f}", comm)
        r_qmin, r_qmax = df.MeshQuality.radius_ratio_min_max(self.mesh)
        printout(f"Minimal radius ratio: {r_qmin}", comm)
        printout(f"Maximal radius ratio: {r_qmax}", comm)



class lv_mesh(object):
    """
    object for lv mesh
    input: mesh, facet, edge, matid, fibre file
    output: mesh
    """

    def default_parameters(self):
        return {
            "directory": "../CRT27/",
            "casename": "CRT27",
            "fibre_quad_degree": 4,
            "outputfolder": "../Outputs/",
            "topid": 4,
            "LVendoid": 2,
            "epiid": 1,
        }

    def update_parameters(self, params):
        self.parameters.update(params)

    def eval_lv(self):
        printout(f"number of ceLLs in mesh are: {self.mesh.num_cells()}", self.mesh.mpi_comm())

    def __init__(self, params, SimDet):
        self.mesh = df.Mesh()
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        params = self.parameters

        meshfilename = os.path.join(params["directory"], params["casename"] + ".hdf5")
        f = df.HDF5File(df.MPI.comm_world, meshfilename, "r")
        f.read(self.mesh, params["casename"], False)
        mesh_scale_env = os.environ.get("WAORTA_MESH_SCALE")
        mesh_scale_opt = SimDet.get("mesh_scale_waorta", 1.0)
        try:
            mesh_scale_env_val = float(mesh_scale_env) if mesh_scale_env is not None else None
        except (TypeError, ValueError):
            mesh_scale_env_val = None
            printout(
                f"WARNING: Ignoring invalid WAORTA_MESH_SCALE={mesh_scale_env}; using SimDet mesh_scale_waorta={mesh_scale_opt}",
                self.mesh.mpi_comm(),
            )
        mesh_scale = mesh_scale_env_val if mesh_scale_env_val is not None else float(mesh_scale_opt)
        self.mesh.scale(mesh_scale)


        self.facetboundaries = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        facet_dataset = SimDet.get(
            "facetboundaries_dataset", params.get("facetboundaries_dataset", "facetboundaries")
        )
        f.read(self.facetboundaries, f"{params['casename']}/{facet_dataset}")

        self.edgeboundaries = df.MeshFunction("size_t", self.mesh, 1)
        edgeboundary_path = params["casename"] + "/edgeboundaries"
        if has_dataset_parallel(
            edgeboundary_path,
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            try:
                f.read(self.edgeboundaries, edgeboundary_path)
            except (MemoryError, RuntimeError, ValueError) as exc:
                self.edgeboundaries.set_all(0)
                printout(f"WARNING: No edge boundaries read ({exc})", self.mesh.mpi_comm())
        else:
            self.edgeboundaries.set_all(0)
            printout(
                "WARNING: No edge boundaries read (dataset missing)",
                self.mesh.mpi_comm(),
            )

        self.iswaorta = SimDet.get("iswaorta", False)

        self.fiberFS, storage_fs, deg = _create_fiber_spaces(self.mesh, params, SimDet)

        self.f0 = df.Function(self.fiberFS)
        self.s0 = df.Function(self.fiberFS)
        self.n0 = df.Function(self.fiberFS)

        fiber_suffix = "_proj_DTI" if SimDet.get("DTI_ME", False) else ""
        for func, name in (
            (self.f0, f"eF{fiber_suffix}"),
            (self.s0, f"eS{fiber_suffix}"),
            (self.n0, f"eN{fiber_suffix}"),
        ):
            _load_fiber_field(f, params["casename"], name, storage_fs, func)

        if has_dataset_parallel(
            params["casename"] + "/eC",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eC0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eC", storage_fs, self.eC0)

        if has_dataset_parallel(
            params["casename"] + "/eL",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eL0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eL", storage_fs, self.eL0)

        if has_dataset_parallel(
            params["casename"] + "/eC_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eC0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eC_ao", storage_fs, self.eC0_ao)

        if has_dataset_parallel(
            params["casename"] + "/eL_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eL0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eL_ao", storage_fs, self.eL0_ao)

        if has_dataset_parallel(
            params["casename"] + "/M1_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eclgn0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "M1_ao", storage_fs, self.eclgn0_ao)

        if has_dataset_parallel(
            params["casename"] + "/M2_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eclgn1_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "M2_ao", storage_fs, self.eclgn1_ao)

        if has_dataset_parallel(
            params["casename"] + "/eR",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eR0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eR", storage_fs, self.eR0)

        self.f0_ori = df.Function(self.fiberFS)
        self.f0_ori.assign(self.f0)
        self.s0_ori = df.Function(self.fiberFS)
        self.s0_ori.assign(self.s0)
        self.n0_ori = df.Function(self.fiberFS)
        self.n0_ori.assign(self.n0)

        _normalize_vector_function_inplace(self.f0)
        _normalize_vector_function_inplace(self.s0)
        _normalize_vector_function_inplace(self.n0)

        self.matid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        matid_dataset = SimDet.get(
            "matid_dataset", params.get("matid_dataset", "matid1")
        )
        if not _read_dataset(
            f,
            self.matid,
            f"{params['casename']}/{matid_dataset}",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.matid.set_all(0)

        self.AHAid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        if not _read_dataset(
            f,
            self.AHAid,
            params["casename"] + "/AHAid",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.AHAid.set_all(0)

        EpiBCid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        if not _read_dataset(
            f,
            EpiBCid,
            params["casename"] + "/EpiBCid_Corr",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            EpiBCid.set_all(0)

        self.EpiBCid_me = EpiBCid

        _validate_mesh_marker(self.matid, self.mesh.topology().dim(), "matid")
        _validate_mesh_marker(self.facetboundaries, self.mesh.topology().dim() - 1, "facetboundaries")
        _validate_mesh_marker(self.EpiBCid_me, self.mesh.topology().dim() - 1, "EpiBCid")

        f.close()

        self.topid = params["topid"]
        self.LVendoid = params["LVendoid"]
        self.epiid = params["epiid"]

        _validate_mesh_marker(self.matid, self.mesh.topology().dim(), "matid")
        _validate_mesh_marker(self.facetboundaries, self.mesh.topology().dim() - 1, "facetboundaries")
        _validate_mesh_marker(self.EpiBCid_me, self.mesh.topology().dim() - 1, "EpiBCid")

        self.dx = df.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=self.matid,
            metadata={"quadrature_degree": deg},
        )
        self.ds = df.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facetboundaries,
        )

        comm = self.mesh.mpi_comm()
        printout(f"df.Mesh size is : {self.mesh.num_cells():.6f}", comm)
        r_qmin, r_qmax = df.MeshQuality.radius_ratio_min_max(self.mesh)
        printout(f"Minimal radius ratio: {r_qmin}", comm)
        printout(f"Maximal radius ratio: {r_qmax}", comm)




class fch_mesh(object):
    """
    object for fch mesh
    input: mesh, facet, edge, matid, fibre file
    output: mesh
    """

    def default_parameters(self):
        return {
            "directory": "../FCHMesh/",
            "casename": "fch_mesh",
            "fibre_quad_degree": 0,
            "outputfolder": "../Outputs/",
            "epiid": 1,
            "LVendoid": 26,
            "fiber_element": "DG0",
        }

    def update_parameters(self, params):
        self.parameters.update(params)

    def eval_fch(self):
        printout(f"number of ceLLs in mesh are: {self.mesh.num_cells()}", self.mesh.mpi_comm())

    def __init__(self, params, SimDet):
        self.mesh = df.Mesh()
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        params = self.parameters

        # Mechanics mesh FILE basename is decoupled from the HDF5 group/`casename`:
        # the canonical four-chamber file is `fch_clregion_whole.hdf5` while its
        # internal group (and every dataset path) is still `fch_clregion`. A
        # `mesh_basename` override (params or SimDet; defaults to `casename`)
        # selects the file without touching any dataset-group read below.
        mesh_basename = (
            params.get("mesh_basename") or SimDet.get("mesh_basename") or params["casename"]
        )
        meshfilename = os.path.join(params["directory"], mesh_basename + ".hdf5")
        printout(f"FCH mechanics mesh file: {meshfilename} (group '{params['casename']}')",
                 df.MPI.comm_world)

        f = df.HDF5File(df.MPI.comm_world, meshfilename, "r")
        f.read(self.mesh, params["casename"], False)
        mesh_scale_env = os.environ.get("FCH_MESH_SCALE")
        mesh_scale_opt = SimDet.get("mesh_scale_fch", 1.0)
        try:
            mesh_scale_env_val = float(mesh_scale_env) if mesh_scale_env is not None else None
        except (TypeError, ValueError):
            mesh_scale_env_val = None
            printout(
                f"WARNING: Ignoring invalid FCH_MESH_SCALE={mesh_scale_env}; using SimDet mesh_scale_fch={mesh_scale_opt}",
                self.mesh.mpi_comm(),
            )
        mesh_scale = mesh_scale_env_val if mesh_scale_env_val is not None else float(mesh_scale_opt)
        self.mesh.scale(mesh_scale)

        # Initialize facet connectivity BEFORE constructing the facet MeshFunction:
        # the HDF5 reader matches stored facets to mesh facets by vertex topology,
        # which requires dim-1 entities to exist and the MeshFunction to be sized to
        # them. Idempotent for meshes whose facets are already numbered, but required
        # for freshly-(re)meshed FCH files (else the function sizes to 0 facets and
        # the read overflows -> "cannot create std::vector larger than max_size()").
        self.mesh.init(self.mesh.topology().dim() - 1)
        self.facetboundaries = df.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        facet_dataset = SimDet.get("facetboundaries_dataset", "facetboundaries")
        f.read(self.facetboundaries, f"{params['casename']}/{facet_dataset}")

        self.edgeboundaries = df.MeshFunction("size_t", self.mesh, 1)

        self.fiberFS, storage_fs, deg = _create_fiber_spaces(self.mesh, params, SimDet)

        self.f0 = df.Function(self.fiberFS)
        self.s0 = df.Function(self.fiberFS)
        self.n0 = df.Function(self.fiberFS)

        if SimDet.get("DTI_ME", False):
            suffixes = ("eF_proj_DTI", "eS_proj_DTI", "eN_proj_DTI")
        else:
            default_fiber_names = {"f0": "eF_a", "s0": "eS_a", "n0": "eN_a"}
            fiber_names = SimDet.get("fiber_datasets") or default_fiber_names
            suffixes = (
                fiber_names.get("f0", default_fiber_names["f0"]),
                fiber_names.get("s0", default_fiber_names["s0"]),
                fiber_names.get("n0", default_fiber_names["n0"]),
            )
        for func, name in zip((self.f0, self.s0, self.n0), suffixes):
            _load_fiber_field(f, params["casename"], name, storage_fs, func)
        # Permanent guard: a deformed/morphed/inflated FCH mesh whose fibre transform
        # skipped Gram-Schmidt re-orthogonalization would feed the Guccione law a skewed
        # frame. Re-orthonormalize at load (loud WARNING if it fires; no-op when clean).
        _orthonormalize_fiber_triad_inplace(
            self.f0, self.s0, self.n0, self.mesh.mpi_comm(), tag="fch_mesh")

        if has_dataset_parallel(
            params["casename"] + "/eC",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eC0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eC", storage_fs, self.eC0)

        if has_dataset_parallel(
            params["casename"] + "/eL",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eL0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eL", storage_fs, self.eL0)

        if has_dataset_parallel(
            params["casename"] + "/eC_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eC0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eC_ao", storage_fs, self.eC0_ao)

        if has_dataset_parallel(
            params["casename"] + "/eL_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eL0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eL_ao", storage_fs, self.eL0_ao)

        if has_dataset_parallel(
            params["casename"] + "/M1_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eclgn0_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "M1_ao", storage_fs, self.eclgn0_ao)

        if has_dataset_parallel(
            params["casename"] + "/M2_ao",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eclgn1_ao = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "M2_ao", storage_fs, self.eclgn1_ao)

        if has_dataset_parallel(
            params["casename"] + "/eR",
            self.mesh.mpi_comm(),
            h5_path=meshfilename,
            h5_file=f,
        ):
            self.eR0 = df.Function(self.fiberFS)
            _load_fiber_field(f, params["casename"], "eR", storage_fs, self.eR0)

        _normalize_vector_function_inplace(self.f0)
        _normalize_vector_function_inplace(self.s0)
        _normalize_vector_function_inplace(self.n0)

        self.matid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())

        matid_dataset = SimDet.get(
            "matid_dataset", params.get("matid_dataset", "matid1")
        )
        
        if not _read_dataset(
            f,
            self.matid,
            f"{params['casename']}/{matid_dataset}",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.matid.set_all(0)

        self.AHAid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        if not _read_dataset(
            f,
            self.AHAid,
            params["casename"] + "/AHAid",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            self.AHAid.set_all(0)

        EpiBCid = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        if not _read_dataset(
            f,
            EpiBCid,
            params["casename"] + "/EpiBCid_Corr",
            h5_path=meshfilename,
            comm=self.mesh.mpi_comm(),
        ):
            EpiBCid.set_all(0)

        self.EpiBCid_me = EpiBCid

        f.close()

        # Fail loudly if the loaded mesh does not carry the canonical FCH region/facet
        # marker contract (a regenerated/wrong mesh would otherwise be wired silently
        # wrong). Skips only if explicitly disabled.
        if SimDet.get("validate_fch_region_contract", True):
            _validate_fch_region_contract(
                self.matid, self.facetboundaries, SimDet, params,
                self.mesh.mpi_comm(), mesh_file=meshfilename,
            )

        self.LVendoid = params["LVendoid"]
        self.epiid = params["epiid"]

        self.dx = df.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=self.matid,
            metadata={"quadrature_degree": deg},
        )
        self.ds = df.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facetboundaries,
        )

        comm = self.mesh.mpi_comm()
        printout(f"df.Mesh size is : {self.mesh.num_cells():.6f}", comm)
        r_qmin, r_qmax = df.MeshQuality.radius_ratio_min_max(self.mesh)
        printout(f"Minimal radius ratio: {r_qmin}", comm)
        printout(f"Maximal radius ratio: {r_qmax}", comm)


class DiffusingMedium(object):
    """
    Object for anisotropic diffuion in FHN, and Monodomain equation
    Having a seperate object allows to add effects of Purkinjee and all that

    Input: anisotropic and isotropic diffusion coefficients could be updated
    Myofibre angles

    Output: Diffusion tensor (degree and family depends on the myofibre vectors)
    """

    def __init__(self, params, mesh, mId, isLBBB):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mesh = mesh
        self.mId = mId
        self.isLBBB = isLBBB

    def default_parameters(self):
        return {
            "D_iso": df.Constant(
                (("0.2", "0.0", "0.0"), ("0.0", "0.2", "0.0"), ("0.0", "0.0", "0.2"))
            )
        }

    def Dmat(self):
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]


        D_iso = self.parameters["D_iso"]
        d_ani = self.parameters["d_ani"]

        if self.isLBBB == True:
            d_ani = defCPP_LBBB_Matprop(mesh=self.mesh, mId=self.mId)

        Dij = d_ani * f0[i] * f0[j] + D_iso[i, j]
        D_tensor = df.as_tensor(Dij, (i, j))

        return D_tensor




class PV_Elas(object):
    """
    Elasticity equations object
    """

    def default_parameters(self):
        return {"outputfolder": "../Outputs/", "foldername": "NashPanfilov_BiV_04"}

    def update_parameters(self, params):
        self.parameters.update(params)

    def __init__(self, biVMesh, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)

        self.deg = self.parameters["degree"]
        deg = self.deg

        self.isincomp = self.parameters["is_incompressible"]
        isincomp = self.isincomp

        mesh = biVMesh.mesh
        self.isLV = SimDet["isLV"]

        Q = df.FunctionSpace(mesh, "CG", 1)

        Velem = df.VectorElement("CG", mesh.ufl_cell(), 2, quad_scheme="default")
        Velem._quad_scheme = "default"
        Qelem = df.FiniteElement("CG", mesh.ufl_cell(), 1, quad_scheme="default")
        Qelem._quad_scheme = "default"
        Relem = df.FiniteElement("Real", mesh.ufl_cell(), 0, quad_scheme="default")
        Relem._quad_scheme = "default"
        Quadelem = df.FiniteElement(
            "df.Quadrature", mesh.ufl_cell(), degree=deg, quad_scheme="default"
        )
        Quadelem._quad_scheme = "default"

        Telem2 = df.TensorElement(
            "df.Quadrature",
            mesh.ufl_cell(),
            degree=deg,
            shape=2 * (3,),
            quad_scheme="default",
        )
        Telem2._quad_scheme = "default"
        for e in Telem2.sub_elements():
            e._quad_scheme = "default"
        Telem4 = df.TensorElement(
            "df.Quadrature",
            mesh.ufl_cell(),
            degree=deg,
            shape=4 * (3,),
            quad_scheme="default",
        )
        Telem4._quad_scheme = "default"
        for e in Telem4.sub_elements():
            e._quad_scheme = "default"

        VRelem = df.MixedElement([Relem, Relem, Relem, Relem, Relem])

        if isincomp:
            if self.isLV:
                W = df.FunctionSpace(mesh, df.MixedElement([Velem, Qelem, Relem, VRelem]))
            else:
                W = df.FunctionSpace(
                    mesh, df.MixedElement([Velem, Qelem, Relem, Relem, VRelem])
                )
        else:
            if self.isLV:
                W = df.FunctionSpace(mesh, df.MixedElement([Velem, Relem, VRelem]))
            else:
                W = df.FunctionSpace(mesh, df.MixedElement([Velem, Relem, Relem, VRelem]))

        Quad = df.FunctionSpace(mesh, Quadelem)

        TF = df.FunctionSpace(mesh, Telem2)

        self.W = W
        self.Q = Q

        self.TF = TF

        self.we_n = df.Function(W.sub(0).collapse())

    def get_Jn_Cn(self):
        we_n = self.we_n

        d_n = we_n.ufl_domain().geometric_dimension()
        I_n = df.Identity(d_n)
        F_n = I_n + df.grad(we_n)
        C_n = F_n.T * F_n

        Ic_n = tr(C_n)
        J_n = df.det(F_n)

        return (J_n, C_n)

    def get_Fn(self):
        we_n = self.we_n

        d_n = we_n.ufl_domain().geometric_dimension()
        I_n = df.Identity(d_n)
        F_n = I_n + df.grad(we_n)

        return F_n

    def set_BCs(self, bivMesh_o, SimDet):
        facetboundaries = bivMesh_o.facetboundaries
        edgeboundaries = bivMesh_o.edgeboundaries
        topid = bivMesh_o.topid
        W = self.W

        bctop = df.DirichletBC(
            W.sub(0).sub(2), df.Constant(0.0), facetboundaries, topid
        )

        if "springbc" in list(SimDet.keys()) and SimDet["springbc"]:
            bcs = [bctop]
        else:
            bcs = [bctop]

        return bcs






class FHN(object):
    """
    FHN equations object

    Essentially three variables
    phi : action potential, normalized, PDE
    r   : inhibitory variable, ODE
    Ta  : active stress, ODE

    Better to have a seperate object for active stress Ta
    """

    def default_parameters(self):
        return {"outputfolder": "../Outputs/", "foldername": "NashPanfilov_BiV_04"}


    def update_parameters(self, params):
        self.parameters.update(params)


    def __init__(self, biVMesh, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)

        mesh = biVMesh.mesh
        dx = biVMesh.dx
        ds = biVMesh.ds

        P1_fhn = df.FiniteElement("CG", mesh.ufl_cell(), 1, quad_scheme="default")
        P1_fhn._quad_scheme = "default"

        W_fhn = df.FunctionSpace(mesh, df.MixedElement([P1_fhn, P1_fhn, P1_fhn]))

        w_fhn = df.Function(W_fhn)
        dw_fhn = df.TrialFunction(W_fhn)
        wtest_fhn = df.TestFunction(W_fhn)

        self.W_fhn = W_fhn
        self.w_fhn = w_fhn
        self.dw_fhn = dw_fhn
        self.wtest_fhn = wtest_fhn

        w_n_fhn = self.set_ICs()

        self.w_n_fhn = w_n_fhn
        phi_n, r_n, Ta_n_fhn = df.split(self.w_n_fhn)
        phi, r, Ta_fhn = df.split(w_fhn)

        bcs_FHN = self.set_BCs()

        self.bcs_FHN = bcs_FHN

        self.phi = phi
        self.r = r
        self.Ta_fhn = Ta_fhn

        self.phi_n = phi_n
        self.r_n = r_n
        self.Ta_n_fhn = Ta_n_fhn

        phi_test, r_test, Ta_test_fhn = df.split(wtest_fhn)
        self.phi_test = phi_test
        self.r_test = r_test
        self.Ta_test_fhn = Ta_test_fhn

        gradPhi_testFunc = VectorFunctionSpace(mesh, "CG", 1)
        gradPhi_test = df.Function(gradPhi_testFunc)
        self.gradPhi_test = gradPhi_test

        folderName = self.parameters["foldername"].rstrip("/")
        outputfolder = self.parameters["outputfolder"]
        output_dir = os.path.join(outputfolder, folderName)
        caseID = self.parameters["caseID"]

        vtkfile_phi = df.File(os.path.join(output_dir, f"{caseID}solution_phi.pvd"))
        vtkfile_r = df.File(os.path.join(output_dir, f"{caseID}solution_r.pvd"))
        vtkfile_Ta_fhn = df.File(
            os.path.join(output_dir, f"{caseID}solution_Ta_fhn.pvd")
        )

        fdataECG = open(
            os.path.join(output_dir, f"{caseID}IntegralCharge_.txt"), "w", 0
        )
        self.fdataECG = fdataECG

        self.vtkfile_phi = vtkfile_phi
        self.vtkfile_r = vtkfile_r
        self.vtkfile_Ta_fhn = vtkfile_Ta_fhn

        self.Dmat = self.parameters["Diffusion_Tensor"]

        stimPoint = self.parameters["Stimulus_Point"]
        stimCondition = self.parameters["Stimulus_Condition"]

        if stimCondition == "apex":
            stimuLusString = "(x[2] <= -6.44120193470301 ) ? 1*isStim : 0.0"
        elif stimCondition == "LVFW":
            stimuLusString = "(x[0] >= 19.199910386985) ? 1*isStim : 0.0"
        else:
            stimuLusString = "( (x[0] > -0.3 + {x0} ) && (x[0] < 0.3 + {x0}) && (x[1] > -0.3 + {x1}) && (x[1] < 0.3 + {x1}) && (x[2] > - 0.3 + {x2}) && (x[2] < 0.3 + {x2}) ) ? 10.0*isStim : 0.0".format(
                x0=stimPoint[0], x1=stimPoint[1], x2=stimPoint[2]
            )

        print(stimuLusString)
        self.stimuLus = SimpleNamespace(isStim=df.Constant(1.0))
        x = df.SpatialCoordinate(mesh)
        if stimCondition == "apex":
            stim_mask = df.conditional(x[2] <= -6.44120193470301, 1.0, 0.0)
        elif stimCondition == "LVFW":
            stim_mask = df.conditional(x[0] >= 19.199910386985, 1.0, 0.0)
        else:
            x0, x1, x2 = stimPoint
            half = 0.3
            cond_x = df.And(df.gt(x[0], -half + x0), df.lt(x[0], half + x0))
            cond_y = df.And(df.gt(x[1], -half + x1), df.lt(x[1], half + x1))
            cond_z = df.And(df.gt(x[2], -half + x2), df.lt(x[2], half + x2))
            stim_mask = df.conditional(df.And(df.And(cond_x, cond_y), cond_z), 10.0, 0.0)
        self.f_1 = self.stimuLus.isStim * stim_mask

        self.mesh = mesh
        self.dx = dx
        self.ds = ds


    def update_state(self, w_new):
        self.w_n_fhn = w_new


    def set_BCs(self):
        W_fhn = self.W_fhn
        bcs_FHN = []

        return bcs_FHN


    def set_ICs(self):
        W_fhn = self.W_fhn
        phi0 = df.interpolate(df.Constant(0.0), W_fhn.sub(0).collapse())
        r0 = df.interpolate(df.Constant(0.0), W_fhn.sub(1).collapse())
        Ta0_fhn = df.interpolate(df.Constant(0.0), W_fhn.sub(2).collapse())

        w_n_fhn = df.Function(W_fhn)
        assign(w_n_fhn, [phi0, r0, Ta0_fhn])

        return w_n_fhn


    def update_state_w_n(self, w_n_fhn):
        self.w_n_fhn = w_n_fhn
        phi_n, r_n, Ta_n_fhn = df.split(self.w_n_fhn)

        self.phi_n = phi_n
        self.r_n = r_n
        self.Ta_n_fhn = Ta_n_fhn


    def FHN_F_J(self, J_fhn=None, C_fhn=None):
        phi = self.phi
        r = self.r
        Ta_fhn = self.Ta_fhn

        phi_n = self.phi_n
        r_n = self.r_n
        Ta_n_fhn = self.Ta_n_fhn

        phi_test = self.phi_test
        r_test = self.r_test
        Ta_test_fhn = self.Ta_test_fhn
        mesh = self.mesh

        if J_fhn == None:
            J_n = df.Constant(1.0)
        else:
            J_n = J_fhn

        if C_fhn == None:
            C_n = df.Identity(mesh.ufl_cell().geometric_dimension())
        else:
            C_n = C_fhn

        myC = Constant_definitions()

        alpha = myC.parameters["alpha"]
        g = myC.parameters["gamma"]
        b = myC.parameters["b"]
        c = myC.parameters["c"]
        mu1 = myC.parameters["mu1"]
        mu2 = myC.parameters["mu2"]

        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))



        f_phi = f_phi(phi, r)
        f_r = f_r(phi, r)

        e0 = df.Constant(1.0)
        phi_mid = df.Constant(0.1)
        ephi = df.conditional(gt(phi, phi_mid), e0, 10 * e0)
        kTa = df.Constant(2 * 84000.0)

        f_1 = self.f_1

        timeDt = self.parameters["time_step"]
        k = df.Constant(timeDt)

        Dmat = self.parameters["Diffusion_Tensor"]
        dx = self.dx

        F_FHN = (
            ((phi - phi_n) / k) * phi_test * dx
            + 1 / J_n * df.dot(J_n * df.inv(C_n) * Dmat * df.grad(phi), df.grad(phi_test)) * dx
            - f_phi * phi_test * dx
            + ((r - r_n) / k) * r_test * dx
            - f_r * r_test * dx
            + ((Ta_fhn - Ta_n_fhn) / k) * Ta_test_fhn * dx
            - ephi * (kTa * phi - Ta_fhn) * Ta_test_fhn * dx
            - f_1 * phi_test * dx
        )

        J_FHN = derivative(F_FHN, self.w_fhn, self.dw_fhn)

        return (F_FHN, J_FHN)


    def write_state(self, t, tau):
        outputfolder = self.parameters["outputfolder"]
        foldername = self.parameters["foldername"]

        vtkfile_phi = self.vtkfile_phi
        vtkfile_r = self.vtkfile_r
        vtkfile_Ta_fhn = self.vtkfile_Ta_fhn

        w_n_fhn = self.w_n_fhn
        phi_, r_, Ta_fhn_ = w_n_fhn.df.split(deepcopy=True)

        print(("Time = : %0.2f, tau = : %0.2f" % (t, tau)))
        print("Tau is arbitrary")
        vtkfile_phi << phi_
        vtkfile_r << r_
        vtkfile_Ta_fhn << Ta_fhn_

        qVec = [df.assemble(df.grad(phi_)[i] * dx) for i in range(3)]
        print(qVec[:])
        qVecNorm = np.linalg.norm(qVec)
        print(t, qVec[0], qVec[1], qVec[2], qVecNorm, file=self.fdataECG)

    def advance_timeStepssolver_FHN(self, solver_FHN, fhn_steps):
        for ii in range(0, fhn_steps):
            print("Solving FHN")
            solver_FHN.solvenonlinear()
            self.w_n_fhn.assign(self.w_fhn)
            self.update_state_w_n(self.w_n_fhn)




class State_Variables(object):
    """
    State variables for LV and RV, ejecting, relaxing, and filling
    """

    def __init__(self, mpi_comm, SimDet):
        self.isLV = SimDet["isLV"]

        self.isLVeject = 0
        self.isLVfill = 0
        self.isLVfilling = False
        self.isLVejecting = False
        self.isLVrelaxing = False

        if not self.isLV:
            self.isRVeject = 0
            self.isRVfill = 0
            self.isRVfilling = False
            self.isRVejecting = False
            self.isRVrelaxing = False

        self.tstep = 0
        self.BCL = SimDet["HeartBeatLength"]

        self.cycle = 0.0
        self.t = 0
        self.tstep = 0
        self.dt = SimpleNamespace(dt=0.0)

        self.comm = mpi_comm

    def check_isnot_isovolumic(self):
        if self.isLV:
            return bool(self.isLVeject) or bool(self.isLVfill)
        else:
            return (
                bool(self.isLVeject)
                or bool(self.isRVeject)
                or bool(self.isLVfill)
                or bool(self.isRVfill)
            )

    def update_state(self, pv_o, circuit_o, t_a):
        mmhg = 0.0075
        dt_val = self.dt.dt
        bcl_remaining = self.BCL - t_a.t_a
        LVP_mmHg = pv_o.LVP_cav * mmhg
        if self.isLVejecting and circuit_o.Qlv < 0.005:
            self.isLVeject = 0
            self.isLVejecting = False
            self.isLVrelaxing = True

            pv_o.LVESV = pv_o.LVV_cav

        if LVP_mmHg > circuit_o.Pao and not self.isLVrelaxing:
            self.isLVeject = 1
            self.isLVejecting = True

        if not self.isLV:
            RVP_mmHg = pv_o.RVP_cav * mmhg
            if self.isRVejecting and circuit_o.Qrv < 0.005:
                self.isRVeject = 0
                self.isRVejecting = False
                self.isRVrelaxing = True

                pv_o.RVESV = pv_o.RVV_cav

            if RVP_mmHg > circuit_o.Ppu and not self.isRVrelaxing:
                self.isRVeject = 1
                self.isRVejecting = True

        if LVP_mmHg < circuit_o.Pmv and self.isLVrelaxing and pv_o.LVPfilling_step == 0:
            self.isLVfill = 1
            self.isLVfilling = True

            denom = bcl_remaining or 1.0
            pv_o.LVPfilling_step = (
                (pv_o.LVEDP - pv_o.LVP_cav) / denom * dt_val
            )
            pv_o.LVVfilling_step = (
                (pv_o.LVEDV - pv_o.LVESV) / denom * dt_val
            )

        if not self.isLV:
            if (
                RVP_mmHg < circuit_o.Ptr
                and self.isRVrelaxing
                and pv_o.RVPfilling_step == 0
            ):
                self.isRVfill = 1
                self.isRVfilling = True

                denom = bcl_remaining or 1.0
                pv_o.RVPfilling_step = (
                    (pv_o.RVEDP - pv_o.RVP_cav) / denom * dt_val
                )
                pv_o.RVVfilling_step = (
                    (pv_o.RVEDV - pv_o.RVESV) / denom * dt_val
                )

    def print_time_details(self):
        string_to_print = (
            "Cycle number = "
            + str(self.cycle)
            + " cell time = "
            + str(self.t)
            + " tstep = "
            + str(self.tstep)
            + " dt = "
            + str(self.dt.dt)
        )
        printout(string_to_print, self.comm)

    def print_LV_state(self):
        string_to_print = (
            "isLVejecting: "
            + str(self.isLVejecting)
            + " isLVrelaxing: "
            + str(self.isLVrelaxing)
            + " isLVfilling: "
            + str(self.isLVfilling)
        )
        printout(string_to_print, self.comm)

    def print_RV_state(self):
        string_to_print = (
            "isRVejecting: "
            + str(self.isRVejecting)
            + " isRVrelaxing: "
            + str(self.isRVrelaxing)
            + " isRVfilling: "
            + str(self.isRVfilling)
        )
        printout(string_to_print, self.comm)




class Windkessel(object):
    def __init__(self, elas_o, mpi_comm, SimDet):
        self.Pao = SimDet["Pao"]  # 50 # LV ejection start
        self.isLV = SimDet["isLV"]
        if not self.isLV:
            self.Ppu = SimDet["Ppu"]  # 12 # RV ejection start

        self.Qlv = 0.0
        self.Qlv_prev = 0.0

        self.Qrv = 0.0
        self.Qrv_prev = 0.0

        self.Cmpl_Ao = 0.007 * 1.2  # 0.01
        self.Cmpl_Pa = 0.01  # 0.005

        self.Rper_sys = 100000 * 1.2  # 100000 MRC latest results
        self.Rper_pul = 60000

        self.Rao = 5500 * 1.2  # 800
        self.Rpa = 500

        self.Pper_sys = SimDet["Pper_sys"]  # 600 #(Pa)
        self.Pper_pul = SimDet["Pper_pul"]  # 200 # RV pressure

        self.Pmv = SimDet["Pmv"]  # 5 #mmHg
        self.Ptr = SimDet["Ptr"]  # 4 #mmHg

        if self.isLV:
            self.F = 0
            self.Vdiff = 0
            self.dFdV = 0
            self.Ch = 0
        else:
            self.F = np.zeros((2, 1))
            self.Vdiff = np.zeros((2, 1))
            self.dFdV = np.array([[0, 0], [0, 0]])
            self.Ch = np.array([[0, 0], [0, 0]])

        W_elas = elas_o.W
        self.w_cur = df.Function(W_elas)

        self.normVdiff = 1e8
        self.normFdiff = 1e8

        self.comm = mpi_comm

    def calculate_F_Q(self, state_o, pv_o):
        Pper_sys = self.Pper_sys
        Pper_pul = self.Pper_pul

        Rper_sys = self.Rper_sys
        Rper_pul = self.Rper_pul

        Cmpl_Ao = self.Cmpl_Ao
        Cmpl_Pa = self.Cmpl_Pa

        Qlv = self.Qlv
        Qlv_prev = self.Qlv_prev

        if not self.isLV:
            Qrv = self.Qrv
            Qrv_prev = self.Qrv_prev

        Rao = self.Rao
        Rpa = self.Rpa

        isLVeject = state_o.isLVeject
        isLVfill = state_o.isLVfill
        if not self.isLV:
            isRVeject = state_o.isRVeject
            isRVfill = state_o.isRVfill

        LVP_cav = pv_o.LVP_cav
        LVV_cav = pv_o.LVV_cav

        LVP_cav_prev = pv_o.LVP_cav_prev
        LVV_cav_prev = pv_o.LVV_cav_prev

        if not self.isLV:
            RVP_cav = pv_o.RVP_cav
            RVV_cav = pv_o.RVV_cav

            RVP_cav_prev = pv_o.RVP_cav_prev
            RVV_cav_prev = pv_o.RVV_cav_prev

        LVVfilling_step = pv_o.LVVfilling_step
        LVPfilling_step = pv_o.LVPfilling_step
        if not self.isLV:
            RVVfilling_step = pv_o.RVVfilling_step
            RVPfilling_step = pv_o.RVPfilling_step

        if bool(isLVeject):
            Qlv = (
                (LVP_cav - Pper_sys) / (Rper_sys)
                + Cmpl_Ao * (LVP_cav - LVP_cav_prev + Rao * Qlv_prev) / state_o.dt.dt
            ) / (1.0 + Rao / Rper_sys + Cmpl_Ao * Rao / state_o.dt.dt)
            if self.isLV:
                self.F = LVV_cav - LVV_cav_prev + state_o.dt.dt * Qlv
            else:
                self.F[0] = LVV_cav - LVV_cav_prev + state_o.dt.dt * Qlv
        elif bool(isLVfill):
            printout("LVPfilling_step = " + str(LVPfilling_step), self.comm)
            if self.isLV:
                self.F = LVP_cav - LVP_cav_prev - LVPfilling_step
            else:
                self.F[0] = LVP_cav - LVP_cav_prev - LVPfilling_step

        else:
            Qlv = 0.0
            if self.isLV:
                self.F = LVV_cav - LVV_cav_prev
            else:
                self.F[0] = LVV_cav - LVV_cav_prev

        if not self.isLV:
            if bool(isRVeject):
                Qrv = (
                    (RVP_cav - Pper_pul) / (Rper_pul)
                    + Cmpl_Pa
                    * (RVP_cav - RVP_cav_prev + Rpa * Qrv_prev)
                    / state_o.dt.dt
                ) / (1.0 + Rpa / Rper_pul + Cmpl_Pa * Rpa / state_o.dt.dt)
                self.F[1] = RVV_cav - RVV_cav_prev + state_o.dt.dt * Qrv
            elif bool(isRVfill):
                printout("RVPfilling_step = " + str(RVPfilling_step), self.comm)
                self.F[1] = RVP_cav - RVP_cav_prev - RVPfilling_step

            else:
                Qrv = 0.0
                self.F[1] = RVV_cav - RVV_cav_prev

        if self.isLV:
            printout("Qlv = " + str(Qlv), self.comm)
        else:
            printout("Qlv = " + str(Qlv) + " Qrv = " + str(Qrv), self.comm)

        self.normFdiff = np.linalg.norm(self.F)

        if self.isLV:
            string_to_print = f"Fdiff = {self.normFdiff} F[0] = {self.F}"
        else:
            string_to_print = f"Fdiff = {self.normFdiff} F[0] = {self.F[0]} F[1] = {self.F[1]}"

        printout(string_to_print, self.comm)

        self.Qlv = Qlv
        if not self.isLV:
            self.Qrv = Qrv

        self.AorPres = LVP_cav - LVP_cav_prev + Rao * Qlv_prev
        if not self.isLV:
            self.PulPres = RVP_cav - RVP_cav_prev + Rpa * Qrv_prev


    def set_dFdV(self, state_o, pv_o):
        Pper_sys = self.Pper_sys
        Pper_pul = self.Pper_pul

        Rper_sys = self.Rper_sys
        Rper_pul = self.Rper_pul

        Cmpl_Ao = self.Cmpl_Ao
        Cmpl_Pa = self.Cmpl_Pa

        Qlv = self.Qlv
        Qlv_prev = self.Qlv_prev

        Qrv = self.Qrv
        Qrv_prev = self.Qrv_prev

        Rao = self.Rao
        Rpa = self.Rpa

        isLVeject = state_o.isLVeject
        isLVfill = state_o.isLVfill
        if not self.isLV:
            isRVeject = state_o.isRVeject
            isRVfill = state_o.isRVfill

        Ch = self.Ch

        if self.isLV:
            dFdV = (
                1.0
                + float(isLVeject)
                * Ch
                * (state_o.dt.dt / Rper_sys + Cmpl_Ao)
                / (1.0 + Rao / Rper_sys + Cmpl_Ao * Rao / state_o.dt.dt)
                + float(isLVfill) * Ch
            )
        else:
            dFdV11 = (
                1.0
                + float(isLVeject)
                * Ch[0, 0]
                * (state_o.dt.dt / Rper_sys + Cmpl_Ao)
                / (1.0 + Rao / Rper_sys + Cmpl_Ao * Rao / state_o.dt.dt)
                + float(isLVfill) * Ch[0, 0]
            )
            dFdV12 = (
                float(isLVeject)
                * Ch[0, 1]
                * (state_o.dt.dt / Rper_sys + Cmpl_Ao)
                / (1.0 + Rao / Rper_sys + Cmpl_Ao * Rao / state_o.dt.dt)
                + float(isLVfill) * Ch[0, 1]
            )
            dFdV21 = (
                float(isRVeject)
                * Ch[1, 0]
                * (state_o.dt.dt / Rper_pul + Cmpl_Pa)
                / (1.0 + Rpa / Rper_pul + Cmpl_Pa * Rpa / state_o.dt.dt)
                + float(isRVfill) * Ch[1, 0]
            )
            dFdV22 = (
                1.0
                + float(isRVeject)
                * Ch[1, 1]
                * (state_o.dt.dt / Rper_pul + Cmpl_Pa)
                / (1.0 + Rpa / Rper_pul + Cmpl_Pa * Rpa / state_o.dt.dt)
                + float(isRVfill) * Ch[1, 1]
            )

            dFdV = np.matrix([[dFdV11, dFdV12], [dFdV21, dFdV22]])
        self.dFdV = dFdV


    def update_Q(self):
        self.Qlv_prev = self.Qlv
        if not self.isLV:
            self.Qrv_prev = self.Qrv

    def computeCompliance(self, solver_elas, pv_o, LVCavityvol, RVCavityvol, uflforms):
        LVP_cav = pv_o.LVP_cav
        LVV_cav = pv_o.LVV_cav

        LVP_cav_prev = pv_o.LVP_cav_prev
        LVV_cav_prev = pv_o.LVV_cav_prev

        if not self.isLV:
            RVP_cav = pv_o.RVP_cav
            RVV_cav = pv_o.RVV_cav

        w_cur = self.w_cur
        w = solver_elas.parameters["w"]

        printout("######################################################", self.comm)
        printout("Calculating compliance ", self.comm)
        w_cur.vector()[:] = w.vector().array()[:]
        dV = 0.01
        _update_cavity_control_value(LVCavityvol, LVV_cav + dV, "vol")
        if not self.isLV:
            _update_cavity_control_value(RVCavityvol, RVV_cav, "vol")
        solver_elas.solvenonlinear()

        if self.isLV:
            dPlv_dVlv = (uflforms.LVcavitypressure() - LVP_cav) / dV
            printout("dPlv_dVlv = " + str(dPlv_dVlv), self.comm)
        else:
            dPrv_dVlv = (uflforms.RVcavitypressure() - RVP_cav) / dV
            printout(
                "dPlv_dVlv = " + str(dPlv_dVlv) + " dPrv_dVlv = " + str(dPrv_dVlv),
                self.comm,
            )

        w.vector()[:] = w_cur.vector().array()[:]
        _update_cavity_control_value(LVCavityvol, LVV_cav, "vol")
        if not self.isLV:
            _update_cavity_control_value(RVCavityvol, RVV_cav + dV, "vol")
            solver_elas.solvenonlinear()

            dPlv_dVrv = (uflforms.LVcavitypressure() - LVP_cav) / dV
            dPrv_dVrv = (uflforms.RVcavitypressure() - RVP_cav) / dV
            w.vector()[:] = w_cur.vector().array()[:]
            printout(
                "dPlv_dVrv = " + str(dPlv_dVrv) + " dPrv_dVrv = " + str(dPrv_dVrv),
                self.comm,
            )
            printout(
                "######################################################", self.comm
            )

            _update_cavity_control_value(RVCavityvol, RVV_cav, "vol")

        if self.isLV:
            self.Ch = dPlv_dVlv
        else:
            self.Ch = np.array([[dPlv_dVlv, dPlv_dVrv], [dPrv_dVlv, dPrv_dVrv]])




class PV_Ventricles(object):
    def __init__(self, uflforms, mpi_comm, SimDet):
        self.isLV = SimDet["isLV"]

        LVV_unload = uflforms.LVcavityvol()
        if not self.isLV:
            RVV_unload = uflforms.RVcavityvol()

        prescribedLVEDV_Shift = SimDet["LVEDV_Shift"]
        if not self.isLV:
            prescribedRVEDV_Shift = SimDet["RVEDV_Shift"]

        LVEDV = LVV_unload + prescribedLVEDV_Shift  # 30
        if not self.isLV:
            RVEDV = RVV_unload + prescribedRVEDV_Shift  # 38



        LVESV = 0
        RVESV = 0

        self.LVV_unload = LVV_unload
        if not self.isLV:
            self.RVV_unload = RVV_unload

        self.LVEDV = LVEDV
        self.LVESV = LVESV
        if not self.isLV:
            self.RVEDV = RVEDV
            self.RVESV = RVESV

        LVEDP = 0.0
        LVESP = 0.0
        if not self.isLV:
            RVEDP = 0.0
            RVESP = 0.0

        self.LVEDP = LVEDP
        self.LVESP = LVESP
        if not self.isLV:
            self.RVEDP = RVEDP
            self.RVESP = RVESP

        LVP_cav = 0.0
        LVV_cav = 0.0
        if not self.isLV:
            RVP_cav = 0.0
            RVV_cav = 0.0

        LVP_cav_prev = 0.0
        LVV_cav_prev = 0.0
        if not self.isLV:
            RVP_cav_prev = 0.0
            RVV_cav_prev = 0.0

        self.LVP_cav = LVP_cav
        self.LVV_cav = LVV_cav
        if not self.isLV:
            self.RVP_cav = RVP_cav
            self.RVV_cav = RVV_cav

        self.LVP_cav_prev = LVP_cav_prev
        self.LVV_cav_prev = LVV_cav_prev
        if not self.isLV:
            self.RVP_cav_prev = RVP_cav_prev
            self.RVV_cav_prev = RVV_cav_prev

        LVVfilling_step = 0
        LVPfilling_step = 0
        if not self.isLV:
            RVVfilling_step = 0
            RVPfilling_step = 0

        self.LVVfilling_step = LVVfilling_step
        self.LVPfilling_step = LVPfilling_step
        if not self.isLV:
            self.RVVfilling_step = RVVfilling_step
            self.RVPfilling_step = RVPfilling_step

        self.comm = mpi_comm

    def update_pv(self, uflforms):
        self.LVP_cav = uflforms.LVcavitypressure()
        self.LVV_cav = uflforms.LVcavityvol()
        if not self.isLV:
            self.RVP_cav = uflforms.RVcavitypressure()
            self.RVV_cav = uflforms.RVcavityvol()

    def update_pv_prev(self):
        self.LVP_cav_prev = self.LVP_cav
        self.LVV_cav_prev = self.LVV_cav
        if not self.isLV:
            self.RVP_cav_prev = self.RVP_cav
            self.RVV_cav_prev = self.RVV_cav

    def print_PV_details(self):
        if not self.isLV:
            printout(
                "LVP_cav = "
                + str(self.LVP_cav * 0.0075)
                + " LVV_cav = "
                + str(self.LVV_cav)
                + " RVP_cav = "
                + str(self.RVP_cav * 0.0075)
                + " RVV_cav = "
                + str(self.RVV_cav),
                self.comm,
            )
        else:
            printout(
                "LVP_cav = "
                + str(self.LVP_cav * 0.0075)
                + " LVV_cav = "
                + str(self.LVV_cav),
                self.comm,
            )




class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type(df.Constant(1.0))):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class exportfiles(object):
    def __init__(self, mpi_comm_me, mpi_comm_ep, IODet, SimDet):
        if "isLV" in list(SimDet.keys()):
            self.isLV = SimDet["isLV"]
        else:
            self.isLV = False  # Default
        if "iswaorta" in list(SimDet.keys()):
            self.iswaorta = SimDet["iswaorta"]
        else:
            self.iswaorta = False  # Default
        if "isFCH" in list(SimDet.keys()):
            self.isFCH = SimDet["isFCH"]
        else:
            self.isFCH = False  # Default
        if "isBiV" in list(SimDet.keys()):
            self.isBiV = SimDet["isBiV"]
        else:
            self.isBiV = False  # Default

        self.outputfolder = IODet["outputfolder"]
        self.folderName = os.path.join(IODet["folderName"], IODet["caseID"])
        self.SimDet = SimDet
        self.IODet = IODet

        self.comm_me = mpi_comm_me
        self.comm_ep = mpi_comm_ep

        if not SimDet.get("isrestart"):
            self.removeAllfiles()
        self.opendatafilestreams()
        self.openHDF5filestreams()

    def dump_input_file(self):
        with open(os.path.join(self.outputfolder, self.folderName, "SimDet.log"), "w") as f:
            json.dump(self.SimDet, f, indent=4, cls=json_serialize)
        with open(os.path.join(self.outputfolder, self.folderName, "IODet.log"), "w") as f:
            json.dump(self.IODet, f, indent=4, cls=json_serialize)


    def dump_restart_file(self, CLmodel, V_LV, V_RV, V_LA, V_RA):

        isBiV = False
        if "isBiV" in list(self.SimDet.keys()):
            isBiV = self.SimDet["isBiV"]

        if "isFCH" in list(self.SimDet.keys()):
            isFCH = self.SimDet["isFCH"]

        SimDetCopy = self.SimDet
        SimDetCopy["closedloopparam"]["V_sa"] = CLmodel.V_sa 
        SimDetCopy["closedloopparam"]["V_ad"] = CLmodel.V_ad 
        SimDetCopy["closedloopparam"]["V_sv"] = CLmodel.V_sv 
        SimDetCopy["closedloopparam"]["V_LA"] = V_LA
        SimDetCopy["closedloopparam"]["V_LV"] = V_LV

        if isBiV or isFCH: 
            SimDetCopy["closedloopparam"]["V_pa"] = CLmodel.V_pa
            SimDetCopy["closedloopparam"]["V_pv"] = CLmodel.V_pv 
            SimDetCopy["closedloopparam"]["V_RV"] = V_RV
            SimDetCopy["closedloopparam"]["V_RA"] = V_RA

        with open(
            os.path.join(self.outputfolder, self.folderName, "SimDetRestart.log"), "w"
        ) as f:
            json.dump(SimDetCopy, f, indent=4, cls=json_serialize)



    def removeAllfiles(self):
        base = Path(self.outputfolder) / self.folderName

        patterns = ["*.pvd", "*.vtu", "*.pvtu", "*.txt", "*.hdf5", "*.xdmf"]
        for pat in patterns:
            for f in base.glob(pat):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass

        for f in base.glob("*.h5"):
            if f.name == "Data_r.h5":
                continue
            try:
                f.unlink()
            except FileNotFoundError:
                pass

        subdirs = [
            "FHN",
            "IMP",
            "IMP2",
            "IMP_J",
            "IMP_Constraint",
            "FHN_coarse",
            "FHN_coarse2",
            "FHN_coarseRef",
            "deformation",
            "deformation_loadED",
            "deformation_unloadED",
            "stretch",
            "active",
        ]
        for d in subdirs:
            target = base / d
            if target.exists():
                shutil.rmtree(str(target), ignore_errors=True)

    def exportVTKobj(self, filename, vtkobj):
        outputfolder = self.outputfolder
        folderName = self.folderName
        df.File(os.path.join(outputfolder, folderName, filename)) << vtkobj

        return

    def opendatafilestreams(self):
        outputfolder = self.outputfolder
        folderName = self.folderName
        output_dir = os.path.join(outputfolder, folderName)
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            self.fdatalog = open(os.path.join(output_dir, "BiV_log.txt"), "w", -1)
            self.fdataPV = open(os.path.join(output_dir, "BiV_PV.txt"), "w", -1)
            self.fdataQ = open(os.path.join(output_dir, "BiV_Q.txt"), "w", -1)
            self.fdataP = open(os.path.join(output_dir, "BiV_P.txt"), "w", -1)
            self.fdataV = open(os.path.join(output_dir, "BiV_V.txt"), "w", -1)
            self.fdatatpt = open(os.path.join(output_dir, "tpt.txt"), "w", -1)

            self.fdataIMP = open(os.path.join(output_dir, "BiV_IMP.txt"), "w", -1)
            self.fdataIMP2 = open(os.path.join(output_dir, "BiV_IMP2.txt"), "w", -1)
            self.fdataIMP3 = open(
                os.path.join(output_dir, "BiV_IMP_InC.txt"), "w", -1
            )

            self.fdataStress = open(os.path.join(output_dir, "BiV_fiberStress.txt"), "w", -1)
            self.fdataStrain = open(os.path.join(output_dir, "BiV_fiberStrain.txt"), "w", -1)
            self.fdataWork = open(os.path.join(output_dir, "BiV_fiberWork.txt"), "w", -1)

            self.fdata_C_Strain = open(os.path.join(output_dir, "BiV_CStrain.txt"), "w", -1)
            self.fdata_L_Strain = open(os.path.join(output_dir, "BiV_LStrain.txt"), "w", -1)
            self.fdata_R_Strain = open(os.path.join(output_dir, "BiV_RStrain.txt"), "w", -1)

        return

    def openVTKfilestreams(self):
        outputfolder = self.outputfolder
        folderName = self.folderName
        output_dir = os.path.join(outputfolder, folderName)

        self.vtkfile_phi = df.File(os.path.join(output_dir, "FHN", "phi.pvd"))
        self.vtkfile_phi_ref = df.File(
            os.path.join(output_dir, "FHN_coarseRef", "phi_ref.pvd")
        )
        self.vtkfile_phi_me = df.File(os.path.join(output_dir, "FHN_coarse", "phi_me.pvd"))
        self.vtkfile_r = df.File(os.path.join(output_dir, "FHN", "r.pvd"))

        self.vtkfile_IMP = df.File(os.path.join(output_dir, "IMP", "IMP.pvd"))
        self.vtkfile_IMP2 = df.File(os.path.join(output_dir, "IMP2", "IMP2.pvd"))
        self.vtkfile_IMP_Constraint = df.File(
            os.path.join(output_dir, "IMP_Constraint", "IMP_Constraint.pvd")
        )

        self.displacementfile = df.File(os.path.join(output_dir, "deformation", "u_disp.pvd"))
        self.displacementED_file = df.File(
            os.path.join(output_dir, "deformation_loadED", "u_disp.pvd")
        )

        self.activationFile = df.File(
            os.path.join(output_dir, "active", "activationTime.pvd")
        )
        self.vtkfile_Ecc = df.File(os.path.join(output_dir, "active", "Ecc.pvd"))
        self.vtkfile_Ell = df.File(os.path.join(output_dir, "active", "Ell.pvd"))
        self.vtkfile_Err = df.File(os.path.join(output_dir, "active", "Err.pvd"))
        self.vtkfile_Efiber = df.File(os.path.join(output_dir, "active", "Eff.pvd"))
        self.vtkfile_fstress = df.File(os.path.join(output_dir, "active", "fstress.pvd"))

        return

    def openHDF5filestreams(self):
        outputfolder = self.outputfolder
        folderName = self.folderName
        comm_me = self.comm_me
        comm_ep = self.comm_ep

        self.hdf = SafeH5Wrapper(
            df.HDF5File(comm_me, os.path.join(outputfolder, folderName, "Data.h5"), "w")
        )

        return

    def closeHDF5filestreams(self):
        if hasattr(self, "hdf") and self.hdf:
            self.hdf.close()

    def writePV(self, MEmodel, t):
        isLV = self.isLV
        iswaorta = self.iswaorta
        isFCH = self.isFCH
        isBiV = self.isBiV
        ispctrl = getattr(MEmodel, "ispctrl", None)

        comm = self.comm_ep

        if ispctrl:
            LVP = float(MEmodel.LVCavitypres) * 0.0075
            LVV = MEmodel.get_lv_volume()
            if isBiV or isFCH:
                RVP = float(MEmodel.RVCavitypres) * 0.0075
                RVV = MEmodel.get_rv_volume()
        else:
            LVP = MEmodel.get_lv_pressure() * 0.0075
            LVV = MEmodel.get_lv_volume()

            if not isLV:
                RVP = MEmodel.get_rv_pressure() * 0.0075
                RVV = MEmodel.get_rv_volume()

        if df.MPI.rank(comm) == 0:
            fdataPV = self.fdataPV
            if isLV:
                print(t, LVP, LVV, file=fdataPV, flush=True)
            elif iswaorta:
                print(t, LVP, LVV, file=fdataPV, flush=True)
            elif isFCH:
                print(t, LVP, LVV, RVP, RVV, file=fdataPV, flush=True)
            elif isBiV:
                print(t, LVP, LVV, RVP, RVV, file=fdataPV, flush=True)

        return

    def writeQ(self, MEmodel, Qarray, t):
        isLV = self.isLV
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataQ = self.fdataQ
            print(
                t, " ".join(map(lambda x: "%.5e" % x, Qarray)), file=fdataQ, flush=True
            )

        return

    def writeP(self, MEmodel, Parray, t):
        isLV = self.isLV
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataP = self.fdataP
            print(
                t, " ".join(map(lambda x: "%.5e" % x, Parray)), file=fdataP, flush=True
            )

        return

    def writeV(self, MEmodel, Varray, t):
        isLV = self.isLV
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataV = self.fdataV
            print(
                t, " ".join(map(lambda x: "%.5e" % x, Varray)), file=fdataV, flush=True
            )

        return

    def writetpt(self, MEmodel, tpt):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdatatpt = self.fdatatpt
            print(tpt, file=fdatatpt, flush=True)

        return

    def writeIMP(self, MEmodel, t, fIMP):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataIMP = self.fdataIMP
            print(t, " ".join(map(str, fIMP)), file=fdataIMP)

        return

    def writeIMP2(self, MEmodel, t, fIMP2):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataIMP2 = self.fdataIMP2
            print(t, " ".join(map(str, fIMP2)), file=fdataIMP2)

        return

    def writeIMP3(self, MEmodel, t, fIMP3):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataIMP3 = self.fdataIMP3
            print(t, " ".join(map(str, fIMP3)), file=fdataIMP3)

        return

    def writefStress(self, MEmodel, t, fStress):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataStress = self.fdataStress
            print(t, " ".join(map(str, fStress)), file=fdataStress)

        return

    def writefStrain(self, MEmodel, t, fStrain):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdataStrain = self.fdataStrain
            print(t, " ".join(map(str, fStrain)), file=fdataStrain)

        return

    def writeCStrain(self, MEmodel, t, CStrain):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdata_C_Strain = self.fdata_C_Strain
            print(t, " ".join(map(str, CStrain)), file=fdata_C_Strain)

        return

    def writeLStrain(self, MEmodel, t, LStrain):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdata_C_Strain = self.fdata_C_Strain
            print(t, " ".join(map(str, LStrain)), file=fdata_C_Strain)

        return

    def writeRStrain(self, MEmodel, t, RStrain):
        comm = self.comm_ep

        if df.MPI.rank(comm) == 0:
            fdata_C_Strain = self.fdata_C_Strain
            print(t, " ".join(map(str, RStrain)), file=fdata_C_Strain)

        return

    def write_datalog(self, statement):
        """Append a line to the run's on-disk data log.

        Named apart from the module-level ``printout``: this writes to a FILE HANDLE, not the
        log stream, and the shared name made two unrelated sinks indistinguishable at call sites.
        """
        comm = self.comm_ep
        if df.MPI.rank(comm) == 0:
            fdata_log = self.fdatalog
            print(statement, file=fdata_log, flush=True)
