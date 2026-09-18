import dolfin as df
from numbers import Number


AORTA_DEFAULTS = {
    "NeoHookean": {
        "mu": df.Constant(50.0),
    },
    "Delfino": {
        "D1": df.Constant(33.78),
        "D2": df.Constant(1.73),
    },
    "HGO_twofiber": {
        "Cgr": df.Constant(41.82),
        "gamma": df.Constant(54.46),
        "C1": [df.Constant(1.78), df.Constant(1.78)],
        "C2": [df.Constant(2.61), df.Constant(2.61)],
    },
    "HGO_fourfiber": {
        "Cgr_ff": df.Constant(11.71),
        "gamma_ff": df.Constant(38.41),
        "C1_ff": [
            df.Constant(11.17),
            df.Constant(11.17),
            df.Constant(22.29),
            df.Constant(8.00),
        ],
        "C2_ff": [
            df.Constant(1.45),
            df.Constant(1.45),
            df.Constant(0.16),
            df.Constant(1.41),
        ],
    },
}

DEFAULT_AORTA_MODEL = "Delfino"


def _ensure_constant(value):
    # Normalize numeric parameters to dolfin.Constants so downstream UFL expressions remain differentiable.
    if isinstance(value, df.Constant):
        return value
    if isinstance(value, Number):
        return df.Constant(value)
    if isinstance(value, (list, tuple)):
        return [_ensure_constant(v) for v in value]
    return value


def _model_defaults(model_name):
    defaults = {}
    for key, value in AORTA_DEFAULTS.get(model_name, {}).items():
        defaults[key] = _ensure_constant(value)
    return defaults


def fill_aorta_params(overrides=None):
    overrides = overrides or {}
    model_name = overrides.get("Name", DEFAULT_AORTA_MODEL)
    # Merge order: model defaults -> user overrides (excluding Name); keeps a complete param dict per model.
    defaults = {"Name": model_name}
    defaults.update(_model_defaults(model_name))
    result = defaults.copy()
    for key, value in overrides.items():
        if key == "Name":
            continue
        result[key] = value
    return result


def default_aorta_params():
    return fill_aorta_params()
