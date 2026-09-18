import csv
import json
import os


PERIODICITY_LOG_HEADER = [
    "cycle",
    "t",
    "mode",
    "eps",
    "signal",
    "P_rel_l2",
    "P_rel_linf",
    "V_rel_l2",
    "V_rel_linf",
    "ok_signal",
    "ok_overall",
]


def append_periodicity_rows(log_path, rows):
    """
    Append rows to the periodicity log CSV, creating directories and headers as needed.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not (os.path.exists(log_path) and os.path.getsize(log_path) > 0)
    with open(log_path, "a", newline="") as lf:
        writer = csv.writer(lf)
        if write_header:
            writer.writerow(PERIODICITY_LOG_HEADER)
        writer.writerows(rows)


def write_periodicity_metrics_json(json_path, case_id, scalar_metrics):
    """
    Write summary metrics to JSON in a structured form for downstream tooling.
    """
    if not scalar_metrics:
        return
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # Payload structure is keyed by case_id to allow multi-case aggregation by external scripts.
    payload = {case_id: {}}
    for name, value in scalar_metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        keywords = [part for part in name.split("_") if part] or [name]
        payload[case_id][name] = {
            "keywords": keywords,
            "weight": numeric,
        }
    with open(json_path, "w") as jf:
        json.dump(payload, jf, indent=2, sort_keys=True)
