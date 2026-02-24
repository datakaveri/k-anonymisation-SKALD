import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, List

import numpy as np

logger = logging.getLogger("SKALD")


class RustOla2Error(RuntimeError):
    pass


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _skald_ola2_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "skald_ola2")


def _rust_binary_path() -> str:
    env_bin = os.getenv("SKALD_RUST_OLA2_BIN")
    if env_bin:
        return env_bin
    return os.path.join(_skald_ola2_dir(), "target", "release", "ola2_full")


def rust_ola2_is_supported(quasi_identifiers: List, multiplication_factors: Dict[str, int]) -> bool:
    # Current Rust lattice assumes numeric-only and doubling factor per dimension.
    for qi in quasi_identifiers:
        if getattr(qi, "is_categorical", False):
            return False

        base_col = qi.column_name[:-8] if getattr(qi, "is_encoded", False) else qi.column_name
        if int(multiplication_factors.get(base_col, 0)) != 2:
            return False

    return True


def _ensure_rust_binary() -> str:
    binary = _rust_binary_path()
    source_file = os.path.join(_skald_ola2_dir(), "src", "bin", "ola2_full.rs")
    if os.path.isfile(binary) and os.path.isfile(source_file):
        if os.path.getmtime(binary) >= os.path.getmtime(source_file):
            return binary

    cmd = ["cargo", "build", "--release", "--bin", "ola2_full"]
    logger.info("Rust OLA2 binary missing; building with: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=_skald_ola2_dir(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RustOla2Error("cargo not found while building Rust OLA2 binary") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RustOla2Error(
            f"failed to build Rust OLA2 binary (code {proc.returncode}): {stderr or stdout}"
        )

    if not os.path.isfile(binary):
        raise RustOla2Error(f"Rust OLA2 binary was not produced at '{binary}'")
    return binary


def _compute_max_levels(max_bin_width: int) -> int:
    if max_bin_width <= 0:
        raise RustOla2Error(f"invalid max bin width: {max_bin_width}")
    return int(np.ceil(np.log2(max_bin_width))) + 1


def run_rust_ola2(
    global_hist: np.ndarray,
    initial_ri: List[int],
    quasi_identifiers: List,
    k: int,
    suppression_limit: float,
    total_records: int,
) -> Dict:
    if global_hist is None:
        raise RustOla2Error("global_hist cannot be None")
    if k <= 0:
        raise RustOla2Error("k must be > 0")
    if total_records <= 0:
        raise RustOla2Error("total_records must be > 0")
    if not (0 <= suppression_limit <= 1):
        raise RustOla2Error("suppression_limit must be in [0,1]")
    if not initial_ri:
        raise RustOla2Error("initial_ri cannot be empty")

    binary = _ensure_rust_binary()
    with tempfile.TemporaryDirectory(prefix="skald_rust_ola2_") as td:
        input_path = os.path.join(td, "python_result.json")
        shape_path = os.path.join(td, "global_hist_shape.json")
        flat_path = os.path.join(td, "global_hist_flat.json")
        output_path = os.path.join(td, "rust_result.json")

        max_levels = [_compute_max_levels(int(qi.get_range())) for qi in quasi_identifiers]
        payload = {
            "initial_ri": [int(x) for x in initial_ri],
            "max_levels": max_levels,
            "k": int(k),
            "suppression_limit": float(suppression_limit),
            "total_records": int(total_records),
        }

        with open(input_path, "w") as f:
            json.dump(payload, f)
        with open(shape_path, "w") as f:
            json.dump([int(x) for x in global_hist.shape], f)
        with open(flat_path, "w") as f:
            json.dump([int(x) for x in global_hist.flatten()], f)

        timeout_sec = int(os.getenv("SKALD_RUST_OLA2_TIMEOUT", "900"))
        proc = subprocess.run(
            [binary, input_path, shape_path, flat_path, output_path],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RustOla2Error(
                f"Rust OLA2 failed (code {proc.returncode}): {stderr or stdout}"
            )

        try:
            with open(output_path, "r") as f:
                result = json.load(f)
        except Exception as e:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RustOla2Error(
                f"failed reading Rust OLA2 output: {e}; stdout='{stdout}', stderr='{stderr}'"
            ) from e

    if not isinstance(result, dict):
        raise RustOla2Error("invalid Rust OLA2 output format")

    for key in ["best_rf", "lowest_dm_star", "num_equivalence_classes", "elapsed_time"]:
        if key not in result:
            raise RustOla2Error(f"Rust OLA2 output missing key '{key}'")

    return result
