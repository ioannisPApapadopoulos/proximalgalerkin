"""CSV helpers for proximal Galerkin experiment results.

The main entry points are:

    append_result("results.csv", result)
    write_results("results.csv", results)
    solve_and_append(problem, "results.csv")

where ``result`` is the dictionary returned by ``ProximalGalerkin.pg_solve()``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


DEFAULT_COLUMNS = (
    "problem",
    "preconditioner",
    "n",
    "refinements",
    "cells_per_side",
    "degree",
    "epsilon",
    "model_parameter",
    "alpha0",
    "alpha_max",
    "pg_rtol",
    "smoothing_its",
    "proximal_steps",
    "newton_steps",
    "outer_fgmres_iterations",
    "avg_outer_fgmres_per_newton",
    "max_outer_fgmres_per_newton",
    "top_left_block_inverses",
    "inner_cg_iterations",
    "avg_inner_cg_iterations_per_top_left_inverse",
    "max_inner_cg_iterations_per_top_left_inverse",
    "final_cauchy_error",
    "elapsed_seconds",
    "converged_pg",
)


@contextlib.contextmanager
def tee_process_output(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()

    def copy_output():
        with os.fdopen(read_fd, "rb", closefd=True) as pipe, log_path.open("wb") as log:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                os.write(stdout_fd, chunk)
                log.write(chunk)
                log.flush()

    thread = threading.Thread(target=copy_output, daemon=True)
    thread.start()
    try:
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        thread.join()
        os.close(stdout_fd)
        os.close(stderr_fd)


def parse_inner_cg_stats(output: str) -> dict:
    reason_re = re.compile(
        r"Linear .*fieldsplit_0_ solve "
        r"(?:converged|diverged) due to [A-Z_]+ iterations (\d+)"
    )
    counts = [int(match.group(1)) for match in reason_re.finditer(output)]
    total = sum(counts)
    solves = len(counts)
    return {
        "top_left_block_inverses": solves,
        "inner_cg_iterations": total,
        "avg_inner_cg_iterations_per_top_left_inverse": total / solves if solves else "",
        "max_inner_cg_iterations_per_top_left_inverse": max(counts) if counts else "",
    }


def parse_outer_fgmres_stats(output: str) -> dict:
    reason_re = re.compile(
        r"Linear (?!.*fieldsplit).* solve "
        r"(?:converged|diverged) due to [A-Z_]+ iterations (\d+)"
    )
    counts = [int(match.group(1)) for match in reason_re.finditer(output)]
    return {
        "max_outer_fgmres_per_newton": max(counts) if counts else "",
    }


def parse_result_json(output: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    return None


def add_cg_stats_from_log(result: MutableMapping, log_path) -> MutableMapping:
    output = Path(log_path).read_text()
    outer_stats = parse_outer_fgmres_stats(output)
    if outer_stats["max_outer_fgmres_per_newton"] != "":
        result.update(outer_stats)
    result.update(parse_inner_cg_stats(output))
    return result


def append_log_result(csv_path, log_path, columns: Sequence[str] | None = None) -> MutableMapping:
    output = Path(log_path).read_text()
    result = parse_result_json(output)
    if result is None:
        result = {"converged_pg": False}
    outer_stats = parse_outer_fgmres_stats(output)
    if outer_stats["max_outer_fgmres_per_newton"] != "":
        result.update(outer_stats)
    result.update(parse_inner_cg_stats(output))
    append_result(csv_path, result, columns=columns)
    return result



def problem_metadata(problem) -> dict:
    """Collect common scalar attributes from a ProximalGalerkin subclass."""
    names = (
        "preconditioner",
        "n",
        "refinements",
        "cells_per_side",
        "degree",
        "epsilon",
        "alpha0",
        "alpha_max",
        "pg_rtol",
        "smoothing_its",
        "model_parameter"
    )
    metadata = {"problem": problem.__class__.__name__}
    for name in names:
        if hasattr(problem, name):
            metadata[name] = getattr(problem, name)

    if "n" in metadata and "refinements" in metadata and "cells_per_side" not in metadata:
        metadata["cells_per_side"] = metadata["n"] * 2 ** metadata["refinements"]
    
    if "model_parameter" not in metadata:
        metadata["model_parameter"] = None
    return metadata

def normalise_row(result: Mapping, extra: Mapping | None = None) -> dict:
    """Return a CSV-safe row from a result dictionary plus optional metadata."""
    row = dict(result)
    if extra:
        row.update(extra)
    for key, value in list(row.items()):
        if value is None:
            row[key] = ""
    return row


def columns_for(rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> list[str]:
    """Choose stable CSV columns."""
    return list(columns or DEFAULT_COLUMNS)


def write_results(path, results: Iterable[Mapping], columns: Sequence[str] | None = None) -> None:
    """Write all result rows to ``path``, replacing any existing file."""
    rows = [normalise_row(result) for result in results]
    fieldnames = columns_for(rows, columns)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_result(path, result: Mapping, columns: Sequence[str] | None = None) -> None:
    """Append one result row to ``path``, creating a header if needed."""
    row = normalise_row(result)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size:
        with output.open(newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader)
        write_header = False
    else:
        fieldnames = columns_for([row], columns)
        write_header = True

    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def solve_and_append(problem, path, extra: Mapping | None = None, columns: Sequence[str] | None = None, log_path=None) -> MutableMapping:
    """Run ``problem.pg_solve()``, save output to a log, append CSV, and return the result."""
    result = problem_metadata(problem)
    if log_path is None:
        log_path = Path("logs") / (Path(path).stem + ".log")

    with tee_process_output(log_path):
        result.update(problem.pg_solve())

    add_cg_stats_from_log(result, log_path)
    if extra:
        result.update(extra)
    append_result(path, result, columns=columns)
    return result
