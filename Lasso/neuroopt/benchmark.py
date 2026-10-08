"""
Compare backends on accuracy, convergence, operation counts and estimated energy.

Energy is estimated, not measured. Per-operation costs come from published
numbers (see ENERGY_PJ) and can be overridden.

    CPU:   ~200 pJ per flop   (desktop i7: ~95 W TDP / ~450 GFLOPS peak FP32)
    GPU:   ~20 pJ per flop    (NVIDIA A100: 400 W / 19.5 TFLOPS FP32)
    Loihi: 23.6 pJ per synaptic event, 81 pJ per neuron update
           (Davies et al., "Loihi: a neuromorphic manycore processor", IEEE Micro 2018)

CPU/GPU estimates assume the dense simulation; Loihi estimates count only the
spikes that actually happen, which is the point of event-driven hardware.
"""

from __future__ import annotations

import time

import numpy as np

from .api import solve

ENERGY_PJ = {
    "cpu_per_flop": 200.0,
    "gpu_per_flop": 20.0,
    "loihi_per_synaptic_event": 23.6,
    "loihi_per_neuron_update": 81.0,
}

DEFAULT_BACKENDS = ("reference", "fista", "analog", "spiking")


def _iterations_to(history, key, target, index_key):
    """First recorded iteration/time where history[key] drops below target."""
    for index, value in zip(history.get(index_key, []), history.get(key, [])):
        if value <= target:
            return index
    return np.nan


def benchmark(problem, backends=DEFAULT_BACKENDS, options=None, repeats=3, kkt_target=1e-4,
              energy_pj=None):
    """
    Returns a list of dicts, one per backend. Wall time is the median of `repeats` runs.
    """
    options = options or {}
    energy_pj = {**ENERGY_PJ, **(energy_pj or {})}
    problem = problem if hasattr(problem, "gram") else problem.canonicalize()
    if problem.constraint_matrix is not None:
        backends = [b for b in backends if b not in ("ista", "fista")]

    reference = solve(problem, backend="reference")
    rows = []

    for backend in backends:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = solve(problem, backend=backend, **options.get(backend, {}))
            times.append(time.perf_counter() - start)

        stats = result.stats
        flops = stats.get("flops", np.nan)
        # compare supports at a common threshold so a backend's own noise floor doesn't decide
        common = max(result.active_threshold, reference.active_threshold)
        same_support = np.array_equal(np.abs(result.x_raw) > common, np.abs(reference.x_raw) > common)
        row = {
            "backend": result.solver,
            # accuracy
            "objective": result.objective,
            "objective_gap": result.objective - reference.objective,
            "error_vs_reference": float(np.linalg.norm(result.x - reference.x)),
            "support_matches": bool(same_support),
            "n_active": result.n_active,
            "kkt_residual": result.kkt_residual,
            "active_threshold": result.active_threshold,
            # convergence
            "converged": result.converged,
            "iterations": result.iterations,
            "sim_time": stats.get("sim_time", np.nan),
            f"iterations_to_kkt_{kkt_target:g}": _iterations_to(
                result.history, "kkt_residual", kkt_target,
                "iteration" if "iteration" in result.history else "time"),
            "wall_time_ms": 1e3 * float(np.median(times)),
            # work
            "flops": flops,
            "total_spikes": stats.get("total_spikes", np.nan),
            "synaptic_events": stats.get("synaptic_events", np.nan),
            "neuron_updates": stats.get("neuron_updates", np.nan),
            # energy estimates (nJ)
            "energy_cpu_nJ": flops * energy_pj["cpu_per_flop"] * 1e-3,
            "energy_gpu_nJ": flops * energy_pj["gpu_per_flop"] * 1e-3,
            "energy_loihi_nJ": np.nan,
        }
        if "synaptic_events" in stats:
            row["energy_loihi_nJ"] = 1e-3 * (
                stats["synaptic_events"] * energy_pj["loihi_per_synaptic_event"]
                + stats["neuron_updates"] * energy_pj["loihi_per_neuron_update"])
        rows.append(row)
    return rows


def print_table(rows):
    columns = [
        ("backend", "{:<22}"), ("error_vs_reference", "{:>10.2e}"), ("support_matches", "{!s:>8}"),
        ("kkt_residual", "{:>10.2e}"), ("converged", "{!s:>6}"), ("iterations", "{:>8d}"),
        ("wall_time_ms", "{:>9.1f}"), ("flops", "{:>10.3g}"), ("total_spikes", "{:>10.3g}"),
        ("energy_cpu_nJ", "{:>9.3g}"), ("energy_gpu_nJ", "{:>9.3g}"), ("energy_loihi_nJ", "{:>9.3g}"),
    ]
    headers = ["backend", "|x-ref|", "support", "KKT", "conv", "iters", "ms",
               "flops", "spikes", "CPU nJ", "GPU nJ", "Loihi nJ"]
    widths = [22, 10, 8, 10, 6, 8, 9, 10, 10, 9, 9, 9]
    print("".join(f"{h:>{w}}" if i else f"{h:<{w}}" for i, (h, w) in enumerate(zip(headers, widths))))
    print("-" * sum(widths))
    for row in rows:
        cells = []
        for key, fmt in columns:
            value = row[key]
            if isinstance(value, float) and np.isnan(value):
                cells.append(f"{'-':>{widths[len(cells)]}}")
            else:
                cells.append(fmt.format(value))
        print("".join(cells))
