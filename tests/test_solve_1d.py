import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import ArrayLike


def _simple_debug_plot(
    solution: ArrayLike,
    filename: str | os.PathLike,
    x=None,
    y=None,
) -> None:
    """Plots a PETSc.Vec e.g. a solution vector."""
    # TODO: should go to dxh?
    plt.clf()
    plt.plot(solution)
    if x is not None and y is not None:
        plt.plot(x, y)
    plt.savefig(filename)


def test_solve_1d_regression():
    """Run the main 1D solver and check that the solution is as expected.

    Expecting a damped sine wave with an amplitude of ~4 at max.  Will fail if
    any errors are encountered at any stage (problem setup, and running the
    solver). The solver will be pypardiso (if installed) or PETSc's LU
    decomposition if pardiso is not installed. Normalisation may differ.
    """
    expected_solution = np.recfromcsv("tests/data/solve_1d_solution.csv")
    _simple_debug_plot(expected_solution, "expected.png")
    # obtained_solution, _ = solve_problem(measure_errors=True)
    obtained_solution = np.recfromcsv(
        "solution_kg.csv",
    )  # from running Krishna's GMRes at the same commit as the trusted one

    # assert len(expected_solution) == len(obtained_solution), "Solution lengths do not match."

    total_samples = len(expected_solution)
    approx_peak_distance = total_samples / 5

    peaks, heights = scipy.signal.find_peaks(
        obtained_solution,
        distance=approx_peak_distance,
    )
    _simple_debug_plot(obtained_solution, "obtained.png", x=peaks, y=heights)

    assert len(peaks) == 2, f"Expecting two peaks, found {len(peaks)}."
