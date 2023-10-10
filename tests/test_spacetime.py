import numpy as np
from dolfinx.mesh import CellType, GhostMode, create_box

from dxss.solve_1d import omega_ind_convex, sample_sol, dt_sample_sol
from dxss.space_time import SpaceTime

from mpi4py import MPI

def test_spacetime_slab_matrix():
    test_mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [2, 2, 2],
        CellType.hexahedron,
        ghost_mode=GhostMode.shared_facet,
    )
    spacetime = SpaceTime(
        q=2,
        qstar=0,
        k=2,
        kstar=1,
        N=2,
        T=1.0,
        t=0,
        msh=test_mesh,
        omega_ind=omega_ind_convex,
        stabs={"data": 1e4,
               "dual": 1.0,
               "primal": 1e-3,
               "primal-jump": 1.0,
               },
        sol=sample_sol,
        dt_sol=dt_sample_sol,
    )
    spacetime.setup_spacetime_finite_elements()
    spacetime.prepare_precondition_gmres()
    slabmatrix = spacetime.get_slab_matrix()

    # regression test against the slab matrix we've set up
    assert slabmatrix.getSize() == (804, 804)
    assert slabmatrix.getValues()
