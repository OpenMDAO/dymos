#!/usr/bin/env bash
set -euo pipefail

docker run --rm --shm-size=2g \
  --env OPENMDAO_CHECK_ALL_PARTIALS=True \
  --env PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe \
  --env OMPI_MCA_rmaps_base_oversubscribe=1 \
  --env OMPI_MCA_btl=^openib \
  --env TMPDIR=/tmp \
  --entrypoint /usr/local/bin/pixi "$1" \
  run --locked -e dev --manifest-path /home/omdao/dymos/pyproject.toml \
  bash -e -o pipefail -c '
    python -m pip check
    mpirun -n 3 python -c "from mpi4py import MPI; assert MPI.COMM_WORLD.size == 3"
    mpirun -n 3 python -c "from mpi4py import MPI; from petsc4py import PETSc; v = PETSc.Vec().createMPI(3, comm=MPI.COMM_WORLD); v.set(1); assert v.sum() == 3"
    testflo dymos -n 4 --pre_announce --show_skipped --durations 20
  '
