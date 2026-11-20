#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash .github/scripts/test-docker-image.sh [OPTIONS] [IMAGE]

Build the native Dymos image and run its tests using Docker or Apple container.
Run this script from any directory. Docker is preferred when both services work.

Options:
  --runtime auto|docker|container  Select the runtime (default: auto).
  --no-build                      Test an existing image without rebuilding it.
  -h, --help                      Show this help.

IMAGE defaults to dymos-test:local.
Apple container build/run memory defaults to 8g; set DYMOS_CONTAINER_MEMORY to override.
USAGE
}

runtime=auto
build=true
image=dymos-test:local
image_set=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      if [[ $# -lt 2 ]]; then
        echo 'Missing value for --runtime' >&2
        exit 2
      fi
      runtime=$2
      shift 2
      ;;
    --no-build) build=false; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ "$image_set" == true ]]; then
        echo 'Only one image name can be supplied' >&2
        exit 2
      fi
      image=$1
      image_set=true
      shift
      ;;
  esac
done

runtime_ready() {
  command -v "$1" >/dev/null 2>&1 || return 1
  case "$1" in
    docker) docker info >/dev/null 2>&1 ;;
    container) container system status >/dev/null 2>&1 ;;
  esac
}

case "$runtime" in
  auto)
    if runtime_ready docker; then
      runtime=docker
    elif runtime_ready container; then
      runtime=container
    else
      echo 'No working container runtime found. Start Docker or run: container system start' >&2
      exit 1
    fi
    ;;
  docker|container)
    if ! runtime_ready "$runtime"; then
      echo "The $runtime runtime is missing or its service is not running" >&2
      exit 1
    fi
    ;;
  *) echo "Unsupported runtime: $runtime" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
echo "Using $runtime to test $image"

# Apple container runs each container in a VM with explicit resource limits.
build_options=(build)
run_options=(--rm --shm-size=2g)
if [[ "$runtime" == container ]]; then
  build_options+=(--cpus 4 --memory "${DYMOS_CONTAINER_MEMORY:-8g}")
  run_options+=(--cpus 4 --memory "${DYMOS_CONTAINER_MEMORY:-8g}")
fi

if [[ "$build" == true ]]; then
  case "$(uname -m)" in
    arm64|aarch64) dockerfile=Dockerfile.arm; platform=linux/arm64 ;;
    x86_64|amd64) dockerfile=Dockerfile; platform=linux/amd64 ;;
    *) echo 'Unsupported host architecture' >&2; exit 1 ;;
  esac
  "$runtime" "${build_options[@]}" --platform "$platform" \
    --file "$repo_root/$dockerfile" --tag "$image" "$repo_root"
fi

"$runtime" run "${run_options[@]}" \
  --env OPENMDAO_CHECK_ALL_PARTIALS=True \
  --env PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe \
  --env OMPI_MCA_rmaps_base_oversubscribe=1 \
  --env OMPI_MCA_btl=^openib \
  --env TMPDIR=/tmp \
  --entrypoint /usr/local/bin/pixi "$image" \
  run --locked -e dev --manifest-path /home/omdao/dymos/pyproject.toml \
  bash -e -o pipefail -c '
    python -m pip check
    mpirun -n 3 python -c "from mpi4py import MPI; assert MPI.COMM_WORLD.size == 3"
    mpirun -n 3 python -c "from mpi4py import MPI; from petsc4py import PETSc; v = PETSc.Vec().createMPI(3, comm=MPI.COMM_WORLD); v.set(1); assert v.sum() == 3"
    testflo dymos -n 4 --pre_announce --show_skipped --durations 20
  '
