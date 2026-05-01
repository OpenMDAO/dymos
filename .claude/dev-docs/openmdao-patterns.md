# OpenMDAO Patterns in Dymos

This document captures OpenMDAO behaviors that are non-obvious and have caused bugs or
confusion during dymos development. Read this before working on connection/promotion code.

---

## Promoted Paths and `get_val`

### `get_val` uses promoted paths, not absolute paths

`p.get_val('path.to.var')` resolves using the **promoted** name hierarchy, not the absolute
path through subsystems. When a group uses `promotes=['*']`, its children's inputs and
outputs are promoted to the group's level.

**Example — GaussLobattoNew:**

`ode_iter_group` is added with `promotes=['*']`. Inside it, `ode_interp_group` promotes
everything, and inside that, `ode` is a subsystem. The `ode` component's output `xdot` is
promoted through both inner groups to the phase level. So from outside the phase:

```python
# Correct promoted path:
p.get_val('traj.phase0.ode.xdot_comp.xdot')

# WRONG (absolute path, not promoted):
p.get_val('traj.phase0.ode_iter_group.ode_interp_group.ode.xdot_comp.xdot')
```

**Example — RadauNew:**

`ode_iter_group` promotes everything. Inside it, `ode_all` is a subsystem (not promoted).
`ode_all`'s output `x0dot` is promoted to the phase level:

```python
# Correct:
p.get_val('traj.phase0.ode_all.x0dot')

# WRONG:
p.get_val('traj.phase0.ode_iter_group.ode_all.x0dot')
```

### `traj.phases.phase0` vs `traj.phase0`

The absolute path to a phase subsystem under a trajectory is `traj.phases.phase0`
(because the `Trajectory` stores phases in a group named `phases`).

However, `p.get_val()` uses the **promoted path**, and the trajectory promotes phases
to the top level, so `p.get_val('traj.phase0.var')` is correct, not
`p.get_val('traj.phases.phase0.var')`.

---

## `Group.connect()` vs `Group.promotes()`

### `connect(src_name, tgt_name, src_indices=..., flat_src_indices=...)`

- Connects a source output to one or more target inputs **within the same group or below**.
- `src_name` is resolved relative to the group calling `connect()`.
- `src_indices` selects a subset of the source for the connection.
- **No `src_shape` parameter.** Passing `src_shape` to `connect()` raises `TypeError`.

```python
# CORRECT:
self.connect('ode.xdot', 'defects.f_computed:x',
             src_indices=om.slicer[col_idxs, ...])

# WRONG -- src_shape is not a connect() parameter:
self.connect('ode.xdot', 'defects.f_computed:x',
             src_indices=om.slicer[col_idxs, ...],
             src_shape=(nn,))   # TypeError!
```

### `promotes(subsys_name, inputs=..., outputs=..., src_indices=..., src_shape=...)`

- Used **after** `add_subsystem` in `configure()` to promote specific I/O with optional
  index slicing.
- `src_shape` tells OpenMDAO the shape of the external source, so `auto_ivc` can create
  an appropriately-sized IndepVarComp output.
- Used in dymos to slice `dt_dstau` (full-node array) at col-only nodes for the defect
  component:

```python
self.promotes('defects', inputs=('dt_dstau',),
              src_indices=om.slicer[col_idxs, ...],
              src_shape=(nn,))   # tells auto_ivc the source has shape (nn,)
```

---

## `src_indices` Behavior

### Normal (non-distributed) sources

`src_indices` is validated against the **global size** of the source variable. This is
the standard case and works as expected at any group level.

### Distributed sources

A component output marked `distributed=True` is split across MPI ranks. Each rank owns
a contiguous slice of `io_size` elements, where `io_size = nn // n_ranks`.

**Critical:** When you call `self.connect()` **inside a group**, OpenMDAO validates
`src_indices` against the **local (per-rank) size**, not the global size. If
`max(src_indices) >= local_size`, you get:

```
index N is out of bounds for source dimension of size M
```

even though `N < global_size`.

**Fix:** Move the connection to the **phase level** (or any level above the group
containing the distributed source). At the phase level, OpenMDAO uses the global size
for validation.

```python
# WRONG -- inside RadauIterGroup (self.connect):
self.connect('ode_all.x0dot', 'f_ode:x0',
             src_indices=om.slicer[col_idxs, ...])
# → fails if ode_all is distributed and max(col_idxs) >= local_size

# CORRECT -- at the phase level (phase.connect):
phase.connect('ode_all.x0dot', 'f_ode:x0',
              src_indices=om.slicer[col_idxs, ...])
```

This is why `RadauNew.configure_defects()` makes the `ode_all → f_ode` connection
at the phase level, not inside `RadauIterGroup`.

### `flat_src_indices=True`

When `src_indices` is a 1-D integer array and `flat_src_indices=True`, the indices
address the **flattened** source array. Used in dymos for time targets:

```python
phase.connect('t', ['ode.t_target'],
              src_indices=all_idxs, flat_src_indices=True)
```

Without `flat_src_indices=True`, a 1-D index array is interpreted as selecting rows
from a multi-dimensional source.

---

## `om.slicer[...]`

`om.slicer` creates a slice object accepted by `src_indices`. Common patterns:

```python
# Select specific global row indices (for a 2-D+ source):
om.slicer[col_idxs, ...]     # rows = col_idxs, all columns

# Select a scalar index:
om.slicer[0, ...]            # first row only

# Broadcast: repeat the same row for N output rows:
om.slicer[np.zeros(n, dtype=int), ...]   # repeats row 0 N times (for parameters)
```

---

## `auto_ivc` and Unconnected Inputs

If an input has no explicit source and no `set_input_defaults`, OpenMDAO creates an
`auto_ivc` IndepVarComp to drive it. The shape of this auto-IVC is inferred from the
input's declared shape.

When `promotes()` is used with `src_indices` to slice a larger external array down to a
smaller input shape, OpenMDAO needs `src_shape` to know the shape of the *external*
source (so it can create an auto_ivc of the right size if needed):

```python
self.promotes('defects', inputs=('dt_dstau',),
              src_indices=om.slicer[col_idxs, ...],
              src_shape=(nn,))
```

Without `src_shape`, OpenMDAO would size the auto_ivc based on the input shape (col
nodes only), which is wrong when `dt_dstau` is a full-node array.

---

## Setup vs Configure

OpenMDAO performs two phases of initialization:

1. **`setup()`** — builds the system tree: adds subsystems, declares I/O.
   Shape/unit information is not available. Use `add_subsystem`, `add_input`,
   `add_output`, `add_design_var`, `declare_partials`.

2. **`configure()`** — called after the first setup pass resolves shapes/units.
   Use `connect`, `promotes`, `set_input_defaults`. Also safe to add additional
   I/O here (deferred I/O).

In dymos, `setup_*` methods use `add_subsystem`; `configure_*` methods use
`connect`, `promotes`, and deferred I/O like `configure_io()`.

---

## Duplicate Connection Error

If you attempt to `connect` a target that already has a source (from a previous
`connect` call or promotion), OpenMDAO raises:

```
Input 'X' is already connected to 'Y'.
```

This often surfaces as a hidden duplicate when a connection is made in two places
(e.g., once inside a sub-group's `configure_io` and again in the transcription's
`configure_defects`). Always check for existing connections when adding new ones.

**Known case in dymos:** `f_ode:{name}` in RadauNew is connected by
`RadauNew.configure_defects()`. Do not also connect it inside
`RadauIterGroup.configure_io()`.

---

## DYMOS_2 Path Differences

When `DYMOS_2=1`:
- `dm.GaussLobatto` → `GaussLobattoNew`; ODE runs at all `n_all` nodes
- `dm.Radau` → `RadauNew`; ODE runs at all `n_all` nodes

| Variable | DYMOS_2=0 path | DYMOS_2=1 path |
|----------|----------------|----------------|
| ODE output (GL) | `traj.phase0.rhs_col.{comp}.{var}` (n=n_col) OR `traj.phase0.rhs_disc.{comp}.{var}` (n=n_disc) | `traj.phase0.ode.{comp}.{var}` (n=n_all) |
| ODE output (Radau) | `traj.phase0.rhs_all.{var}` | `traj.phase0.ode_all.{var}` |

For tests that must work with both `DYMOS_2=0` and `DYMOS_2=1`, check the env var:

```python
import os
if os.environ.get('DYMOS_2') == '1':
    path = f'traj.phase0.ode.{comp}.{var}'
    expected_size = n_all   # e.g. 30 for 3 segments × order=10
else:
    path = f'traj.phase0.rhs_col.{comp}.{var}'
    expected_size = n_col
```

---

## MPI / Distributed Component Hang

When one MPI rank raises an exception during `setup()`, the other ranks continue
to wait at the MPI barrier, causing the process to hang. The symptom looks like a hang
but the root cause is a setup error on one rank. Fix the setup error first; the hang
will resolve itself.

The vanderpol distributed ODE (`VanderpolODE` with `distrib=True`) is the canonical
example: it uses `evenly_distrib_idxs(comm.size, num_nodes)` so each rank owns
`num_nodes // n_ranks` elements. With `num_nodes=120` and 2 ranks, each rank owns 60
nodes. Global col indices go up to 118, which exceeds 60 → group-level connection fails
→ hang.
