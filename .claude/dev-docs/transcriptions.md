# Dymos Transcription Architecture

## Overview

A transcription converts a continuous optimal control problem into a finite-dimensional NLP by
choosing how the ODE is discretized in time. All transcription classes inherit from
`TranscriptionBase` (`dymos/transcriptions/transcription_base.py`).

### Transcription Map (DYMOS_2=1)

When the environment variable `DYMOS_2=1` is set, `dm.GaussLobatto` maps to `GaussLobattoNew`
and `dm.Radau` maps to `RadauNew`. The "New" transcriptions use a single ODE at all nodes,
whereas the originals run separate ODEs at disc and col nodes.

| User API | DYMOS_2=0 class | DYMOS_2=1 class | ODE structure |
|----------|-----------------|-----------------|---------------|
| `dm.GaussLobatto` | `GaussLobatto` | `GaussLobattoNew` | single ODE at all nodes |
| `dm.Radau` | `RadauPseudospectral` | `RadauNew` | single ODE at all nodes |

### Setup vs Configure

OpenMDAO calls `setup()` then `configure()` on all systems. Dymos uses this split:

- **`setup_*(phase)`** methods: add subsystems (`phase.add_subsystem`), declare static structure.
  Shape/unit information is not yet available.
- **`configure_*(phase)`** methods: add connections, promotes, and design variables after
  introspection has determined shapes and units.

Call order within `Phase.setup()` / `Phase.configure()`:
```
setup:    setup_time → setup_states → setup_controls → setup_ode → setup_defects
          → setup_boundary_constraints → setup_path_constraints → setup_timeseries

configure: configure_time → configure_states → configure_controls → configure_ode
           → configure_defects → configure_boundary_constraints
           → configure_path_constraints → configure_timeseries_outputs
           → configure_solvers
```

---

## GaussLobattoNew

**File:** `dymos/transcriptions/pseudospectral/gauss_lobatto_new.py`
**Key component:** `dymos/transcriptions/pseudospectral/components/gauss_lobatto_iter_group.py`

### Grid

```python
self.grid_data = GaussLobattoGrid(num_segments=..., nodes_per_seg=order, ...)
```

- `nodes_per_seg = order` (not order+1)
- For `order=3`: 3 nodes/segment → 2 disc (indices 0,2) + 1 col (index 1) per segment
- Disc nodes sit at segment endpoints (LGL endpoints); col nodes are interior LGL points

### `_rhs_source`

```python
self._rhs_source = 'ode_iter_group.ode_interp_group.ode'
```

Used by introspection helpers to find ODE input/output metadata.

### Phase Subsystem Hierarchy

```
phase/
  time                       promotes_inputs=['*'], promotes_outputs=['*']
  control_comp               promotes_inputs=['*controls*']
                             promotes_outputs=['control_values:*', 'control_rates:*',
                                              'control_boundary_values:*',
                                              'control_boundary_rates:*']
  ode_iter_group             promotes=['*']      <-- GaussLobattoIterGroup
    ode_interp_group         promotes_inputs=['*'], promotes_outputs=['*']
                             nonlinear_solver: NonlinearBlockGS (Picard loop)
      states_all_init        promotes_inputs=['*'], promotes_outputs=['*']
      ode                    (no promotes; connected explicitly)
      lgl_interp_comp        promotes_inputs=['*'], promotes_outputs=['*']
    defects                  promotes_inputs=['*'], promotes_outputs=['*']
    states_resids_comp       (optional; only when solve_segments is used)
  boundary_vals              promotes_inputs=['initial_states:*', 'final_states:*']
  timeseries/{name}          ...
```

**Important:** Because `ode_iter_group` uses `promotes=['*']`, all inputs/outputs of
`GaussLobattoIterGroup` appear at the phase level. The promoted path to ODE outputs
from outside the phase (e.g., in `p.get_val(...)`) is:

```
traj.phase0.ode.{subcomp}.{var}      # promoted path for get_val
```
NOT `traj.phase0.ode_iter_group.ode_interp_group.ode.{subcomp}.{var}`.

### Key Promoted Variables (at phase level)

| Promoted name | Source component | Shape | Notes |
|---------------|-----------------|-------|-------|
| `states:{name}` | design var / IVC | `(n_input,) + shape` | state values at input nodes |
| `states_all:{name}` | `states_all_init` | `(n_all,) + shape` | states at ALL nodes |
| `states_col:{name}` | `lgl_interp_comp` | `(n_col,) + shape` | states at col nodes only |
| `staterate_disc:{name}` | `lgl_interp_comp` input | `(n_disc,) + shape` | ODE rates at disc nodes |
| `staterate_col:{name}` | `lgl_interp_comp` output | `(n_col,) + shape` | interpolated rates at col |
| `f_approx:{name}` | `defects` input | `(n_col,) + shape` | Hermite-approximated rates |
| `f_computed:{name}` | `defects` input | `(n_col,) + shape` | ODE-computed rates at col |
| `defects:{name}` | `defects` output | `(n_col,) + shape` | collocation residuals |
| `ode.{var}` | `ode` output | `(n_all,) + shape` | ODE outputs at all nodes |
| `dt_dstau` | `time` | `(n_all,)` | time scaling at all nodes |

### Key Connections by Method

#### `configure_time` (GaussLobattoNew)
```python
# t/t_phase → ode targets at all nodes
phase.connect('t', ['ode.{t}' for t in options['targets']],
              src_indices=grid_data.subset_node_indices['all'], flat_src_indices=True)

# scalar time targets (t_initial, t_duration, t_final)
phase.connect('{name}_val', 'ode.{t}', src_indices=None)   # shape (1,) target
phase.connect('{name}_val', 'ode.{t}',                      # broadcast target
              src_indices=np.zeros(n_all, dtype=int), flat_src_indices=True)
```

#### `configure_controls` (GaussLobattoNew)
```python
# control values/rates → ode targets (already at all nodes, no src_indices needed)
phase.connect('control_values:{name}', ['ode.{t}' for t in options['targets']])
phase.connect('control_rates:{name}_rate', ['ode.{t}' for t in options['rate_targets']])
phase.connect('control_rates:{name}_rate2', ['ode.{t}' for t in options['rate2_targets']])
```

#### `configure_ode` (GaussLobattoNew)
```python
# Delegates to GaussLobattoIterGroup.configure_io(phase)
phase._get_subsystem('ode_iter_group').configure_io(phase)
```

Inside `GaussLobattoIterGroup.configure_io(phase)`:
```python
# state targets: states_all:{name} → ode.{tgt}   (inside ode_interp_group)
ode_interp_group.connect(f'states_all:{name}', f'ode.{tgt}')

# ODE rate at disc nodes → lgl_interp_comp input  (inside ode_interp_group)
ode_interp_group.connect(f'ode.{rate_source}', f'staterate_disc:{name}',
                         src_indices=om.slicer[disc_idxs, ...])

# ODE rate at col nodes → defects f_computed  (self = GaussLobattoIterGroup)
self.connect(f'ode.{rate_source}', f'f_computed:{name}',
             src_indices=om.slicer[col_idxs, ...])

# state-as-rate at col nodes → defects f_computed  (self level)
self.connect(f'states_all:{rate_source}', f'f_computed:{name}',
             src_indices=om.slicer[col_idxs, ...])
```

#### `configure_defects` (GaussLobattoNew)
Handles non-ODE, non-state rate sources (time, control, parameter):
```python
# For 'time', 'control', 'parameter' rate sources:
phase.connect(rate_src_path, f'f_computed:{name}', src_indices=om.slicer[col_idxs, ...])
phase.connect(rate_src_path, f'staterate_disc:{name}', src_indices=om.slicer[disc_idxs, ...])

# f_approx is connected inside GaussLobattoIterGroup:
self.connect(f'staterate_col:{name}', f'f_approx:{name}')
```

### Constraint / Response Paths

**`_get_response_src(var, loc, phase)` returns:**

| var_type | loc | constraint_path | index logic |
|----------|-----|-----------------|-------------|
| `state` | any | `states_all:{var}` | see below |
| `ode` | any | `ode.{var}` | see below |
| `control` | `path` | `control_values:{var}` | all nodes |
| `control` | `initial`/`final` | `control_boundary_values:{var}` | 2-node array |
| `control_rate` | `path` | `control_rates:{var}` | all nodes |
| `control_rate` | `initial`/`final` | `control_boundary_rates:{var}` | 2-node array |
| `parameter` | any | `parameter_vals:{var}` | flat_idxs |

**`_get_constraint_kwargs` index logic:**

`num_nodes = grid_data.num_nodes` (= `n_all`, total nodes in phase)
`size = prod(shape)`

- **State initial:** `indices = flat_idxs`  (row 0)
- **State final:** `indices = (num_nodes - 1) * size + flat_idxs`  (last row)
- **ODE initial:** `indices = flat_idxs`
- **ODE final:** `indices = (num_nodes - 1) * size + flat_idxs`
- **Control boundary (2-node array):** initial = `flat_idxs`; final = `size + flat_idxs`
  (detected by `con_path.startswith(('control_boundary_values:', 'control_boundary_rates:'))`)

**Note:** `_has_boundary_ode=True` causes introspection to initially set
`con_path = 'boundary_vals.{var}'` for ODE outputs and
`con_path = 'initial_states:{var}'` / `'final_states:{var}'` for states.
`_get_constraint_kwargs` rewrites these to `'ode.{var}'` and `'states_all:{var}'`
respectively (see lines 520–527 in `gauss_lobatto_new.py`).

### `states_all_init` Component

`StatesAllInitComp` assembles the full-node state array from:
- `states:{name}` (input nodes, expanded to disc positions via `input_to_disc` map)
- `states_col:{name}` (col nodes, written at col positions)

This feeds back to `ode` via the NLBGS algebraic loop inside `ode_interp_group`.

---

## RadauNew

**File:** `dymos/transcriptions/pseudospectral/radau_new.py`
**Key component:** `dymos/transcriptions/pseudospectral/components/radau_iter_group.py`

### Grid

```python
self.grid_data = RadauGrid(num_segments=..., nodes_per_seg=order + 1, ...)
```

- `nodes_per_seg = order + 1`
- For `order=3`: 4 nodes/segment → 3 col (LGR interior) + 1 disc (segment end)
- Disc nodes are at segment *ends* only; col nodes are LGR interior points
- Example: 30 segments, order=3 → 120 total nodes, 90 col, 30 disc

### `_rhs_source`

```python
self._rhs_source = 'ode_iter_group.ode_all'
```

### Phase Subsystem Hierarchy

```
phase/
  time                       promotes=['*']
  control_comp               (same promotes as GaussLobattoNew)
  ode_iter_group             promotes=['*']      <-- RadauIterGroup
    ode_all                  (no promotes)
    defects                  promotes_inputs=['*'], promotes_outputs=['*']
    states_resids_comp       (optional; only when solve_segments is used)
  boundary_vals              promotes_inputs=['initial_states:*', 'final_states:*']
  timeseries/{name}          ...
```

Because `ode_iter_group` uses `promotes=['*']`, `ode_all` outputs are accessible
from the phase level as `ode_all.{var}` (NOT `ode_iter_group.ode_all.{var}`).

### Key Promoted Variables (at phase level)

| Promoted name | Source | Shape | Notes |
|---------------|--------|-------|-------|
| `states:{name}` | design var / IVC | `(n_input,) + shape` | state values at input nodes |
| `f_ode:{name}` | `defects` input | `(n_col,) + shape` | rate source at col nodes |
| `defects:{name}` | `defects` output | `(n_col,) + shape` | collocation residuals |
| `ode_all.{var}` | `ode_all` output | `(n_all,) + shape` | ODE outputs at all nodes |
| `dt_dstau` | `time` | `(n_all,)` | time scaling at all nodes |

### Key Connections by Method

#### `configure_time` (RadauNew)
```python
# t/t_phase → ode_all targets at all nodes
phase.connect('t', ['ode_all.{t}' for t in targets],
              src_indices=src_idxs, flat_src_indices=True)
phase.connect('{name}_val', 'ode_all.{t}', ...)
```

#### `configure_controls` (RadauNew)
```python
phase.connect('control_values:{name}', ['ode_all.{t}' for t in options['targets']])
# boundary controls go to boundary_vals subsystem (separate 2-node ODE)
phase.connect('control_boundary_values:{name}', ['boundary_vals.{t}' for t in options['targets']])
```

#### `configure_ode` (RadauNew)
```python
phase._get_subsystem('boundary_vals').configure_io(phase)
phase._get_subsystem('ode_iter_group').configure_io(phase)
```

Inside `RadauIterGroup.configure_io(phase)`:
```python
# state targets: states:{name}[state_input_to_disc] → ode_all.{tgt}
self.promotes('ode_all', [(tgt, f'states:{name}')],
              src_indices=om.slicer[state_src_idxs_input_to_all, ...])
self.set_input_defaults(f'states:{name}', val=1.0, units=units, src_shape=(nin,)+shape)

# dt_dstau sliced to col nodes for defects
self.promotes('defects', inputs=('dt_dstau',),
              src_indices=om.slicer[col_idxs, ...], src_shape=(nn,))

# The f_ode:{name} connection (ode_all rate source → defects) is NOT made here.
# It is made at the phase level by RadauNew.configure_defects().
```

#### `configure_defects` (RadauNew)
```python
# ODE-type rate sources → f_ode (at phase level, not group level)
# MUST be phase-level to correctly handle distributed ODEs.
phase.connect('ode_all.{rate_source}', f'f_ode:{name}',
              src_indices=om.slicer[col_idxs, ...])

# parameter rate sources get zero-broadcast indices
phase.connect('parameter_vals:{var}', f'f_ode:{name}',
              src_indices=om.slicer[np.zeros_like(col_idxs), ...])

# state-as-rate: connect states:{name} (not 'states:' prefix check skips this)
# rate_src_path.startswith('states:') skips it (no connect made for state-as-rate here)
```

**CRITICAL:** The `f_ode:{name}` connection **must** be made by `configure_defects` at the
phase level using the promoted path `ode_all.{rate_source}`. Do **not** add a group-level
`self.connect('ode_all.{rate_source}', 'f_ode:{name}', ...)` inside `RadauIterGroup`,
because that creates a duplicate and fails for distributed ODEs (see openmdao-patterns.md).

### Constraint / Response Paths

RadauNew uses the same `_get_response_src` / `_get_constraint_kwargs` logic as GaussLobattoNew
with these key differences:

- State arrays: `initial_states:{name}` / `final_states:{name}` paths come from `boundary_vals`
- ODE outputs: `boundary_vals.{var}` is the initial source for boundary constraints
  (2-node array from the separate boundary ODE); `_get_constraint_kwargs` rewrites
  `boundary_vals.{var}` → `ode_all.{var}` using the full node path

See `radau_new.py:_get_response_src` and `_get_constraint_kwargs` for exact index logic,
which mirrors GaussLobattoNew but targeting the `n_all`-size `ode_all` array.

---

## Linkage Sources

For trajectory linkage (connecting final state of one phase to initial of next),
`_get_linkage_source_ode()` returns the path to query for ODE output values:

- **GaussLobattoNew:** `'ode_iter_group.ode_interp_group'` (promoted) or
  `'ode_iter_group.ode_interp_group.ode'` (absolute)
- **RadauNew:** uses `boundary_vals` ODE for initial/final values

---

## Parameter Connections

Parameters are connected to the ODE through `get_parameter_connections()`.
For both GaussLobattoNew and RadauNew, parameters are broadcast to all nodes:

```python
# GaussLobattoNew: target is 'ode.{tgt}'
connection_info.append(('ode.{tgt}', (src_idxs,)))

# RadauNew: target is 'ode_all.{tgt}'
connection_info.append(('ode_all.{tgt}', (src_idxs,)))
```

---

## Known Pitfalls

1. **Duplicate `f_ode` connection in RadauNew:** `configure_defects` makes the phase-level
   connection. Do not also add it inside `RadauIterGroup.configure_io`. Adding a second
   group-level connect produces "already connected" errors.

2. **Distributed ODE and `src_indices`:** Group-internal `self.connect()` validates
   `src_indices` against the local (per-rank) size of a distributed source, not the global
   size. Use `phase.connect()` at the phase level for any connection involving distributed
   ODE outputs with global-index `src_indices`. See openmdao-patterns.md for details.

3. **`src_shape` is not a parameter of `Group.connect()`:** It is a parameter of
   `Group.promotes()` (helps auto_ivc determine the shape to provide). Do not pass `src_shape`
   to `connect()`.

4. **Promoted path for `get_val`:** Because `ode_iter_group` uses `promotes=['*']`, the
   promoted path to ODE outputs in `get_val` is `traj.phase0.ode.{subcomp}.{var}` for
   GaussLobattoNew and `traj.phase0.ode_all.{var}` for RadauNew. These are NOT the
   absolute paths.
