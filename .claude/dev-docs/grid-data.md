# Dymos Grid Data Reference

## GridData Class

**File:** `dymos/transcriptions/grid_data.py`

`GridData` stores all node information for a phase. It is constructed once in
`TranscriptionBase.init_grid()` and stored as `self.grid_data` on the transcription object.
It is passed to every component that needs node index information.

---

## Key Attributes

### Scalars

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_segments` | int | Number of segments in the phase |
| `num_nodes` | int | Total number of nodes across all segments |
| `compressed` | bool | Whether compressed transcription is used |

### Arrays

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `node_ptau` | `(num_nodes,)` | Node positions in phase tau (global [-1, 1]) |
| `node_dptau_dstau` | `(num_nodes,)` | d(ptau)/d(stau) at each node (time scaling factor) |
| `segment_indices` | `(num_segments, 2)` | Start and end (exclusive) node index for each segment |
| `transcription_order` | `(num_segments,)` | Transcription order for each segment |

### Dicts

| Attribute | Keys | Description |
|-----------|------|-------------|
| `subset_node_indices` | see below | Global node indices belonging to each subset |
| `subset_num_nodes` | see below | Total count of nodes in each subset |
| `subset_num_nodes_per_segment` | see below | Per-segment count of nodes in each subset |
| `input_maps` | see below | Maps between input and discretization node sets |

---

## Node Subsets

`gd.subset_node_indices['subset_name']` returns an `ndarray` of global node indices.
`gd.subset_num_nodes['subset_name']` returns the total count.

| Subset name | Description |
|-------------|-------------|
| `'all'` | All nodes (every node in the phase) |
| `'state_disc'` | State discretization nodes (used as ODE evaluation points for state continuity) |
| `'state_input'` | State input nodes (the design variable inputs; subset of `state_disc`) |
| `'col'` | Collocation nodes (interior nodes where defect residuals are evaluated) |
| `'control_disc'` | Control discretization nodes |
| `'control_input'` | Control input nodes (design variable inputs for controls) |
| `'segment_ends'` | First and last node of each segment |

### Subset Relationships

- `state_input` ⊆ `state_disc` ⊆ `all`
- `col` ⊆ `all`
- `state_disc` ∪ `col` = `all`  (for both GL and Radau)
- With **compressed** transcription: `state_input` omits the repeated first node of each
  segment after the first (the endpoint is shared with the previous segment's last node).
- With **uncompressed** transcription: `state_input` == `state_disc`.

---

## Input Maps

`gd.input_maps['state_input_to_disc']` is an index array `I` such that
`state_disc_values = state_input_values[I]`.

In other words: for each discretization node `i`, `I[i]` is the index into the
`state_input` array that provides its value. This is used in `StatesAllInitComp` and
`RadauIterGroup.configure_io` to expand input-node state arrays to disc-node arrays.

```python
# Usage pattern:
state_src_idxs = gd.input_maps['state_input_to_disc']
# When promoting state targets:
self.promotes('ode_all', [(tgt, f'states:{name}')],
              src_indices=om.slicer[state_src_idxs, ...])
```

Similarly, `input_maps['dynamic_control_input_to_disc']` maps control input nodes to
control disc nodes.

---

## Node Layout by Transcription

### GaussLobatto / GaussLobattoNew

```python
GaussLobattoGrid(num_segments=ns, nodes_per_seg=order, ...)
```

- `nodes_per_seg = order` (must be odd for LGL)
- For `nodes_per_seg=3` (order=3):
  ```
  Segment layout (local indices 0,1,2):
    0 = state_disc, segment_end (LGL endpoint)
    1 = col (LGL interior)
    2 = state_disc, segment_end (LGL endpoint)
  ```
- State disc nodes are at **even** local indices (0, 2, 4, ...)
- Col nodes are at **odd** local indices (1, 3, 5, ...)

**Example:** `GaussLobattoGrid(num_segments=2, nodes_per_seg=3, compressed=True)`

```
Global node layout:
  seg 0: 0(disc), 1(col), 2(disc)
  seg 1: 3(col),  4(disc)     <- compressed: node 0 of seg1 = node 2 of seg0

subset_node_indices:
  'all':        [0, 1, 2, 3, 4]
  'state_disc': [0, 2, 4]
  'state_input': [0, 2, 4]    (compressed: seg0 includes both endpoints)
  'col':        [1, 3]
num_nodes = 5
```

### Radau / RadauNew

```python
RadauGrid(num_segments=ns, nodes_per_seg=order+1, ...)
```

- `nodes_per_seg = order + 1`
- For `order=3` (nodes_per_seg=4):
  ```
  Segment layout (local indices 0,1,2,3):
    0 = col (LGR point)
    1 = col (LGR point)
    2 = col (LGR point)
    3 = state_disc, segment_end
  ```
- State disc nodes are at the **last** local index of each segment
- Col nodes occupy the remaining (first `order`) positions

**Example:** `RadauGrid(num_segments=2, nodes_per_seg=4, compressed=True)`

```
Global node layout:
  seg 0: 0(col), 1(col), 2(col), 3(disc)
  seg 1: 4(col), 5(col), 6(col), 7(disc)

subset_node_indices:
  'all':         [0, 1, 2, 3, 4, 5, 6, 7]
  'state_disc':  [3, 7]
  'state_input': [3, 7]   (or [7] for compressed? check actual code)
  'col':         [0, 1, 2, 4, 5, 6]
num_nodes = 8
```

**Node counts for common configurations:**

| order | nodes_per_seg | ns=10 total | ns=10 col | ns=10 disc |
|-------|--------------|-------------|-----------|------------|
| 3 | 4 | 40 | 30 | 10 |
| 4 | 5 | 50 | 40 | 10 |
| 30 segs, order=3 | 4 | 120 | 90 | 30 |

---

## dt_dstau

`dt_dstau` is computed by `TimeComp` and has shape `(num_nodes,)`. It equals
`(t_duration / 2) * node_dptau_dstau` at each node.

- Defect components use `dt_dstau[col_idxs]` (col nodes only).
- `node_dptau_dstau` converts from segment-normalized tau to phase-normalized tau.

---

## Grid Methods

### `get_state_bnd_idxs()`

Returns indices of the first and last state-disc node, used to extract initial/final
state values from the full discretization node array.

### `subset_segment_indices[subset_name]`

Shape `(num_segments, 2)`. For segment `i`, `[i, 0]` and `[i, 1]` are the start and
end (exclusive) indices into `subset_node_indices[subset_name]`. Useful for
per-segment slicing.

---

## Usage Patterns

### Slicing a full-node array at col nodes
```python
col_idxs = gd.subset_node_indices['col']
om.slicer[col_idxs, ...]   # use as src_indices in connect/promotes
```

### Slicing a full-node array at disc nodes
```python
disc_idxs = gd.subset_node_indices['state_disc']
om.slicer[disc_idxs, ...]
```

### Expanding from input nodes to disc nodes
```python
input_to_disc = gd.input_maps['state_input_to_disc']
om.slicer[input_to_disc, ...]  # src_indices for promotes
```

### Getting col-node count
```python
n_col = gd.subset_num_nodes['col']
n_disc = gd.subset_num_nodes['state_disc']
n_input = gd.subset_num_nodes['state_input']
n_all = gd.num_nodes   # equivalently: gd.subset_num_nodes['all']
```
