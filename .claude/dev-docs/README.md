# Dymos Developer Reference Docs

These documents capture internal implementation knowledge accumulated while developing dymos.
Consult the relevant doc **before** editing transcription, ODE connection, or grid-data code.
Update the relevant doc **after** making a discovery or fixing a non-obvious bug.

## Index

| File | Contents |
|------|----------|
| [transcriptions.md](transcriptions.md) | Phase subsystem hierarchy, setup/configure call sequence, key connections and promotions for GaussLobattoNew and RadauNew |
| [grid-data.md](grid-data.md) | Node subset names, index maps, node counts by transcription type |
| [openmdao-patterns.md](openmdao-patterns.md) | OpenMDAO idioms used in dymos: src_indices, promotes, distributed components, get_val paths |

## Quick Reference: Which doc to read

- Editing `gauss_lobatto_new.py` or `gauss_lobatto_iter_group.py` → **transcriptions.md** (GaussLobattoNew section)
- Editing `radau_new.py` or `radau_iter_group.py` → **transcriptions.md** (RadauNew section)
- Working with `grid_data.py` or node indexing → **grid-data.md**
- Confused about a promote/connect behavior, `get_val` path, or distributed ODE → **openmdao-patterns.md**
- Adding a new constraint or objective → **transcriptions.md** (`_get_response_src` / `_get_constraint_kwargs` sections)
