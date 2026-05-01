# Project

dymos is a pytho package for trajectory optimization within OpenMDAO.

# Development Guidelines

- Use the "dev-local" environment in the pyproject.toml file in the root of this project.

- Do not invoke git commit, but suggest a commit message.

# Code Style

- Use single quotes, except to define docstrings. When nesting quotes, use double quotes for the outer string.

- Documentation is in the numpy doc style.

- Do not use emojis.

# Validation

- Generated code should pass `ruff check`. Rules for ruff are established in pyproject.toml.

- Use `pixi run testflo <path>` to invoke tests in the relevant scope. Only run `pixi run testflo` if the user asks for the full test suite.

# Developer Reference Docs

Internal implementation knowledge is stored in `.claude/dev-docs/`. Consult these
before modifying transcription, ODE connection, or grid-data code, and update them
when making non-obvious discoveries.

- `.claude/dev-docs/transcriptions.md` — phase subsystem hierarchy, setup/configure
  call sequence, key connections and promotions for GaussLobattoNew and RadauNew.
  Also covers constraint/response path logic and known pitfalls.
- `.claude/dev-docs/grid-data.md` — node subset names, index maps, node layout
  by transcription type, and common usage patterns.
- `.claude/dev-docs/openmdao-patterns.md` — promoted paths, `connect` vs `promotes`,
  `src_indices` for distributed components, `src_shape`, duplicate connection errors,
  DYMOS_2 path differences, MPI hang diagnosis.
