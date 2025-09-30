********************************
# Release Notes for Dymos 1.15.0

September 30, 2025

This release features a refactor of the Radau transcription that should be transparent to users, but cleans up some of the internal machinery. Among other things, this will fix issues with checking totals in the model due to the hybrid implicit/explicit behavior of some of the old components. It also moves towards unifying the transcriptions so that all of them can separately accept "states:{name}", "initial_states:{name}", and "final_states:{name}". The plan is to apply this behavior to the Gauss-Lobatto transcription in the next release. To enable the use of this refactor Radau transcription set the environment variable `DYMOS_2=1`

## Backwards Incompatible API Changes & Deprecations

None

## Enhancements

- Completed refactor of the Radau transcription. [#1196](https://github.com/OpenMDAO/dymos/pull/1196)
- `add_calc_expr` can now promote variable names. [#1204](https://github.com/OpenMDAO/dymos/pull/1204)

## Bug Fixes

- Fixed a bug where picard shooting wasn't allowing initial or final state values to be design variables. [#1210](https://github.com/OpenMDAO/dymos/pull/1210)

## Miscellaneous

- Removed unncessary internal unit conversion from Timeseries and Interleave components. [#1198](https://github.com/OpenMDAO/dymos/pull/1198)
- Updated GitHub workflows, use NumPy 2.2 as the baseline [#1199](https://github.com/OpenMDAO/dymos/pull/1199)
- Hull problem Rework [#1201](https://github.com/OpenMDAO/dymos/pull/1201)
- Updated ruff configuration and fixed issues found by ruff. [#1206](https://github.com/OpenMDAO/dymos/pull/1206)
- Added fix for change in conda env export. [#1208](https://github.com/OpenMDAO/dymos/pull/1208)
- Removed redundant final_setup calls. [#1211](https://github.com/OpenMDAO/dymos/pull/1211)

********************************
# Release Notes for Dymos 1.14.0

June 16, 2025

This release addresses issues found in 1.13.1.

This is largely a bug fix release that catches a few issues with the recently-added PicardShooting transcription. One new experimental feature is the addition of `dt_dstau_targets` in the Phase time options. This lets ODE components have access to the dt_dstau ratio of each node, which in turn lets ODE components do things like computing rates of variables if they are known at all nodes across the phase.

## Backwards Incompatible API Changes & Deprecations

- When dymos automatically determines the rate units for state and control, variables with units of "unitless" will now have rate units of "1/s" (or more generally "1/{time_units}") rather than "unitless/s". This may require existing models to change some units [#1182](https://github.com/OpenMDAO/dymos/pull/1182)

## Enhancements

- Added experimental capability to connect dt_dstau_targets to targets in the ODE. [#1189](https://github.com/OpenMDAO/dymos/pull/1189)

## Bug Fixes

- Fix for failing test due to calling ravel() on a scalar. [#1176](https://github.com/OpenMDAO/dymos/pull/1176)
- Fixed PicardShooting so that it now works with vector-valued states. [#1177](https://github.com/OpenMDAO/dymos/pull/1177)
- Fixed bug that prevented distributed ODE outputs from being timeseries outputs. [#1185](https://github.com/OpenMDAO/dymos/pull/1185)
- Trajectory.configure_parameters will not add parameters to phases if they already exist [#1191](https://github.com/OpenMDAO/dymos/pull/1191)
- Fix a bug in the state units for the multiple shooting update comp. [#1189](https://github.com/OpenMDAO/dymos/pull/1189)

## Miscellaneous

- Updated GitHub workflows to test PyPI package after a release is published [#1173](https://github.com/OpenMDAO/dymos/pull/1173)
- Refactored controls for more commonality between polynomial and "full controls. Also now output separate control values at the boundary nodes for use with Birkhoff and the upcoming Radau transcription refactor. [#1174](https://github.com/OpenMDAO/dymos/pull/1174)
- A bit of cleanup to introspection. [#1178](https://github.com/OpenMDAO/dymos/pull/1178)
- Removed version checks that involve OM versions before minimum supported version. [#1183](https://github.com/OpenMDAO/dymos/pull/1183)
- Added MPI-safe logic to check if an option is left unspecified [#1184](https://github.com/OpenMDAO/dymos/pull/1184)

********************************
# Release Notes for Dymos 1.13.1

Apr 10, 2025

This release addresses issues found in 1.13.0.

The default `val` given for controls were not correctly applied.
Control continuity was left out of the PicardShooting transcription.
These have been fixed, in addition to some maintainence improvements for the devs.

## Backwards Incompatible API Changes & Deprecations

None

## Enhancements

None

## Bug Fixes

- Added support for using `model_options` when simulate=True. [#1164](https://github.com/OpenMDAO/dymos/pull/1164)
- Fixed an issue where default values in add_control or set_control_options were no longer being respected. [#1165](https://github.com/OpenMDAO/dymos/pull/1165)
- Added control continuity to PicardShooting [#1167](https://github.com/OpenMDAO/dymos/pull/1167).

## Miscellaneous

- Quieted plots and IPOPT output in a few tests when testflo is running. [#1161](https://github.com/OpenMDAO/dymos/pull/1161)
- Publish docs to openmdao.org on release event [#1162](https://github.com/OpenMDAO/dymos/pull/1162)
- Added a PyPi release workflow [#1168](https://github.com/OpenMDAO/dymos/pull/1168)

********************************
# Release Notes for Dymos 1.13.0

Apr 04, 2025

Version 1.13.0 of dymos adds a few significant features.

Users can now declare a calculation expression using `phase.add_calc_expr`.
Previously this was done solely in timeseries and constraints.
To generalize the capability, these expressions are now evaluated via an ExecComp that is executed as part of the ODE. This ExecComp is evaluated immediately after the ODE in the model hierarchy.
Any output created with `add_calc_expr` will be seen as an ODE output by dymos.

We've deprecated `set_duration_balance` in favor of `phase.add_boundary_balance`, which allows any parameter-like phase input to be an implicit output subject to some residual at the start or end of the phase. Note that this capability only makes sense in phases where the dynamics are satisfied via a nonlinear solver rather than an optimizer (PicardShooting, ExplicitShooting, or Pseudospectral phases using `solve_segments`). This capability allows the user to satisfy boundary value problems without an optimizer. Following versions of dymos will use this capability in more examples.

This release also adds the new `PicardShooting` transcription, which uses [Picard iteration](https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem) to implicitly propagate states. An internal solver (`PicardShooting.options['ode_nonlinear_solver']` which is NonlinearBlockGS by default) converges the dynamics in each segment, while an outer solver (`PicardShooting.options['ms_nonlinear_solver']` which is NonlinearBlockGS by default) enforces state continuity at segment bounds if multiple segments are used.
`PicardShooting` is a faster alternative to ExplicitShooting.

## Backwards Incompatible API Changes & Deprecations

- Added `phase.add_boundary_balance` and deprecated `phase.set_duration_balance` [#1148](https://github.com/OpenMDAO/dymos/pull/1148)

## Enhancements

- Added interpolation of results to arbitrary nodes, removed use of `scipy.interpolate.interp1d` from `phase.interp`. [#1129](https://github.com/OpenMDAO/dymos/pull/1129)
- Moved Expressions from being calculated in the timeseries to being calculated along side the ODE. [#1147](https://github.com/OpenMDAO/dymos/pull/1147)
- Added an image definition and a workflow to publish the image [#1143](https://github.com/OpenMDAO/dymos/pull/1143)
- Added an ARM image [#1149](https://github.com/OpenMDAO/dymos/pull/1149)
- Added `phase.add_boundary_balance` and deprecated `phase.set_duration_balance`. This PR also adds the new `PicardShooting` transcription. [#1148](https://github.com/OpenMDAO/dymos/pull/1148)

## Bug Fixes

- Fixed a NumPy 2.x testing issue and added a NumPy 2.x build to the test workflow [#1114](https://github.com/OpenMDAO/dymos/pull/1114)
- Fixed bug in timeseries plots under MPI. [#1116](https://github.com/OpenMDAO/dymos/pull/1116)
- Fixed a bug that caused a shape mismatch error when using explicit shooting on a problem with a matrix valued parameter. [#1127](https://github.com/OpenMDAO/dymos/pull/1127)
- Allow double linkage between phases [#1121](https://github.com/OpenMDAO/dymos/pull/1121)
- Trajectory timeseries reports now work with multiple trajectories in a single problem. [#1123](https://github.com/OpenMDAO/dymos/pull/1123)
- Replaced interp1d with InterpND in tests and docs [#1125](https://github.com/OpenMDAO/dymos/pull/1125)
- Fixed a bug where control rates of polynomial controls were not correct in simulation. [#1133](https://github.com/OpenMDAO/dymos/pull/1133)
- Trajectory results report plots now plot vector variables, respect the x_name argument, and avoid noisy plots when values are nearly constant. [#1135](https://github.com/OpenMDAO/dymos/pull/1135)
- Fixed an issue with implicit duration [#1140](https://github.com/OpenMDAO/dymos/pull/1140)
- Fixed a small bug in the output report when simulate=False [#1150](https://github.com/OpenMDAO/dymos/pull/1150)

## Miscellaneous

- Changed 'latest' job in test and docs workflows to use Python 3.12 [#1122](https://github.com/OpenMDAO/dymos/pull/1122)
- Some cleanup of the Birkhoff transcription and tests. [#1142](https://github.com/OpenMDAO/dymos/pull/1142)
- Removed ControlGroup since ControlInterpComp handles all types of controls [#1146](https://github.com/OpenMDAO/dymos/pull/1146)
- Removed stdout constraint summary from Trajectory configure process. [#1145](https://github.com/OpenMDAO/dymos/pull/1145)
- Added a test of the readme.md file to make sure it doesnt get out-of-date [#1151](https://github.com/OpenMDAO/dymos/pull/1151)
- Changed most examples to use set_state_val instead of set_val. [#1152](https://github.com/OpenMDAO/dymos/pull/1152)
- Get baseline and latest docs building again. [#1153](https://github.com/OpenMDAO/dymos/pull/1153)
- Cleanup of some dead code involving polynomial controls and timeseries expressions [#1154](https://github.com/OpenMDAO/dymos/pull/1154)

********************************
# Release Notes for Dymos 1.12.0

Oct 02, 2024

Dymos 1.12.0 primarily supports the updated paths to OpenMDAO output files in the OpenMDAO 3.35.0 release.

The Birkhoff transcription can involve extraordinarily long segments with hundreds of nodes. The lagrange polynomial interpolation algorith used with shorter segments is inefficient in this case and is replaced (for simulation) by
the use of cubic splines from scipy.interpolate. This provides fast interpolation that's accurate enough for simulation,
when derivatives are not needed.

The SolveIVP transcription, which has long been deprecated in favor of the more general and derivative-capable ExplicitShooting transcription, is removed.

## Backwards Incompatible API Changes & Deprecations
- Removed `SolveIVP` and removed dead code from TranscriptionBase [#1104](https://github.com/OpenMDAO/dymos/pull/1104)

## Enhancements
- bokeh timeseries plots now comply with OpenMDAO's placement of recorder files. [#1106](https://github.com/OpenMDAO/dymos/pull/1106)
- Faster interpolation for explicit shooting [#1109](https://github.com/OpenMDAO/dymos/pull/1109)

## Miscellaneous
- Added readme to pyproject.toml [#1094](https://github.com/OpenMDAO/dymos/pull/1094)
- Reduced the memory allocation in the TimeseriesOutputComp in common use cases. [#1096](https://github.com/OpenMDAO/dymos/pull/1096)
- Timeseries report tooltip hover change. [#1098](https://github.com/OpenMDAO/dymos/pull/1098)
- Removed need for copy_build_artifacts in the doc build process. [#1100](https://github.com/OpenMDAO/dymos/pull/1100)
- Removed use of newshape keyword argument from numpy reshape calls. [#1102](https://github.com/OpenMDAO/dymos/pull/1102)
- Added a pre-commit configuration to enable automatic linting with ruff [#1105](https://github.com/OpenMDAO/dymos/pull/1105)
- Added the Gong challenge problem benchmark for Birkhoff transcription. [#1108](https://github.com/OpenMDAO/dymos/pull/1108)
- Updated GitHub Workflows [#1111](https://github.com/OpenMDAO/dymos/pull/1111)

*******************************
# Release Notes for Dymos 1.11.0

August 08, 2024

Dymos 1.11.0 includes the initial implementation of the Birkhoff transcription and a number of other significant improvements. These improvements have focused on the ability to the user to switch transcriptions or control types with a minimal change to their scripts.

### Unification of controls and polynomial controls

The separate APIs for controls and polynomial controls are no more, in favor of the `set_control_options` or `add_control` methods that can be used to either add a standard "fully collocated" control or a "polynomial" control. Methods `set_polynomial_control_options` and `add_polynomial_control` are deprecated.

### set_time_val, set_state_val, set_control_val, and set_parameter_val

The Birkhoff transcription is significantly different from the Gauss-Lobatto and Radau transcriptions in that it requires the estimated state rate, the initial state value, and the final state value to be additional design variables. This information is redundant from the user's perspective, but provides the Birkhoff method with some greater degree of robustness.

To remove the need to provide these extra values when using the Birkhoff transcription, the `set_state_val` method on phase will automatically populate them based upon the interpolated values of the state in the phase. To make the dymos API consistent, similar methods are added for `set_time_val`, `set_control_val`,  and `set_parameter_val`.

### integ_var_options and set_integ_var_options

For cases when the integration variable is not time, the notion of using `phase.time_options` and `phase.set_time_options` can be confusing.
For this reason, `phase.integ_var_options` and `phase.set_integ_var_options` have been added as alias to `phase.time_options` and `phase.set_time_options`, respectively.

## Backwards Incompatible API Changes & Deprecations
- `set_polynomial_control_options` and `phase.poylnomial_control_options` are deprecated in favor of the common use of `phase.set_control_options` and `phase.control_options` where an order for the control is specified if it is a polynomial control.

## Enhancements
- Changed phase default NonlinearSolver to have a maxiter of 100. [#1034](https://github.com/OpenMDAO/dymos/pull/1034)
- Updated GitHub workflow to use only the conda-forge channel for dependencies [#1036](https://github.com/OpenMDAO/dymos/pull/1036)
- Some Birkhoff constraints are now treated as linear. [#1043](https://github.com/OpenMDAO/dymos/pull/1043)
- Use sparse partials for the Birkhoff state residuals when OpenMDAO bug is fixed. [#1055](https://github.com/OpenMDAO/dymos/pull/1055)
- Use Barycentric Lagrange Interpolation for controls in Simulation [#1056](https://github.com/OpenMDAO/dymos/pull/1056)
- Changed oldest support python to 3.9 [#1067](https://github.com/OpenMDAO/dymos/pull/1067)
- API Change for setting values [#1072](https://github.com/OpenMDAO/dymos/pull/1072)
- Unify controls and polynomial controls [#1078](https://github.com/OpenMDAO/dymos/pull/1078)
- Added optional `integ_var_options` for `time_options` when the integration variable is not time. [#1079](https://github.com/OpenMDAO/dymos/pull/1079)
- Updated phase.load_case so that it uses the state_xxx_vals API.  Specify parent when subproblems are used. [#1087](https://github.com/OpenMDAO/dymos/pull/1087)
- Added Birkhoff Docs [#1088](https://github.com/OpenMDAO/dymos/pull/1088)


## Bug Fixes
- Fixed a bug that was causing trajectory result reports not to be displayed in the documentation. [#1032](https://github.com/OpenMDAO/dymos/pull/1032)
- Fixed an issue where matplotlib was not required to install dymos without specifications but would fail to import. [#1035](https://github.com/OpenMDAO/dymos/pull/1035)
- Fixed an issue with timeseries reports and the Birkhoff transcription. [#1040](https://github.com/OpenMDAO/dymos/pull/1040)
- Fixed issue with running trajectory timeseries report under MPI [#1046](https://github.com/OpenMDAO/dymos/pull/1046)
- Fix for test failure in test_error_estimation.py [#1047](https://github.com/OpenMDAO/dymos/pull/1047)
- Fixed matplotlib calls that were broken by a change in the matplotlib API [#1070](https://github.com/OpenMDAO/dymos/pull/1070)
- Fixed a logic issue in Birkhoff timeseries outputs. [#1071](https://github.com/OpenMDAO/dymos/pull/1071)
- Fixed a bug where timeserires report use of fig.circle instead of fig.scatter stopped working in Bokeh 3.4. [#1073](https://github.com/OpenMDAO/dymos/pull/1073)
- Replace deprecated numpy function `in1d()` with `isin()` [#1081](https://github.com/OpenMDAO/dymos/pull/1081)
- Stopped coloring plots popping up during tests [#1082](https://github.com/OpenMDAO/dymos/pull/1082)
- Fixed bug that caused aviary CI failure. [#1083](https://github.com/OpenMDAO/dymos/pull/1083)


## Miscellaneous
- Replaced `setup.py` with `pyproject.toml` using `hatchling` backend [#1052](https://github.com/OpenMDAO/dymos/pull/1052)
- Minor spelling fixes [#1053](https://github.com/OpenMDAO/dymos/pull/1053)
- Added inline directive to notebooks that use matplotlib [#1054](https://github.com/OpenMDAO/dymos/pull/1054)
- Make docs link correctly formatted [#1059](https://github.com/OpenMDAO/dymos/pull/1059)
- Added skip to handle tests that need matplotlib and/or pydocstyle when not installed [#1062](https://github.com/OpenMDAO/dymos/pull/1062)
- Modified `latest` workflow build to use petsc 3.21.0 [#1064](https://github.com/OpenMDAO/dymos/pull/1064)


*******************************
# Release Notes for Dymos 1.10.0

December 08, 2023

Dymos 1.10.0 includes a few performance improvements, more user-configurability of automatic solver placement, and a still-experimental implementation of the Birkhoff transcription.

Thanks to Shugo Kaneko for pointing out some issues with using dymos under MPI.
Timeseries jacobians were inefficient and used excessive memory when operating on a single system under MPI.
Also, the user now has more control over how solvers are added when necessary. Previously, the presence of a DirectSolver over a distributed system was causing an error.

Trajectory now has a `parallel_phases` option. When True (the default), the top level `Trajectory.phases` container will be an OpenMDAO ParallelGroup. Setting it to `False` makes `Trajectory.phases` a standard serial `OpenMDAO.api.Group` object.

Trajectory and Phase now both have an `auto_solvers` option. It is enabled by default and will result in appropriate nonlinear and linear solvers being placed when it is `True`.
Trajectory.phases needs a nonlinear solver when running phases in parallel under MPI AND the phases are directly connected.
Phases need a solver when there is implicit behavior due to a duration balance or the use of `solve_segments` is pseudospectral phases.

## Backwards Incompatible API Changes & Deprecations
- None

## Enhancements
- Reduce memory usage for timeseries jac computation [#1001](https://github.com/OpenMDAO/dymos/pull/1001)
- Added Birkhoff Transcription (this feature is still somewhat experimental) [#1008](https://github.com/OpenMDAO/dymos/pull/1008) [#1010](https://github.com/OpenMDAO/dymos/pull/1010) [#1021](https://github.com/OpenMDAO/dymos/pull/1021)

## Bug Fixes
- Fixed python 3.12 compatibility issues. [#1004](https://github.com/OpenMDAO/dymos/pull/1004)
- Fixed load_case logic to correctly handle negative-duration phases [#1007](https://github.com/OpenMDAO/dymos/pull/1007)
- Boundary constraints in Birkhoff transcription changed to pull from boundary ODE [#1018](https://github.com/OpenMDAO/dymos/pull/1018)
- Fixed an issue where applying DirectSolver to StateIndependentsComp was breaking when used with other linear solvers under MPI. [#1020](https://github.com/OpenMDAO/dymos/pull/1020)
- Fixed Vanderpol example under updated OpenMDAO MPI operation. [#1023](https://github.com/OpenMDAO/dymos/pull/1023)

## Miscellaneous
- Added a job without MPI to the test workflow [#998](https://github.com/OpenMDAO/dymos/pull/998)
- Removed redundant implementations of timeseries output components. [#1005](https://github.com/OpenMDAO/dymos/pull/1005)
- Remove IVCs from control groups [#1013](https://github.com/OpenMDAO/dymos/pull/1013)


*******************************
# Release Notes for Dymos 1.9.1

September 14, 2023

Dymos 1.9.1 fixes a few bugs introduced in 1.9.0.
For problems in which phases are directly connected, the total derivatives
could be wrong in some cases due to the lack of a linear solver covering
the initial value defect. Also, control rates were showing up in timeseries
outputs by default. Now, as expected, they must be requested explicitly
as a timeseries output.

## Backwards Incompatible API Changes & Deprecations
- None

## Enhancements
- None

## Bug Fixes
- Added linear solver for cases where states solved via optimizer have input_initial=True. [#991](https://github.com/OpenMDAO/dymos/pull/991)
- Fixed an issue where control rate values are showing up in timeseries outputs by default. [#987](https://github.com/OpenMDAO/dymos/pull/987)

## Miscellaneous
- Minor tweaks to the github issue templates. [#985](https://github.com/OpenMDAO/dymos/pull/985) [#993](https://github.com/OpenMDAO/dymos/pull/993)


*******************************
# Release Notes for Dymos 1.9.0

September 07, 2023

Dymos 1.9.0 changes the default timeseries behavior so that state rates,
control rates, and t_phase are not included by default. These values tended
to clutter the timeseries plots on more complex problems.

A new `add_duration_balance` method is added to phases that allows the final time
of a phase to be found with a solver without the need for an optimizer. This is useful
for explicit shooting phases which are effectively just simulating out the dynamics
until some target condition is achieved. An example of this behavior is available [in the docs](https://openmdao.github.io/dymos/examples/cannonball_implicit_duration/cannonball_implicit_duration.html?highlight=duration).

Trajectory parameters are improved. Parameters can now have a mix of static and dynamic targets, which was not possible before.
Also, trajectories will now use variable data obtained by introspection in the phases for
setting properties of trajectory-level parameters.

The behavior of some scipy.sparse routines was changed in 1.11 and caused incorrect derivatives.
This has been addressed.

Dymos had previously been calling `allgather` excessively under MPI which could
cause performance issues during setup.

Finally, Trajectory.simulate() should now work correctly under MPI so that results can
be checked with explicit integration when working under MPI.

## Backwards Incompatible API Changes & Deprecations
- Remove state rates, control rates, and t_phase from the timeseries by default, and remove type prefix. [#935](https://github.com/OpenMDAO/dymos/pull/935)

## Enhancements
- Remove state rates, control rates, and t_phase from the timeseries by default, and remove type prefix. [#935](https://github.com/OpenMDAO/dymos/pull/935)
- Implicit duration [#939](https://github.com/OpenMDAO/dymos/pull/939)
- Updated dymos to utilize new OpenMDAO load_case capability. [#954](https://github.com/OpenMDAO/dymos/pull/954)
- Updated parameter introspection and traj parameters [#964](https://github.com/OpenMDAO/dymos/pull/964)
- Allow parameter shape to be given as `None`, rather than just `_unspecified` [#970](https://github.com/OpenMDAO/dymos/pull/970)

## Bug Fixes
- Fixed an error message that gave outdated instructions solve segments [#930](https://github.com/OpenMDAO/dymos/pull/930)
- Fixed a Trajectory.simulate method specification bug [#931](https://github.com/OpenMDAO/dymos/pull/931)
- Fixed issue where optimal trajectory parameters would also attempt to make corresponding phase parameter a design variable. [#938](https://github.com/OpenMDAO/dymos/pull/938)
- Fixed issue where "connected" could not be used when linking all variables of a phase [#942](https://github.com/OpenMDAO/dymos/pull/942)
- Fixed issues with scipy sparse storage in scipy 1.11 [#943](https://github.com/OpenMDAO/dymos/pull/943)
- Fixed an issue where grid refinement tests were still expecting some things to be in the timeseries. [#946](https://github.com/OpenMDAO/dymos/pull/946)
- Fixed openmdao warning messages. [#950](https://github.com/OpenMDAO/dymos/pull/950)
- Fixed timeseries plots under MPI so that all phases show up. [#959](https://github.com/OpenMDAO/dymos/pull/959)
- Fixed an MPI slowdown introduced by parameter introspection at the Trajectory level. [#977](https://github.com/OpenMDAO/dymos/pull/977)
- Fixed some issues that were causing simulation to fail under MPI. [#978](https://github.com/OpenMDAO/dymos/pull/978)
- Fixed an issue where dymos calling excessive allgathers during setup. [#980](https://github.com/OpenMDAO/dymos/pull/980)
- Fixed an issue where a DirectSolver was automatically added to a phase if any state has input_initial=True. [#982](https://github.com/OpenMDAO/dymos/pull/982)

## Miscellaneous
- Changed all uses of the deprecated 'value' keyword to 'val' [#926](https://github.com/OpenMDAO/dymos/pull/926)
- Bump to next dev cycle version [#927](https://github.com/OpenMDAO/dymos/pull/927)
- Updated the GitHub workflow to test with '[all]' extra dependencies [#929](https://github.com/OpenMDAO/dymos/pull/929)
- Fixed out-of-order systems in some tests [#941](https://github.com/OpenMDAO/dymos/pull/941)
- Refactored time_extents away in favor of housing extents in param_comp [#952](https://github.com/OpenMDAO/dymos/pull/952)
- Set defect_scaler values to 1.0 in finite_burn_orbit_raise problem [#953](https://github.com/OpenMDAO/dymos/pull/953)
- Fixed a couple PEP8 failures [#961](https://github.com/OpenMDAO/dymos/pull/961)
- Added various phase options dictionaries to the API documentation. [#956](https://github.com/OpenMDAO/dymos/pull/956)
- Updated GitHub workflow to upload documentation to web site [#965](https://github.com/OpenMDAO/dymos/pull/965)
- Minor doc fix [#967](https://github.com/OpenMDAO/dymos/pull/967)
- Slight changes to scaling in the water rocket example. [#969](https://github.com/OpenMDAO/dymos/pull/969)
- Updated baseline test to use Numpy version 1.22.4 [#975](https://github.com/OpenMDAO/dymos/pull/975)



*******************************
# Release Notes for Dymos 1.8.0

April 18, 2023

Dymos 1.8.0 removes some deprecated OpenMDAO features that are removed as of version 3.26.0.
It also adds a new trajectory results report that is generated by `dymos.run_problem(..., make_plots=True)`.
This single HTML file contains summaries of the parameters in each phase as well as plots of the timeseries values
across phases in each trajectory.
Generating this file requires `bokeh` rather than matplotlib - a message in the HTML file will inform the user if bokeh
was unavailable.
ExplicitShooting is now consistent with `solve_segments` in the way initial state values are specified.
This release also provides better logic when connecting variables of different units via phase linkages.

## Backwards Incompatible API Changes & Deprecations
- Remove deprecated src_indices argument in add_parameter() [#920](https://github.com/OpenMDAO/dymos/pull/920)
- Initial states in ExplicitShooting now provided with `phase_path.initial_states:state_name` [#915](https://github.com/OpenMDAO/dymos/pull/915)

## Enhancements
- trajectory results report now automatically generated by run_problem [#918](https://github.com/OpenMDAO/dymos/pull/918)

## Bug Fixes
- Added better logic to detect different units across linkage constraints. [#913](https://github.com/OpenMDAO/dymos/pull/913)

## Miscellaneous
- Documentation fixes [#906](https://github.com/OpenMDAO/dymos/pull/906)
- Cleaned up GitHub workflows [#907](https://github.com/OpenMDAO/dymos/pull/907)
- Minor changes focused on plotting utility [#908](https://github.com/OpenMDAO/dymos/pull/908)
- Documentation fixes [#909](https://github.com/OpenMDAO/dymos/pull/909)
- Replaced deprecated 'value' keyword with 'val' [#910](https://github.com/OpenMDAO/dymos/pull/910)
- Fixed missing component in Re-entry docs [#914](https://github.com/OpenMDAO/dymos/pull/914)

*******************************
# Release Notes for Dymos 1.7.0

February 09, 2023

Version 1.7.0 contains some significant enhancements to its capabilities.

Major improvements include:
- Users may now specify boundary constraints, path constraints, and the objective within a phase as expressions. These expressions must contain an equals sign and must be complex-step-safe. This allows the user to impose constraints or objectives on quantities that may not be computed by the given equations of motion model.
- The user can now rename the time variable. Integration with respect to another variable (such as range for an aircraft) has always been possible, but dymos still always called the variable "time". Now, it will be more clear.
- The ExplicitShooting transcription has been reworked to include a continuous adjoint for derivatives and uses the stock scipy.integrate.solve_ivp integrators to perform the actual integraiton. Note that this method may result in inaccurate derivatives when the time step varies widely thoughout an integration. (We're working on that part).
- Imposing rate2 continuity on controls would previously generate confusing errors with some optimizers like SLSQP. If the polynomial order of the control representation was quadratic, then the second derivative would always be zero. Continuity was thus guaranteed, but SLSQP would fail because it lacked the ability to affect that continuity despite it always being satisfied. Now Dymos will only respect control rate2 continuity across segment junctions if it can affect continuity at the segment junction.
- The `run_problem` method now includes a sanity check on time bounds. If it detects that the linkage constraints between two phases in time cannot be satisfied due to initial time and duration bounds on all previous phases, a warning will be raised.

## Backwards Incompatible API Changes & Deprecations

- Deprecated Trajectory.add_linkage_constraint arguments sign_a and sign_b [#888](https://github.com/OpenMDAO/dymos/pull/888)
- Deprecated SolveIVP transcription. The `simulate` method now uses the ExplicitShooting transcription without derivatives. [#898](https://github.com/OpenMDAO/dymos/pull/898)

## Enhancements

- Added ability to control continuity scaling for states and controls. [#866](https://github.com/OpenMDAO/dymos/pull/866)
- Parameter flow added to linkage report diagram [#873](https://github.com/OpenMDAO/dymos/pull/873)
- Added ability to allow the user to change the name of the integration variable. [#874](https://github.com/OpenMDAO/dymos/pull/874)
- Added ability to add expressions as constraints [#875](https://github.com/OpenMDAO/dymos/pull/875)
- Added the derivative of speed of sound wrt altitude as an output [#879](https://github.com/OpenMDAO/dymos/pull/879)
- Phase objectives may now be given as expressions. [#880](https://github.com/OpenMDAO/dymos/pull/880)
- Interim continuous-adjoint shooting implementation [#885](https://github.com/OpenMDAO/dymos/pull/885)
- Replaced existing ExplicitShooting instance with the continuous-adjoint Implementation [#891](https://github.com/OpenMDAO/dymos/pull/891)
- Selectively apply rate2 continuity to indices where it can be impacted. [#896](https://github.com/OpenMDAO/dymos/pull/896)
- Raise an exception if the initial time of a phase is unreachable [#899](https://github.com/OpenMDAO/dymos/pull/899)

## Bug Fixes

- Fixed load_case when a phase does not exist in the case. [#863](https://github.com/OpenMDAO/dymos/pull/863)
- Fixed url in _config.yml so that documentation examples can open at the Google colab. [#868](https://github.com/OpenMDAO/dymos/pull/868)
- Fixed dymos.load_case so that it does not load states or controls into AnalyticPhases [#871](https://github.com/OpenMDAO/dymos/pull/871)
- Fixed an unnecessary error when linking phases with different units via connection. [#872](https://github.com/OpenMDAO/dymos/pull/872)
- Handle parameter names containing colons in Dymos linkage report [#900](https://github.com/OpenMDAO/dymos/pull/900)

## Miscellaneous

- Fixed broken image links in the water rocket example in the documentation.  [#902](https://github.com/OpenMDAO/dymos/pull/902)


*******************************
# Release Notes for Dymos 1.6.1

November 14, 2022

Version 1.6.1 of Dymos addresses bugs described below.

This release also includes the start of an implementation to allow calculated expressions to be used as
constraints and timeseries outputs, but this feature is still undergoing development and documentation.

## Backwards Incompatible API Changes & Deprecations

None

## Enhancements

* Added the ability to include calculation expressions in timeseries outputs. This feature is still undergoing development and is not supported as of this release. [#846](https://github.com/OpenMDAO/dymos/pull/846)
* Added ability to specify constraint metadata (scaler, ref, linear, etc) when calling `link_phases`. [#858](https://github.com/OpenMDAO/dymos/pull/858)

## Bug Fixes

* Fixed a bug in polynomial controls and rates were included in timeseries outputs for ExplicitShooting. [#840](https://github.com/OpenMDAO/dymos/pull/840)
* Fixed an interpolation issue where scipy now requires unique x-axis values. [#842](https://github.com/OpenMDAO/dymos/pull/842)
* Added a better error message when time units are `None` and state rate introspection fails to find valid state units based on state rate units. [#851](https://github.com/OpenMDAO/dymos/pull/851)
* Fixed an issue that was causing states not to show up in the timeseries outputs for AnalyticPhase. [#853](https://github.com/OpenMDAO/dymos/pull/853)
* Fixed a bug that was causing errors when phase linkages involved parameters. [#858](https://github.com/OpenMDAO/dymos/pull/858)

## Miscellaneous

* Various github workflow issues addressed due to changes in dependencies. [#844](https://github.com/OpenMDAO/dymos/pull/844) [#849](https://github.com/OpenMDAO/dymos/pull/849) [#854](https://github.com/OpenMDAO/dymos/pull/854)

*******************************
# Release Notes for Dymos 1.6.0

October 17, 2022

Version 1.6.0 of Dymos implements many improvements since our last release.

The new AnalyticPhase is useful for when the state values are analytically known as functions of time and other parameters.
States are not only outputs and therefore values cannot be fixed or bounded, though they can be constrained with boundary and path constraints.
AnalyticPhase does not support controls (or polynomial controls).
Unlike other phase types which use the ODE solver in scipy.solve_ivp to propagate a trajectory based on the current control history, AnalyticPhase simply provides its outputs.
AnalyticPhase does not have a notion of segments.
Instead, it simply evaluates the value of the states at some set of `N` nodes, where `N` is given by the user and the node locations are the Legendre-Gauss-Lobatto nodes in non-dimensional phase time.

It will also be the last version of the code before we introduce more tailored phase classes that are more closely coupled to their transcription.
In the following 1.7.0 release, instantiating `Phase` will result in a deprecation warning and inform the user on how to change their phase instantiation calls.

Dymos now automatically generates a linkage report html file when appropriate.
This report looks similar to the N2, and helps to visualize the continuity of variables across phases within a trajectory.

Other changes include a significant cleanup of the timeseries output code, allowing users to rename states/controls/etc. in the timeseries outputs as they see fit.
Timeseries can now also provide approximate rates of outputs (based on fitting them on the current grid and interpolating the derivative of the resulting polynomial).
In an upcoming release, the user will be able to easily apply constraints to these approximated rates.

## Backwards Incompatible API Changes & Deprecations

* State option `connected_initial` is deprecated in favor of `input_initial`. This is more in-line with the time option, and reflects the fact that there's no requirement that the user actually connect to the value.
* The Lagrange interpolation and differentiation matrices are now generated much more efficiently, but change in the order of operations may result in small numerical differences in interpolatation of states and controls (generally on the order of 1E-12).

## Enhancements

* Refactored TimeComp to avoid unnecessary recalculation of partials. [#759](https://github.com/OpenMDAO/dymos/pull/759)
* Added a new N2-like linkage report to show continuity of variables between phases. [#764](https://github.com/OpenMDAO/dymos/pull/764)
* Added timeseries rates for pseudospectral transcriptions. [#781](https://github.com/OpenMDAO/dymos/pull/781)
* Trajectory.phases subgroup is now available before setup. [#782](https://github.com/OpenMDAO/dymos/pull/782)
* Changed time series plots directory to be under the problem reports directory. [#784](https://github.com/OpenMDAO/dymos/pull/784)
* `run_problem` argument `restart` now allows a specific Case to be provided. [#801](https://github.com/OpenMDAO/dymos/pull/801)
* Added warning when restarting with fix_final=True. [#804](https://github.com/OpenMDAO/dymos/pull/804)
* Added the classic cart-pole example problem - thank you @kanekosh! [#808](https://github.com/OpenMDAO/dymos/pull/808)
* Updated assert_timeseries_near_equal to work with more general cases [#810](https://github.com/OpenMDAO/dymos/pull/810)
* Refactored the timeseries handling to be more general and less transcription-specific. [#816](https://github.com/OpenMDAO/dymos/pull/816)
* Added AnalyticPhase for phases in which no numerical integration is necessary. [#833](https://github.com/OpenMDAO/dymos/pull/833)

## Bug Fixes

* Fixed an error when an invalid rate source is provided for a state. [#761](https://github.com/OpenMDAO/dymos/pull/761)
* Fixed a bug in size of trajectory parameters as passed to phases. [#770](https://github.com/OpenMDAO/dymos/pull/770)
* Minor changes to fix occasional report testing failures. [#785](https://github.com/OpenMDAO/dymos/pull/785)
* Fixed example code impacted by matplotlib change [#826](https://github.com/OpenMDAO/dymos/pull/826)
* Fixed condition under which trajectory linkage reports are generated. [#828](https://github.com/OpenMDAO/dymos/pull/828)
* Fixed issue in reentry example that was causing optimization to fail under SLSQP. [#831](https://github.com/OpenMDAO/dymos/pull/831)

## Miscellaneous

* Updated optimizer settings to improve convergence of the racecar example. [#752](https://github.com/OpenMDAO/dymos/pull/752)
* Changed setup to use setuptools instead of distutils. [#757](https://github.com/OpenMDAO/dymos/pull/757)
* Changes to use OpenMDAOs updated report system. [#763](https://github.com/OpenMDAO/dymos/pull/763)
* Updated github workflow and setup.py. [#768](https://github.com/OpenMDAO/dymos/pull/768)
* Switched to using mamba in CI workflow due to issues with conda. [#772](https://github.com/OpenMDAO/dymos/pull/772)
* Updated optimizer settings to improve convergence of two-burn orbit raise example. [#775](https://github.com/OpenMDAO/dymos/pull/775)
* Cleaned up error message from `assert_cases_equal` test utility a bit. [#779](https://github.com/OpenMDAO/dymos/pull/779)
* Cleaned up CI and testing. [#788](https://github.com/OpenMDAO/dymos/pull/788)
* Fixed two problems with the linkage report test [#793](https://github.com/OpenMDAO/dymos/pull/793)
* Updated pyoptsparse build in github workflows [#805](https://github.com/OpenMDAO/dymos/pull/805)
* Added small changes to improve coverage. [#809](https://github.com/OpenMDAO/dymos/pull/809)
* Fixed CI dependency issues based on some broken jupyter-book dependencies. [#812](https://github.com/OpenMDAO/dymos/pull/812)
* Various code cleanup items added. [#807](https://github.com/OpenMDAO/dymos/pull/807)
* Made a few minor cleanups for the cartpole example. [#815](https://github.com/OpenMDAO/dymos/pull/815)

*******************************
# Release Notes for Dymos 1.5.0

May 05, 2022

This is version 1.5.0 of Dymos.

This version adds better support for OpenMDAO's automatic generation of reports.
Sub-problems, used to wrap ODEs for simulation or the ExplicitShooting transcription, will no longer generate problem reports by default.

Phase parameters have been refactored so that they are always available as an input (`path.to.phase.parameters:{param_name}`) and an output (`path.to.phase.parameter_vals:{param_name}`).
This should make connecting parameters to downstream models more intuitive since the user is no longer required to do so via promotion.
We also fixed some instances where Dymos would not detect invalid targets for parameters in some situations.
This may detect issues in existing models that were previously undetected by Dymos.

The model for providing the 1976 standard atmosphere, `USatm1976Comp`, was originally intended to support some tests in Dymos but over time has been used by other projects.
It has been changed such that it can accept either geodetic or geopotential altitude as input.
The default is currently geopotential input (no transformation is performed on the altitude before interpolating the atmosphere table).
In the future this will likely change to geodetic, since it's more likely to be the intent of the user that the provided altitude is geodetic.
This will result in a minor change in the interpolated atmospheric properties, especially at higher altitudes, but we've found it to have only minor impact on results of several cases.

Several enhancements and bug fixes have been performed regarding constraints.
The recent constraint aliasing update in OpenMDAO allows us to apply separate constraints to different indices of the same variables, meaning that we can apply path and boundary constraints directly to the timeseries outputs and no longer need separate pass-thru components to handle them.
Also, importantly, all boundary constraints and path constraints are currently marked as nonlinear.
Dymos previously attempted to determine when constraints could potentially be linear, but it's difficult to know the linearity of inputs from outside the trajectory or phase.
The logic to determine this was starting to get quite complex, so we decided to remove it entirely for now.
If the user is 100% certain that a boundary constraint or path constraint can validly be treated as linear, then the `linear=True` argument can be added to `add_boundary_constraint` or `add_path_constraint`.
Just note that this WILL cause the failure of the optimization if the constraint output is not actually a linear function of the design variables.
This assumption of non-linearity also works around some issues we were seeing with the way OpenMAO and pyOptSparse handle linear constraints with nonzero y-intercepts.
**This assumption of nonlinearity in boundary and path constraints may cause some models to have different convergence behavior and we welcome feedback if you experience problems due to this.**

## Enhancements

* Users can now pass `simulate_kwargs` from `dymos.run_problem`. [#720](https://github.com/OpenMDAO/dymos/pull/720)
* Added `set_parameter_options` method to `Trajectory` for better API consistency. [#721](https://github.com/OpenMDAO/dymos/pull/721)
* USatm1976Comp now accepts altitude as either geopotential or geodetic. [#735](https://github.com/OpenMDAO/dymos/pull/735)
* Disabled reports by default for subproblems under simulate and ExplicitShooting.  [#741](https://github.com/OpenMDAO/dymos/pull/741)
* Constraints are now applied directly to timeseries outputs, and several other constraint improvements.  [#743](https://github.com/OpenMDAO/dymos/pull/743)

## Bug Fixes

* Fixed a bug that caused a hang when running the constraint report on MPI. [#731](https://github.com/OpenMDAO/dymos/pull/731)
* Fixed a bug where trajectory parameters with no valid target will not flag an error. [#744](https://github.com/OpenMDAO/dymos/pull/744)

## Miscellaneous

* Fixed plots on several of the documentation examples. [#709](https://github.com/OpenMDAO/dymos/pull/709)
* Added an example of using OpenMDAO's FunctionComp API for an ODE. [#710](https://github.com/OpenMDAO/dymos/pull/710)
* Cleanup of the configuration/introspection stack. [#713](https://github.com/OpenMDAO/dymos/pull/713)
* Reuse the OpenMDAO _ReprClass for `_unspecified` to deal with some issues with MPI operation. [#724](https://github.com/OpenMDAO/dymos/pull/724)
* Added workflow_dispatch trigger on github so Dymos tests can be triggered by OpenMDAO pull requests. [#732](https://github.com/OpenMDAO/dymos/pull/732) [#736](https://github.com/OpenMDAO/dymos/pull/736)
* Fixed an incorrect equation in the documentation of the calculation of flight path angle. [#734](https://github.com/OpenMDAO/dymos/pull/734)

*******************************
# Release Notes for Dymos 1.4.0

January 05, 2022

This is version 1.4.0 of Dymos.
It includes a fix for grid refinement and simulation with parameters, some minor performance improvements, and various documentation updates.

## Enhancements

* Disabled check_partials of Dymos core components by default. [#686](https://github.com/OpenMDAO/dymos/pull/686)
* Added performance improvements for PseudospectralTimeseriesOutputComp. [#688](https://github.com/OpenMDAO/dymos/pull/688)
* Added a warning if bounds are applied to a state during solve_segments. [#691](https://github.com/OpenMDAO/dymos/pull/691)
* Expanded the included 1976 standard atmosphere model to cover -15000 ft to 250000 ft and doc cleanup. [#699](https://github.com/OpenMDAO/dymos/pull/699)
* Added the Bryson-Denham example problem and other various documentation improvements. [#700](https://github.com/OpenMDAO/dymos/pull/700) [#702](https://github.com/OpenMDAO/dymos/pull/702) [#704](https://github.com/OpenMDAO/dymos/pull/704)
* Changed deprecated 'value' metadata usages to 'val'. [#706](https://github.com/OpenMDAO/dymos/pull/706)

## Bug Fixes

* Fixed a bug that could result in incorrect parameter values in simulation. [#684](https://github.com/OpenMDAO/dymos/pull/684)
* Fixed example that doesn't work in google collab. [#690](https://github.com/OpenMDAO/dymos/pull/690)
* Fixed a bug in grid refinement error estimation for state rates not from ODE. [#703](https://github.com/OpenMDAO/dymos/pull/703)

## Miscellaneous

* Added [notebooks] spec when installing with pip. [#695](https://github.com/OpenMDAO/dymos/pull/695)
* Added CI matrix entry for testing without pyoptsparse. [#697](https://github.com/OpenMDAO/dymos/pull/697)

*******************************
# Release Notes for Dymos 1.3.0

November 19, 2021

This is version 1.3.0 of Dymos.

This release of Dymos introduces an ExplicitShooting transcription that provides an explicit Runge-Kutta integration of the ODE across a phase.
This transcription is currently limited to fixed-step RK methods (RK4 being the default).
Timeseries outputs are provided at the start/end of each segment in the phase.
This is similar to the solve-segments capability in the collocation transcriptions, but fixed-step will provide _an_ answer (albeit inaccurate) across the integration rather than failing to converge if the dynamics become highly nonlinear.

## Enhancements

* Added path constraints to the explicit shooting transcription. [#659](https://github.com/OpenMDAO/dymos/pull/659)
* Added control continuity enforcement to ExplicitShooting transcription, and refactored continuity components in general. [#660](https://github.com/OpenMDAO/dymos/pull/660)
* Added indication of fixed variables to linkage report. [#662](https://github.com/OpenMDAO/dymos/pull/662)
* Replaced the tensordot in the compute method of timeseries_output_comp with a regular dot product to remove a performance bottleneck.  [#665](https://github.com/OpenMDAO/dymos/pull/665)
* Added constraint report to summarize boundary and path constraints for each phase of a trajectory. [#666](https://github.com/OpenMDAO/dymos/pull/666)
* Added ExplicitShooting to transcriptions [#669](https://github.com/OpenMDAO/dymos/pull/669)
* Significantly improved speed of ExplicitShooting [#670](https://github.com/OpenMDAO/dymos/pull/670)
* Added Radau, BDF and LSODA as options for scipy's integration method when using simulate [#675](https://github.com/OpenMDAO/dymos/pull/675)

## Bug Fixes

* Removed solver for connected linkages. Its only needed for solve_segments. [#668](https://github.com/OpenMDAO/dymos/pull/668)
* Changed default value of units in Trajectory.add_parameter to _unspecified. [#673](https://github.com/OpenMDAO/dymos/pull/673)
* Added fix to allow parameters with static_targets=True to work with ExplicitShooting [#679](https://github.com/OpenMDAO/dymos/pull/679)
* Fixed formatting in the constraint report [#680](https://github.com/OpenMDAO/dymos/pull/680)

## Miscellaneous

* None

*******************************
# Release Notes for Dymos 1.2.0

October 12, 2021

This is version 1.2.0 of Dymos.

The release provides compatibility with OpenMDAO >=3.13.0 and adds
some performance improvements.

While we are beginning to bring a true explicit shooting capability
to Dymos, those capabilities are not fully filled out as of this release.

## Backwards Incompatible API Changes & Deprecations

* Dymos 1.2.0 requires OpenMDAO >= 3.13.0, due to changes in the way indices are specified in OpenMDAO.

## Enhancements

* Update run_problem.py to return success state [#634](https://github.com/OpenMDAO/dymos/pull/634)
* Added an experimental explicit shooting transcription to dymos [#637](https://github.com/OpenMDAO/dymos/pull/637)
* Added control rates and their derivatives when using ExplicitShooting. [#645](https://github.com/OpenMDAO/dymos/pull/645)
* Rewrite of the USatm1976Comp to use pre-computed akima coefficients for interpolation. It is now complex-safe and considerably faster. [#652](https://github.com/OpenMDAO/dymos/pull/652)
* Allow addition of ODE outputs to ExplicitShooting timeseries [#654](https://github.com/OpenMDAO/dymos/pull/654)

## Bug Fixes

* Fixed an issue where simulation was not working when running under MPI in run_problem [#628](https://github.com/OpenMDAO/dymos/pull/628)
* Added a better error message when simulate fails due to the inability to find a good step size. [#630](https://github.com/OpenMDAO/dymos/pull/630)
* Fixes a bug where t_initial_targets and t_duration_targets would not work if input_initial or input_duration were True, respectively. [#656](https://github.com/OpenMDAO/dymos/pull/656)
* Fix to eliminate warning messages related to the recent indexing update to OpenMDAO. [Requires OpenMDAO >= 3.12.0] [#636](https://github.com/OpenMDAO/dymos/pull/636)
* Removed exceptions introduced in OpenMDAO PR [#2279](https://github.com/OpenMDAO/OpenMDAO/pull/2279). [#653](https://github.com/OpenMDAO/dymos/pull/653)

## Miscellaneous

* Fixed issue in executable notebooks. [#631](https://github.com/OpenMDAO/dymos/pull/631)
* Updated CI matrix to test against latest release and development versions of OpenMDAO. [#638](https://github.com/OpenMDAO/dymos/pull/638)

*******************************
# Release Notes for Dymos 1.1.0

July 22, 2021

This is version 1.1.0 of Dymos.
The release provides compatibility with OpenMDAO >=3.10.0, updates the
documentation to JupyterBook, and adds a few new features.

## Backwards Incompatible API Changes & Deprecations

* Removed vectorize_derivs option from phase objectives due to OpenMDAO update. [#605](https://github.com/OpenMDAO/dymos/pull/605)
* The dynamic argument on add_parameter has been removed. A new argument static_target has been added which has opposite meaning of dynamic. [#591](https://github.com/OpenMDAO/dymos/pull/591)
* Updated phase.interpolate to automatically detect variable type, renamed to phase.interp.  Old version is deprecated. [#592](https://github.com/OpenMDAO/dymos/pull/592)

## Enhancements

* Documentation updated to JupyterBook. [#611](https://github.com/OpenMDAO/dymos/pull/611) [#613](https://github.com/OpenMDAO/dymos/pull/613) [#614](https://github.com/OpenMDAO/dymos/pull/614) [#615](https://github.com/OpenMDAO/dymos/pull/615) [#616](https://github.com/OpenMDAO/dymos/pull/616) [#618](https://github.com/OpenMDAO/dymos/pull/618)
* simulate_options are now stored within each Phase. [#610](https://github.com/OpenMDAO/dymos/pull/610)
* Removed vectorize_derivs option from phase objectives due to OpenMDAO update. [#605](https://github.com/OpenMDAO/dymos/pull/605)
* Updated dymos to handle the new OpenMDAO distributed I/O approach [#597](https://github.com/OpenMDAO/dymos/pull/597)
* The dynamic argument on add_parameter has been removed. A new argument static_target has been added which has opposite meaning of dynamic. [#591](https://github.com/OpenMDAO/dymos/pull/591)
* Updated phase.interpolate to automatically detect variable type, renamed to phase.interp.  Old version is deprecated. [#592](https://github.com/OpenMDAO/dymos/pull/592)

## Bug Fixes

* Fixed an issue with units in linkage constraints. [#620](https://github.com/OpenMDAO/dymos/pull/620)
* Fix for key error when performing order reduction under hp adaptive refinement. [#590](https://github.com/OpenMDAO/dymos/pull/590)
* Parameters in the ODE system now respect both dynamic=True and False. [#581](https://github.com/OpenMDAO/dymos/pull/581)

## Miscellaneous

* Added missing example Brachistochrone with upstream initial and duration states. [#623](https://github.com/OpenMDAO/dymos/pull/623)
* Added require_pyoptsparse to all tests that use pyOptSparseDriver [#624](https://github.com/OpenMDAO/dymos/pull/624)
* Added a couple of fixes to examples docs [#622](https://github.com/OpenMDAO/dymos/pull/622)
* Fixed some typos in 'Getting Started' section of the docs [#621](https://github.com/OpenMDAO/dymos/pull/621)
* Install coveralls from pypi in github workflow [#601](https://github.com/OpenMDAO/dymos/pull/601)
* Change base_dir arg to coveralls [#600](https://github.com/OpenMDAO/dymos/pull/600)
* Fixed minor typos in docs. [#599](https://github.com/OpenMDAO/dymos/pull/599)
* Removed require_pyoptsparse and moved it to OpenMDAO. [#595](https://github.com/OpenMDAO/dymos/pull/595)
* Added publishing mkdocs to gh-pages. [#594](https://github.com/OpenMDAO/dymos/pull/594)
* Added github actions workflow for CI. [#589](https://github.com/OpenMDAO/dymos/pull/589)
* Readme updated to point to JOSS paper. [#578](https://github.com/OpenMDAO/dymos/pull/578)
* Updated JOSS bibliography. [#573](https://github.com/OpenMDAO/dymos/pull/573) [#574](https://github.com/OpenMDAO/dymos/pull/574) [#576](https://github.com/OpenMDAO/dymos/pull/576) [#577](https://github.com/OpenMDAO/dymos/pull/577)
* Fixed some references in JOSS paper. [#572](https://github.com/OpenMDAO/dymos/pull/572)
* Minor grammar and consistency edits for JOSS paper. [#571](https://github.com/OpenMDAO/dymos/pull/571)

********************************
# Release Notes for Dymos 1.0.0

March 25, 2021

This is version 1.0.0 of Dymos.
This release primarily removes some deprecated experimental features, along with implementing a few bug fixes.

## Backwards Incompatible API Changes & Deprecations

* The RungeKutta Transcription is removed. [#550](https://github.com/OpenMDAO/dymos/pull/550)
* Two-character location specifiers ('++', '--', '+-', '-+') are removed in favor of 'initial' and 'final' [#556](https://github.com/OpenMDAO/dymos/pull/556)
* User must now specify `solve_segments='forward'` or `solve_segments='backward'` when using solve_segments capability (True is no longer valid). [#557](https://github.com/OpenMDAO/dymos/pull/557)
* `add_input_parameter` and `add_design_parameter` dropped in favor of `add_parameter`. [#558](https://github.com/OpenMDAO/dymos/pull/558) [#561](https://github.com/OpenMDAO/dymos/pull/561)
* Removed the deprecated command line interface due to it being a somewhat hackish abuse of OpenMDAO's hooks. [#563](https://github.com/OpenMDAO/dymos/pull/563)
* Removed the deprecated 'disc' node subset in favor of the more correct 'state_disc' [#565](https://github.com/OpenMDAO/dymos/pull/565)
* Removed the deprecated 'custom_targets' option for parameters.' [#565](https://github.com/OpenMDAO/dymos/pull/565)

## Enhancements

* User will now be warned when multiple timeseries outputs were attempted with the same name. [#567](https://github.com/OpenMDAO/dymos/pull/567)

## Bug Fixes

* The `ode_class` option of phase is now marked non-recordable.  [#555](https://github.com/OpenMDAO/dymos/pull/555)
* Fixed doc building due to a change in OpenMDAO. [#564](https://github.com/OpenMDAO/dymos/pull/564)

## Miscellaneous

* Modified examples to use the preferred `fix_initial=True` for time and states instead of "pinched bounds", e.g. `initial_bounds=(0, 0)`. [#560](https://github.com/OpenMDAO/dymos/pull/560)


********************************
# Release Notes for Dymos 0.18.1

February 18, 2021

This release of Dymos adds several examples demonstrating various capabilities of the code.
Per user request, the ODE of a system can now be provided via a callable function that
takes `num_nodes` and any other potential initialization keywod arguments.
This allows OpenMDAO's ExecComp to be used as an ODE if wrapped in a lambda, for instance.

Some bugs were fixed as part of introducing these examples.  For instance, default values for
states and times were being ignored - this is now fixed.  In addition, Trajectory parameters
are now saved to the simulation database file.

This is expected to be the final release of Dymos before v1.0.0, when several existing
deprecated features will be removed from the code.

## Backwards Incompatible API Changes & Deprecations

None

## Enhancements

* Added the racecar example from @pwmdebuck, demonstrating cycling constraints and integration about arclength instead of time. [#535](https://github.com/OpenMDAO/dymos/pull/535)
* Simplified the SSTO example to use a single ODE component with complex-step. [#534](https://github.com/OpenMDAO/dymos/pull/534)
* Added a balanced field length example. [#533](https://github.com/OpenMDAO/dymos/pull/533)
* Added the ability to use a callable that returns a System as an ODE. [#528](https://github.com/OpenMDAO/dymos/pull/528)

## Bug Fixes

* Removed duplication of inputs to timeseries when multiple outputs may use the same source data. [#543](https://github.com/OpenMDAO/dymos/pull/543)
* Fixed a bug where timeseries outputs of non-dynamic ODE outputs would cause an exception. [#521](https://github.com/OpenMDAO/dymos/pull/521)

## Miscellaneous

* Reworked the cannonball example to make it more simple [#545](https://github.com/OpenMDAO/dymos/pull/545)
* Placed some more tests under the use_tempdirs decorator to clean up output. [#530](https://github.com/OpenMDAO/dymos/pull/530)
* Added a test that checks all docstrings vs the NumpyDoc standard. [#526](https://github.com/OpenMDAO/dymos/pull/526)
* Clean up implementation of wildcard units when adding multiple timeseries at once. [#523](https://github.com/OpenMDAO/dymos/pull/523)

********************************
# Release Notes for Dymos 0.18.0

January 21, 2021

This release of Dymos brings a few improvements and bug fixes.

We've implemented introspection for boundary and path constraints such that units and shapes of the constrained quantities can be determined during the setup process and no longer are required to be specified.
The use of solve_segments to have a solver converge the dynamics and thus provide a shooting method at the optimizer has been improved.  Option solve_segments can now take on values 'forward' or 'backward' to provide forward or backward shooting.
Linkage constraints will now result in an error if the quantity on each side of the linkage is governed by the optimizer (is a design variable) but is fixed in value (not allowed to be varied).

## Backwards Incompatible API Changes & Deprecations

* The command line interface is deprecated due to inconsistent behavior. [#505](https://github.com/OpenMDAO/dymos/pull/505)

## Enhancements

* Changed CoerceDesvar to handle vector values for ref/ref0/scaler/adder. Carried over changes to for defect ref/defect scaler. [#464](https://github.com/OpenMDAO/dymos/pull/464)
* Changed solve_segments to allow it to be used when neither fix_initial nor fix_final are True. [#490](https://github.com/OpenMDAO/dymos/pull/490)
* Added detection of unsatisfiable linkage constraints (where quantities on both sides cannot be changed by the optimizer). [#502](https://github.com/OpenMDAO/dymos/pull/502)
* Added introspection for boundary and path constraints. [#506](https://github.com/OpenMDAO/dymos/pull/506)

## Bug Fixes

* Fixed a bug with using grid refinement if one or more of the states are vector-valued. [#492](https://github.com/OpenMDAO/dymos/pull/492)
* Fixed issue where show_plots argument to run_problem is permanently changing matplotlib backend. [#497](https://github.com/OpenMDAO/dymos/pull/497)
* Fixed a bug where trajectory.simulate() would crash if trajectory were not named the common 'traj' name. [#500](https://github.com/OpenMDAO/dymos/pull/500)
* Fixed a bug that prevented restart from working with multiphase trajectories, and now include static parameters in the solution file. [#510](https://github.com/OpenMDAO/dymos/pull/510)
* Fixed a bug associated with polynomial control rate sources, and added coverage of more cases. [#513](https://github.com/OpenMDAO/dymos/pull/513)
* Fixed exception when simulating a trajectory with a parameter for which None was declared as the target for one of its phases. [#515](https://github.com/OpenMDAO/dymos/pull/515)

## Miscellaneous

None

****************************************************************
# Release Notes for Dymos 0.17.0

December 14, 2020

This release of Dymos adds a few important features:
- Dymos can now automatically generate plots of all timeseries outputs
- State rates may now be tagged in the ODE

There are also several bug fixes.

## Backwards Incompatible API Changes & Deprecations

* None

## Enhancements

* Dymos can now automatically generate plots of all timeseries outputs. [#469](https://github.com/OpenMDAO/dymos/pull/469)
* States can now be discovered automatically by tagging their rate variables in the ODE. [#477](https://github.com/OpenMDAO/dymos/pull/477) [#481](https://github.com/OpenMDAO/dymos/pull/481) [#484](https://github.com/OpenMDAO/dymos/pull/484)

## Bug Fixes

* Fixed issue where polynomial controls were not able to be linked across phases. [#462](https://github.com/OpenMDAO/dymos/pull/462)
* Fixed a bug that was preventing parameters from being state rate sources. [#466](https://github.com/OpenMDAO/dymos/pull/466)
* Users may now specify parameter shapes as integers or iterables other than tuples. [#467](https://github.com/OpenMDAO/dymos/pull/467)

## Miscellaneous

* Fix for apache license string in setup.py classifiers since it was not recognized by PyPI. [#458](https://github.com/OpenMDAO/dymos/pull/458)
* Added a long description for PyPI. [#460](https://github.com/OpenMDAO/dymos/pull/460)
* Some documentation cleanup for the JOSS review  [#474](https://github.com/OpenMDAO/dymos/pull/474) [#488](https://github.com/OpenMDAO/dymos/pull/488)
* Dropped the dependency on the parameterized package with the intent to utilize subTest in the future.  [#479](https://github.com/OpenMDAO/dymos/pull/479)


********************************
# Release Notes for Dymos 0.16.1

November 16, 2020

This release of Dymos fixes an issue that now allows portions of an array
output to be connected to a parameter.

This version works with OpenMDAO 3.3.0 but version 3.4.1 offers some
improved handling of parameters.

## Backwards Incompatible API Changes & Deprecations

* Parameter shapes in 0.16.1 were stored as (1,) + the shape of the parameter.  Now they are shaped as expected. [#444](https://github.com/OpenMDAO/dymos/pull/444)

## Enhancements

* State shapes and units, if not explicitly given, are now pulled from targets (if present and uniquely defined), or from the rate source variable. [#449](https://github.com/OpenMDAO/dymos/pull/449)

## Bug Fixes

* Fixes a bug where user-defined shapes of states were colliding with those found during introspection, and other state introspection updates. [#449](https://github.com/OpenMDAO/dymos/pull/449)

* Fixes a bug that prevented the use of numpy arrays as a boundary constraints [#450](https://github.com/OpenMDAO/dymos/pull/450)

## Miscellaneous

* Added test to verify functionality of OpenMDAO 3.4.1 that allows the final value of a control (or a partial portion of any output) to be connected to a parameter. [#445](https://github.com/OpenMDAO/dymos/pull/445)

********************************
# Release Notes for Dymos 0.16.0

October 23, 2020

This is a long-overdue release of Dymos that features several useful new features.
By taking advantage of recent changes to the OpenMDAO setup process, we can use introspection to determine a lot of things automatically now.
For instance, the user will no longer be required to specify the shape or default units of timeseries outputs or constraints (although they can override units if they choose).

Also on the topic of timeseries, dymos now supports glob patterns when adding ODE outputs to the timeseries.
Adding all 100 outputs of a large ODE to the timeseries output can now be accomplished in a single line:

```
phase.add_timeseries_output('*')
```

Introspection is also used for states, time, and controls.
States will get their default units and shape from the associated state rate variable now.
Controls, Polynomial Controls, and Parameters will now get their default units and shapes from their targets.

With the advent of automatic IndepVarComps in OpenMDAO, there no longer needs to be a distinction between InputParameters and DesignParameters.
From here on out, users just add `parameters` to a Trajectory or Phase.
A user can choose to make them design variables (in which case they perform as DesignParameters), or not.
Connecting an external variable to the parameter makes it function as a InputParameter.

From here on, we plan on making releases of Dymos roughly once per month.

## Backwards Incompatible API Changes & Deprecations:

- `add_input_parameter` and `add_design_parameter` are **deprecated** and replaced with `add_parameter` [#365](https://github.com/OpenMDAO/dymos/pull/365)
- The use of two-character location strings ('--' and '++') are replaced by 'initial' and 'final' when linking phases. [#427](https://github.com/OpenMDAO/dymos/pull/427)
- The endpoint conditions component, used to pull out initial or final values of a variable in a phase, have been removed.  Values should instead by pulled from the timeseries. [#427](https://github.com/OpenMDAO/dymos/pull/427)
- The results of simulate are now automatically recorded to a database recorder 'dymos_simulation.db' and stored in a case named 'final'.  Use `CaseReader('dymos_simulation.db').get_case('final')` to access the case. instead of the previous behavior `CaseReader('dymos_simulation.db').get_case(-1)` [#437](https://github.com/OpenMDAO/dymos/pull/437)
- The RungeKutta transcription is **deprecated**. It is functionally equivalent to the GaussLobatto transcription with an order of 3, but since it's signficantly different under the hood it was being a drag on development.

One note on another backwards incompatibility.
Since we're using some newer OpenMDAO capabilities for the way in which parameters are handled, it is currently not possible to connect an upstream output to a parameter in dymos _when src_indices are used_.
We plan on fixing this issue with an upcoming OpenMDAO release.
If you rely on this capability, we recommend waiting for that update before using this version of Dymos, or using an intermediate "pass-through" component to pull the correct index from the original output and then passing _that_ as the value of a dymos parameter.

## Enhancements:

* Dymos components are no longer listed in check_partials output by default.  Use `dymos.options['include_check_partials'] = True` to override this for debugging.  [#438](https://github.com/OpenMDAO/dymos/pull/438)
* Any outputs can now be linked across phases.  Added a more general and powerful `add_linkage_constraint` method, but the simpler `link_phases` method remains in place for simple continuity. [#422](https://github.com/OpenMDAO/dymos/pull/422)
* Time, controls, and polynomial control units are now determined from targets. [#412](https://github.com/OpenMDAO/dymos/pull/412)
* States now pull their default units and shapes from the rate source. [#398](https://github.com/OpenMDAO/dymos/pull/398)
* Phase method `add_timeseries_output` now supports wildcards to allow multiple timeseries outputs to be added at once. [#387](https://github.com/OpenMDAO/dymos/pull/387), #387
* Timeseries outputs automatically detect shape and default units. [#380](https://github.com/OpenMDAO/dymos/pull/380)
* Added ph-adaptive refinement method that is capable of shrinking the grid. [#379](https://github.com/OpenMDAO/dymos/pull/379)
* Control targets will automatically be set to a top-level input of the ODE, if present. [#373](https://github.com/OpenMDAO/dymos/pull/373)
* State targets will automatically be set to a top-level input of the ODE, if present. [#356](https://github.com/OpenMDAO/dymos/pull/356)
* Water-powered rocket MDO example added. [#343](https://github.com/OpenMDAO/dymos/pull/343)
* State rates are included in timeseries outputs by default. [#329](https://github.com/OpenMDAO/dymos/pull/329)
* By default, run_problem will use a problem recorder to record only the 'final' solution after an optimization. [#328](https://github.com/OpenMDAO/dymos/pull/328)
* Move most setup functionality to configure to allow more introspection changes. [#327](https://github.com/OpenMDAO/dymos/pull/327)
* Include examples using IPOPT via pyoptsparse. [#311](https://github.com/OpenMDAO/dymos/pull/311)
* Parameters may now be selectively omitted from the timeseries outputs, but are included by default. [#305](https://github.com/OpenMDAO/dymos/pull/305)
* Added a test of the distributed ODE capability. [#304](https://github.com/OpenMDAO/dymos/pull/304)
* Replace `load_case` with `reinterpolate_solution`. [#296](https://github.com/OpenMDAO/dymos/pull/296)

## Bug Fixes:

* Fixed a bug that was causing matrix-shaped states to have connection errors. [#441](https://github.com/OpenMDAO/dymos/pull/441)
* Fixed a bug where simulate failed when both standard and polynomial controls were present. [#434](https://github.com/OpenMDAO/dymos/pull/434)
* Fixed an issue where parameters are promoted twice in some cases. This will cause errors in an upcoming versions of OpenMDAO. [#429](https://github.com/OpenMDAO/dymos/pull/429)
* Fixed a bug that was causing matrix-shaped states to have connection errors. [#441](https://github.com/OpenMDAO/dymos/pull/441)
* AnalysisError is now raised if scipy.integrate.solve_ivp fails during simulation. [#400](https://github.com/OpenMDAO/dymos/pull/400)
* Fixed a bug involving GaussLobatto transcriptions `get_rate_source` method. [#334](https://github.com/OpenMDAO/dymos/pull/334)
* Fix for simulate encountering errors when time options `input_initial` or `input_duration` were True. [#317](https://github.com/OpenMDAO/dymos/pull/317)
* Fixed a bug in which shaped input parameters were breaking simulate. [#301](https://github.com/OpenMDAO/dymos/pull/301)
* Sort linkage orders to make convergence more repeatable. [#298](https://github.com/OpenMDAO/dymos/pull/298)

## Miscellaneous:

* Switch to use preferred `assert_near_equal` method instead of `assert_rel_error` from OpenMDAO. [#320](https://github.com/OpenMDAO/dymos/pull/320)
* Documentation is now handled via mkdocs instead of sphinx. [#337](https://github.com/OpenMDAO/dymos/pull/337), [#332](https://github.com/OpenMDAO/dymos/pull/332)

## 0.15.0
2020-02-12

* [__Bug Fix__] Phase Linkage units now checks all optimal control variables from the first phase in the linkage for units. Previously units for things that were neither states, time, nor controls were not being assigned.
* [__Enhancement__] Removed Python 2 support.
* [__Enhancement__] Removed deprecated ODE Decorators.
* [__Enhancement__] Added ability to subclass Phase with an assigned ODE and default state variable options.
* [__Docs__] Added docs on _subclassing phases_.
* [__Enhancement__] Automated grid refinement is now available via the `dymos.run_problem` function.
* [__Enhancement__] Grid data is now available upon instantiation of the Transcription instead of being deferred to setup time.
* [__Bug Fix__] The user now gets a meaningful message if Phase.interpolate is called too soon.
* [__Bug Fix__] State rates are now correctly passed through the interleave component that provides timeseries outputs for Gauss-Lobatto transcription.
* [__Enhancement__] Added hypersensitive example problem.
* [__Docs__] Documentation added for grid refinement.
* [__Enhancement__] Deprecated the use of ODE decorators
* [__Enhancement__] Added shuttle reentry example problem

## 0.13.0
2019-07-18

* [__Enhancement__] Phase methods like `set_state_options` and `add_control` no longer use **kwargs in order to make them more IDE friendly.
* [__Enhancement__] Additional timeseries can for outputs can be added to a phase, with interpolation onto a new set of grid points. This enables the concept of tandem phases, where two different ODE's operating over the same time interval can be integrated on different grids. Fast state variables can be integrated on a dense grid while slower state variables are integrated on a more sparse grid, for performance. For an example see the tandem phase documentation in the feature docs.
* [__Enhancement__] ODE options can be specified at the phase level rather than in the ODE. This feature is experimental, but it allows one way of programmatically defining an ODE.
* [__Enhancement__] Changed the use of OpenMDAO to do `import openmdao.api as om` for consistency with the OpenMDAO documentation.
* [__Enhancement__] Dymos is now imported in the examples as import `dymos as dm`
