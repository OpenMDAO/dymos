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
