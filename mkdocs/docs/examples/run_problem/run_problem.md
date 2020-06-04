# The Dymos run_problem function

In the Brachistochrone example we used two methods on the OpenMDAO Problem class to execute the model.

**run_model** Takes the current model design variables and runs a single execution of the Problem's model.  
Any iterative systems are converged, but no optimization is performed.
When using Dymos with an optimizer-driven implicit transcription, `run_model` will **not** produce a physically valid trajectory on output.
If using a solver-driven transcription, the collocation defects will be satisfied (if possible) and the resulting outputs will provide a physically valid trajectory (to the extent possible given the collocation grid).


**run_driver** Runs a driver wrapped around the model (typically done for optimization) and repeatedly executes `run_model` until the associated optimization problem is satisfied.
This approach will provide a physically valid trajectory, to the extent that the grid is sufficient to accurately model the dynamics.
But what happens if the grid is not dense enough to accurately capture the physics of the problem.
This is the purpose of _grid refinement_.
There have been numerous methods of grid refinment posed for implcit optimal control techniques.
In general, they follow the following procedure:

1. Optimize the trajectory
2. Assess errors in the solution
3. Propose a new grid to reduce these errors to an acceptable level.
4. Repeat until the errors are within some acceptable tolerance.

This requires another layer of iteration outside of the OpenMDAO `run_driver` method.
This is the original motivation for Dymos' `run_problem` function.

{{ api_doc('dymos.run_problem') }}


{{ api_doc('dymos.Trajectory', members=['add_phase', 'link_phases']) }}


<!----------------->

<!--### class dymos.Trajectory-->

<!--A Trajectory object serves as a container for one or more Phases, as well as the linkage conditions between phases.-->

<!--****-->

<!--**Public API Methods:**-->

<!--=== "add_phase"-->

<!--    Add a phase to the trajectory.-->
<!--    Phases will be added to the Trajectory's `phases` subgroup.-->


<!--    **add_phase(self, name, phase, \*\*kwargs)**-->

<!--    **Arguments:**-->

<!--    **name**: The name of the phase being added.-->

<!--    **phase**: The Phase object to be added.-->

<!--=== "link_phases"-->

<!--    !!! abstract ""-->

<!--        Specifies that phases in the given sequence are to be assume continuity of the given variables.-->
<!--        This method caches the phase linkages, and may be called multiple times to express more complex behavior (branching phases, phases only continuous in some variables, etc).  The location at which the variables should be coupled in the two phases are provided with a two character string:  - '--' specifies the value at the start of the phase before an initial state or control jump - '-+' specifies the value at the start of the phase after an initial state or control jump - '+-' specifies the value at the end of the phase before a final state or control jump - '++' specifies the value at the end of the phase after a final state or control jump-->
<!--    -->
<!--    -->
<!--        **link_phases(self, phases, vars=None, locs=('++', '--'), connected=False)**-->
<!--    -->
<!--        **Arguments:**-->
<!--    -->
<!--        **phases**: The names of the phases in this trajectory to be sequentially linked.-->
<!--    -->
<!--        **vars**: The variables in the phases to be linked, or '*'.  Providing '*' will time and all states.  Linking control values or rates requires them to be listed explicitly.-->
<!--    -->
<!--        **locs**: A two-element tuple of the two-character location specification.  For every pair in phases, the location specification refers to which location in the first phase is connected to which location in the second phase.  If the user wishes to specify different locations for different phase pairings, those phase pairings must be made in separate calls to link_phases.-->
<!--    -->
<!--        **units**: The units of the linkage residual.  If an integer (default), then automatically determine the units of each variable in the linkage if possible.  Those that cannot be determined to be a time, state, control, design parameter, or control rate will be assumed to have units None.  If given as a dict, it should map the name of each variable in vars to the approprite units.-->
<!--    -->
<!--        **connected**: Set to True to directly connect the phases being linked. Otherwise, create constraints for the optimizer to solve.-->