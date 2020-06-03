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


