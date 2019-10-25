===================================
Organizing Phases into Trajectories
===================================

The majority of real-world use cases of optimal control involve complex trajectories that cannot be
modeled with a single phase.  For instance, different phases of a trajectory may have different
equations of motion, different control parameterizations, or different path constraints.  Phases
are also necessary if the user wishes to impose intermediate constraints upon some variable, by
imposing them as boundary constraints at a phase junction.
The *Trajectory* class in Dymos is intended to simplify the development of multi-phase problems.
It serves as a Group which contains the various phases belonging to the trajectory, and it provides
linkage constraints that dictate how phases are linked together. This enables trajectories that
are not only a sequence of phases in time, but may include branching behavior, allowing us to do
things like track/constrain the path of a jettisoned rocket stage.
It supports a `get_values` method similar to that of Phases that allows the user to retrieve the value
of a variable within the trajectory.
When verifying an answer with explicit simulation, the `simulate` method of Trajectory can simulate
all of its member phases in parallel, providing a significant performance improvement for some cases.

Instantiating a Trajectory
--------------------------

Instantiating a Trajectory is simple.  Simply invoke `Trajectory()`.  The trajectory object
itself is an OpenMDAO `Group` consisting of two members:

- phases
    An OpenMDAO `ParallelGroup` holding the member phases
- linkages
    A Dymos `PhaseLinkageComp` that manages all of the linkage constraints that dictate how the phases are connected.

Adding Phases
-------------
Phases are added to a Trajectory using the `add_phase` method.  This does two things.  It stores
a reference to the phase in the Trajectory `_phases` dictionary member, which maps phase names to
the Phases themselves.  Secondly, it adds the Phase subsystem to the `phases` ParallelGroup.  At
this time, Dymos does not support promotion of variable names from Phases to the Trajectory.

Defining Phase Linkages
-----------------------

Having added phases to the Trajectory, they now exist as independent Groups within the OpenMDAO model.
In order to enforce continuity among certain variables across phases, the user must declare which variables
are to be continuous in value at each phase boundary.  The `link_phases` method is intended to simplify
this process.
In the finite-burn orbit raising example, there are three phases:  `burn1`, `coast`, `burn2`.  This
case is somewhat unusual in that the thrust acceleration is modeled as a state variable.  The acceleration
needs to be zero in the coast phase, but continuous between `burn1` and `burn2`, assuming no mass
was jettisoned during the coast and that the thrust magnitude doesn't change.

Trajectory Design Parameters and Input Parameters
-------------------------------------------------
Often times, there are parameters which apply to the entirety of a trajectory that potentially
need to be optimized.  If we implemented these as design parameters within each phase individually,
we would need some constraints to ensure that they held the same value within each phase.  To avoid
this complexity, Dymos Trajectory objects support their own Design Parameters and Input Parameters.

Like their Phase-based counterparts, Trajectory design parameters produce an output which can be used
as a design variable to provide an optimal solution.  Similarly, input parameters provide an input
at the trajectory level which may be connected to some output external to the trajectory.

When using Trajectory Design and Input parameters, their values are connected to each phase as an
Input Parameter within the Phase.  Because ODEs in different phases may have different names
for parameters (e.g. 'mass', 'm', 'm_total', etc) Dymos allows the user to specify the targeted
ODE parameters on a phase-by-phase basis using the `targets` and `target_params` option.
It can take on the following values.

*  If `targets` is `None` the trajectory design or input parameter will be connected to the phase input parameter of the same name in each phase, if it exists (otherwise it is not connected to that phase).

*  Otherwise targets should be specified as a dictionary. And the behavior depends on the value associated with each phase name:

    * If the phase name is not in the given dictionary, attempt to connect to an existing input parameter of the same name in that phase.

    * If the associated value is None, explicitly omit a connection to that phase.

    * If the associated value is a string, connect to an existing input parameter whose name is given by the string in that phase.

    * If the associated value is a Sequence, create an input parameter in that phase connected to the ODE targets given by the Sequence.

Explicit Simulation of Trajectories
-----------------------------------

The `simulate` method on Trajectory is similar to that of the `simulate` method of Phases.  When
invoked, it will perform a simulation of each Phase in the trajectory in parallel.
