===================================
Organizing Phases into Trajectories
===================================

The majority of real-world use cases of optimal control involve complex trajectories that cannot be
modeled with a single phase.  For instance, different phases of a trajectory may have different
equations of motion, different control parameterizations, or different path constraints.  Phases
are also necessary if the user wishes to impose intermediate constraints upon some variable, by
imposing them as boundary constraints at a phase junction.
The *Trajectory* class in |project| is intended to simplify the development of multi-phase problems.
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
this time, |project| does not support promotion of variable names from Phases to the Trajectory.

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

Retrieving the Solution
-----------------------

The current solution (the values of the time, states, controls, design parameters, etc.) for the
trajectory can be obtained using the `get_values` method.  This method is almost identical to
the `get_values` method on Phases, with two exceptions.  First, it accepts an argument `phases`,
which allows the user to specify from which phase or phases the values of a variable should be
retrieved.  This allows the user to obtain values from only those phases which are contiguous in time,
such as the abort branch of a trajectory.
The other difference is that the Trajectory `get_values` method provides a `flat` argument which defaults
to False.  By default, `get_values` returns a dictionary which maps the name of a phase to the values
of the requested variable within that phase.  If `flat == True`, then the return value is a single
array of the variable values sorted in ascending time.  If requesting flattened values for one or
more phases which are parallel in time, the user may receive confusing results!

.. note::
    If the variable is not present in a given phase, it is returned as np.nan for each returned node in the phase.

Explicit Simulation of Trajectories
-----------------------------------

The `simulate` method on Trajectory is similar to that of the `simulate` method of Phases.  When
invoked, it will perform a simulation of each Phase in the trajectory in parallel.

Loading Simulation Results from a File
--------------------------------------

.. toctree::
    :maxdepth: 2
    :titlesonly:
