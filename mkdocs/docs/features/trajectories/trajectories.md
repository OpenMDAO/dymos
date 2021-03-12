# Organizing Phases into Trajectories

The majority of real-world use cases of optimal control involve complex trajectories that cannot be modeled with a single phase.
For instance, different phases of a trajectory may have different equations of motion, different control parameterizations, or different path constraints.
Phases are also necessary if the user wishes to impose intermediate constraints upon some variable, by imposing them as boundary constraints at a phase junction.

The *Trajectory* class in Dymos is intended to simplify the development of multi-phase problems.
It serves as a Group which contains the various phases belonging to the trajectory, and it provides linkage constraints that dictate how phases are linked together.
This enables trajectories that are not only a sequence of phases in time, but may include branching behavior, allowing us to do things like track/constrain the path of a jettisoned rocket stage.

It supports a `get_values` method similar to that of Phases that allows the user to retrieve the value of a variable within the trajectory.
When verifying an answer with explicit simulation, the `simulate` method of Trajectory can simulate all of its member phases in parallel, providing a significant performance improvement for some cases.

## Instantiating a Trajectory

Instantiating a Trajectory is simple.  Simply invoke `Trajectory()`.  The trajectory object
itself is an OpenMDAO `Group` which serves as a container for its constituent Phases.

- phases
    An OpenMDAO `Group` or `ParallelGroup` holding the member phases
- linkages
    A Dymos `PhaseLinkageComp` that manages all of the linkage constraints that dictate how the phases are connected.

## Adding Phases

Phases are added to a Trajectory using the `add_phase` method.

{{ api_doc('dymos.Trajectory.add_phase') }}

##  Defining Phase Linkages

Having added phases to the Trajectory, they now exist as independent Groups within the OpenMDAO model.
In order to enforce continuity among certain variables across phases, the user must declare which variables are to be continuous in value at each phase boundary.
There are two methods in dymos which provide this functionality.
The `add_linkage_constraint` method provides a very general way of coupling two phases together.
It does so by generating a constraint of the following form:

\begin{align}
    c = \mathrm{sign}_a \mathrm{var}_a + \mathrm{sign}_b \mathrm{var}_b
\end{align}

Method `add_linkage_constraint` lets the user specify the variables and phases to be compared for this constraint, as well as the location of the variable in each phase (either 'initial' or 'final')
By default this method is setup to provide continuity in a variable between two phases:
- the sign of variable `a` is +1 while the sign of variable `b` is -1.
- the location of variable `a` is 'final' while the location of variable `b` is 'initial'.
- the default value of the constrained quantity is 0.0.

In this way, the default behavior constrains the final value of some variable in phase `a` to be the same as the initial value of some variable in phase `b`.
Other values for these options can provide other functionality.
For instance, to simulate a mass jettison, we could require that the initial value of `mass` in phase `b` be 1000 kg less than the value of mass at the end of phase `a`.
Providing arguments `equals = 1000, units='kg` would achieve this.

Similarly, specifying other values for the locations of the variables in each phase can be used to ensure that two phases start or end at the same condition - such as the case in a branching trajectory or a rendezvous.

While `add_linkage_constraint` gives the user a powerful capability, providing simple state and time continuity across multiple phases would be a very verbose undertaking using this method.
The `link_phases` method is intended to simplify this process.
In the finite-burn orbit raising example, there are three phases:  `burn1`, `coast`, `burn2`.
This case is somewhat unusual in that the thrust acceleration is modeled as a state variable.  
The acceleration needs to be zero in the coast phase, but continuous between `burn1` and `burn2`, assuming no mass was jettisoned during the coast and that the thrust magnitude doesn't change.

### add_linkage_constraint

{{ api_doc('dymos.Trajectory.add_linkage_constraint') }}

### link_phases

{{ api_doc('dymos.Trajectory.link_phases') }}


## Examples of using the `link_phases` method

**Typical Phase Linkage Sequence**

A typical phase linkage sequence, where all phases use the same ODE (and therefore have
the same states), are simply linked sequentially in time.

``` python
t.link_phases(['phase1', 'phase2', 'phase3'])
```

**Adding an Additional Linkage**

If the user wants some control variable, `u`, to be continuous in value between `phase2` and
`phase3` only, they could indicate that with the following code:

``` python
t.link_phases(['phase2', 'phase3'], vars=['u'])
```

**Branching Trajectories**

For a more complex example, consider the case where there are two phases which branch off
from the same point, such as the case of a jettisoned stage.  The nominal trajectory
consists of the phase sequence `['a', 'b', 'c']`.  Let phase `['d']` be the phase that tracks
the jettisoned component to its impact with the ground.  The linkages in this case
would be defined as:

``` python
t.link_phases(['a', 'b', 'c'])
t.link_phases(['b', 'd'])
```

**Specifying Linkage Locations**

Phase linkages assume that, for each pair, the state/control values at the end (`'final'`)
of the first phase are linked to the state/control values at the start of the second phase
(`'initial'`).

The user can override this behavior, but they must specify a pair of location strings for
each pair given in `phases`.  For instance, in the following example phases `a` and `b`
have the same initial time and state, but phase `c` follows phase `b`.  Note since there
are three phases provided, there are two linkages and thus two pairs of location
specifiers given.

``` python
t.link_phases(['a', 'b', 'c'], locs=[('initial', 'initial'), ('final', 'initial')])
```

##  Trajectory-Level Parameters

Often times, there are parameters which apply to the entirety of a trajectory that potentially need to be optimized.
If we implemented these as parameters within each phase individually, we would need some constraints to ensure that they held the same value within each phase.
To avoid this complexity, Dymos Trajectory objects support their own Parameters.

Like their Phase-based counterparts, Trajectory parameters produce may be design variables for the problem or used as inputs to the trajectory from external sources.

When using Trajectory parameters, their values are connected to each phase as an Input Parameter within the Phase.
Because ODEs in different phases may have different names for parameters (e.g. 'mass', 'm', 'm_total', etc) Dymos allows the user to specify the targeted ODE parameters on a phase-by-phase basis using the `targets` and `target_params` option.
It can take on the following values.

*  If `targets` is `None` the trajectory parameter will be connected to the phase input parameter of the same name in each phase, if it exists (otherwise it is not connected to that phase).

*  Otherwise targets should be specified as a dictionary. And the behavior depends on the value associated with each phase name:

    * If the phase name is not in the given dictionary, attempt to connect to an existing parameter of the same name in that phase.

    * If the associated value is None, explicitly omit a connection to that phase.

    * If the associated value is a string, connect to an existing input parameter whose name is given by the string in that phase.

    * If the associated value is a Sequence, create an input parameter in that phase connected to the ODE targets given by the Sequence.

## Explicit Simulation of Trajectories

The `simulate` method on Trajectory is similar to that of the `simulate` method of Phases.  When
invoked, it will perform a simulation of each Phase in the trajectory.
