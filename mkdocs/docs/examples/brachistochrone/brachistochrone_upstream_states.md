# The Brachistochrone with Externally-Sourced Controls

!!! info "Things you'll learn through this example"
    - How to link phase state boundary values to an externally provided value.

This is another modification of the brachistochrone in which the target initial value of a state is provided by an external source (an IndepVarComp in this case).

Rather than the external value being directly connected to the phase, the values are "linked" via constraint.
This is exactly how phase linkages in trajectories work as well, but the trajectory hides some of the implementation.

The following script fully defines the brachistochrone problem with Dymos and solves it.
A new `IndepVarComp` is added before the trajectory which provides `x0`.
An ExecComp then computes the error between `x0_target` (taken from the IndepVarComp) and `x0_actual` (taken from the phase timeseries output).
The result of this calculation (`x0_error`) is then constrained as a normal OpenMDAO constraint.

{{ embed_test('dymos.examples.brachistochrone.doc.test_doc_brachistochrone_upstream_state.TestBrachistochroneUpstreamState.test_brachistochrone_upstream_state') }}
