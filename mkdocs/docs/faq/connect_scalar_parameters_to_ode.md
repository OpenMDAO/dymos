# How do I connect a scalar input to the ODE?

By default, we recommend that users treat all ODE input variables as if they are _potentially_ dynamic.
This allows the user to use the input as either a dynamic control, or as a static design or input parameter.
By default, parameters will "fan" the value out to all nodes.
This allows the partials to be defined in a consistent fashion (generally a diagonal matrix for a scalar input and output) regardless of whether the input is static or dynamic.

**But** there are some cases in which the user may know that a variable will never have the potential to change throughout the trajectory.
In these cases, we can reduce a bit of the data transfer OpenMDAO needs to perform by defining the input as a scalar in the ODE, rather than sizing it based on the number of nodes.

## The Brachistochrone with a static input.

The local gravity `g` in the brachistochrone problem makes a good candidate for a static input parameter.
The brachistochrone generally won't be in an environment where the local acceleration of gravity is varying by any significant amount.


In the slightly modified brachistochrone example below, we add a new option to the BrachistochroneODE `static_gravity` that allows us to decide whether gravity is a vectorized input or a scalar input to the ODE.

=== "brachistochrone_ode.py"
{{ inline_source('dymos.examples.brachistochrone.brachistochrone_ode',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

In the corresponding run script, we pass `{'static_gravity': True}` as one of the `ode_init_kwargs` to the Phase, and declare $g$ as a static design variable using the `dynamic=False` argument.

{{ embed_test('dymos.examples.brachistochrone.doc.test_doc_brachistochrone_static_gravity.TestBrachistochroneStaticGravity.test_brachistochrone_static_gravity') }}