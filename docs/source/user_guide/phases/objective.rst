Objective
---------

Since a |project| problem is optimizer-driven, the problem must have an *objective*.

Unlike typical OpenMDAO problems where the `index` can be used to effectively specify the first or
last value of a variable, optimal control problems have two competing notions of index:  the first
is the location in time where the objective is to be measured, and the second is the index of a
vector valued variable that is to be considered the objective value, which must be scalar.

To remove this ambiguity, Phases provide the method `set_objective`.  In set objective, the option
`loc` may have value `initial` or `final`, specifying whether the objectve is to be quantified at the
start or end of the phase.  The `index` option gives the index into a non-scalar variable value
to be used as the objective, which must be scalar.
