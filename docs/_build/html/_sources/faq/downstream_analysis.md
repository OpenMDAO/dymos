# How do connect the outputs of Dymos to a downstream analysis?

One of the design goals of Dymos is to allow the trajectory to be a part of a larger multidisciplinary optimization problem.
Sometimes, you may want to take the results from the Dymos trajectory and feed them to some downstream analysis.

In the case of only being concerned with the final value of some parameter, this can be accomplished by connecting the relevant output from
the timeseries with `src_indices=[-1]`.

For example, something like the following might be used to connect the final value of the state `range` to some downstream component.

```
problem.model.connect('trajectory.phase0.timeseries.states:range',
                      'postprocess.final_range',
                      src_indices=[-1])
```

!!! note
    We _highly_ recommend you use the phase timeseries outputs to retrieve outputs from Dymos, since it is transcription-indepdendent.

When the downstream analysis requires the entire trajectory, things get slightly more complicated.
We need to know the number of nodes in a phase when we're building the problem, so the downstream component knows how many values of a variable to expect.
To do this, we can initialize the transcription object and obtain the total number of nodes in the phase using the transcriptions `grid_data.num_nodes` attribute.
The [length-constrained brachistochrone example](../examples/length_constrained_brachistochrone/length_constrained_brachistochrone.md) demonstrates how to do this.