# Defining ODEs

Optimal control problems contain ordinary differential equations (ODE) or differential algebraic equations (DAE) that dictate the evolution of the state of the system.
Typically this evolution occurs in time, and the ODE represents equations of motion (EOM).
The equations of motion can define a variety of systems, not just mechanical ones.
In other fields they are sometimes referred to as *process equations*.

\begin{align}
    \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d})
\end{align}

To represent EOM, Dymos uses a standard OpenMDAO System (a Group or Component).
This System takes some set of variables as input and computes outputs that include the time-derivatives of the state variables $\bar{x}$.
The ODE may also be a function of the current time $t$.

Finally, the dynamics may be subject to some set of controllable parameters.
In Dymos these are broken into the dynamic controls $\bar{u}$ and the static parameters $\bar{d}$.

##  System Options

ODE Systems in Dymos need provide values at numerous points in time we call nodes.
For performance reasons, it's best if it can do so using vectorized mathematical operations to compute the values simultaneously rather than using the loop to perform the calculation at each node.
Different optimal control transcriptions will need to have computed ODE values at different numbers of nodes, so each ODE system in Dymos is required to support the option `num_nodes`, which is defined in the `initialize` method.

ODE system may define initial options as well.
Since these options in OpenMDAO are typically provided as arguments to the instantiation of the ODE system, the user has the ability to provide additional input keyword arguments using the `ode_init_kwargs` option on Phase.

## Variables of the Optimal Control Problem

Dymos needs to know how state, time, and control variables are to be connected to the System, and needs to know which outputs to use as state time-derivatives.

### Time

The following time options can be set via the `set_time_options` method of Phase:

{{ embed_options('dymos.phase.options.TimeOptionsDictionary', '###Options for Time') }}

### States

States have the following options set via the `add_state` and `set_state_options` methods of Phase:

{{ embed_options('dymos.phase.options.StateOptionsDictionary', '###Options for States') }}

### State Discovery

Dymos will also automatically find and add any states that have been declared in components in the ODE. The syntax
for declaring them is as follows.

```python3

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['state_rate_source:v', 'state_units:m/s'])

```

The state is defined by adding tags to the state rate's output. The tag 'state_rate_source:v' declares that
'v' is the state for which this output ('vdot') is the rate source.  You can also optionally use a tag
in the format 'state_units:m/s' to define units for that state.  If you need to set any other options, then
use `set_state_options` at the phase level.


###  Controls

Inputs to the ODE which are to be dynamic controls are added via the `add_control` and `set_control_options` methods of Phase.
The available options are as follows:


{{ embed_options('dymos.phase.options.ControlOptionsDictionary', '###Options for Control') }}

###  Parameters

Inputs to the ODE which are non-time-varying can be specified using the `add_parameter` method of Phase.
Parameters may be used as design variables (by setting `opt = True`), or they may be connected to an output external to the Phase or Trajectory.
The available options are as follows:

{{ embed_options('dymos.phase.options.ParameterOptionsDictionary', '###Options for Parameters') }}

By default, Dymos assumes that the ODE is defined such that a value of the parameter is provided at each node.
This makes it easier to define the partials for the ODE in a way such that some inputs may be used interchangeably as either controls or parameters.
If an ODE input is only to be used as a static variable (and sized as `(1,)` instead of by the number of nodes, then the user may specify option `dynamic = False` to override this behavior.

## Vectorizing the ODE

In addition to specifying the ODE Options, a system used as an ODE is required to have a metadata
entry called `num_nodes`.  When evaluating the dynamics, these systems will receive time, states,
controls, and other inputs as *vectorized* values, where item in the vector represents the variable
value at a discrete time in the trajectory.

The nodes are discretization or collocation locations in the polynomials which represent
each segment.  The number of nodes in a given phase (to be evaluated by the ODE system) is determined
by the number of segments in the phase and the polynomial order in each segment.  When Dymos instantiates
the ODE system it provides the total number of nodes at which evaluation is required to the ODE system.
Thus, at a minimum, the `initialize` method of components for an ODE system typically look something
like this:

The inputs and outputs of the system are expected to provide a scalar or dimensioned
value *at each node*.  Vectorization of the component via numpy adds a significant performance increase
compared to using a for loop to cycle through calculations at each node.  It's important to remember
that vectorized data is going to be coming in, this is especially important for defining partials.
From the perspective of the ODE system, the outputs at some time `t` only depend on the values
of the input variables at time `t`.  When the output variables are scalar at any given time, this
results in components whose Jacobian matrices are diagonal.  This large degree of sparsity leads
to computational advantages when using sparse-aware optimizers like SNOPT.  Users should declare
the partial derivatives of their components to be sparse (by specifying nonzero rows and columns)
whenever possible.

```python3

    class MyODESystem(ExplicitComponent):

        def initialize(self):
            self.metadata.declare('num_nodes', types=int)

```

For example, if `MyODEComponent` is to compute the linear function $y = a * x + b$ then the
setup, compute, and compute partials methods might look like this:

```python3

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('a', shape=(nn,), units='m')
        self.add_input('x', shape=(nn,), units='1/s')
        self.add_input('b', shape=(nn,), units='m/s')

        self.add_output('y', shape=(nn,), units='m/s')

        r = c = np.arange(nn)
        self.declare_partials(of='y', wrt='a', rows=r, cols=c)
        self.declare_partials(of='y', wrt='x', rows=r, cols=c)
        self.declare_partials(of='y', wrt='b', rows=r, cols=c, val=1.0)

    def compute(self, inputs, outputs):
        a = inputs['a']
        x = inputs['x']
        b = inputs['b']

        outputs['y'] = a * x + b

    def compute_partials(self, inputs, outputs, partials):
        a = inputs['a']
        x = inputs['x']
        b = inputs['b']

        partials['y', 'a'] = x
        partials['y', 'x'] = a
```

A few things to note here.  We can use the `shape` or `val` argument of `add_input` and `add_output`
to dimension each variable.  In this case each variable is assumed to be a scalar at each point in
time (each node).  We use the `rows` and `cols` arguments of `declare_partials` to provide the sparsity.
Here using `arange(nn)` for both gives us a diagonal jacobian with `nn` rows and `nn` columns.  Since
the number of nonzero values in the jacobian is `nn`, we only need to provide `nn` values in the
`compute_partials` method.  It will automatically fill them into the sparse jacobian matrix, in
row-major order.

In this example, the partial of `y` with respect to `b` is linear, so we can simply provide it in
the `declare_partials` call rather than reassigning it every time `compute_partials` is called.
The provided scalar value of `1.0` is broadcast to all `nn` values of the Jacobian matrix.

## Dimensioned Inputs and Outputs

The above example assumes all inputs and outputs are scalar at each node.  Sometimes the user may
encounter a situation in which the inputs and/or outputs are vectors, matrices, or tensors at
each node.  In this case the dimension of the variable is `num_nodes`, with the dimension of the
variable at a single node filling out the remaining indices. A 3-vector is thus dimensioned
`(num_nodes, 3)`, while a 3 x 3 matrix would be sized `(num_nodes, 3, 3)`.

##  Non-Vector Inputs

Declaring inputs as vectors means that they have the potential to be used either as parameters or as dynamic controls which can assume a different value at each node.
For some quantities, such as gravitational acceleration in the Brachistochrone example, we can assume that the value will never need to be dynamic.
To accommodate this, parameters can be declared static with the argument `dynamic=False`.
This prevents Dymos from "fanning out" the static value to the *n* nodes in the ODE system.

##  Providing the ODE to the Phase

Phases in Dymos are instantiated with both the `ode_class` and the `transcription` to be used.
Internally, Dymos needs to instantiate the ODE multiple times.
This instantiation takes the form:

```python
ode_instance = ode_class(num_nodes=<int>, **ode_init_kwargs)
```

This allows an OpenMDAO ExecComp to be used as an ODE via a lambda.
For instance, the brachistochrone ODE can be written as:

```python
ode = lambda num_nodes: om.ExecComp(['vdot = g * cos(theta)',
                                     'xdot = v * sin(theta)',
                                     'ydot = -v * cos(theta)'],
                                    g={'value': 9.80665, 'units': 'm/s**2'},
                                    v={'shape': (num_nodes,), 'units': 'm/s'},
                                    theta={'shape': (num_nodes,), 'units': 'rad'},
                                    vdot={'shape': (num_nodes,), 'units': 'm/s**2'},
                                    xdot={'shape': (num_nodes,), 'units': 'm/s'},
                                    ydot={'shape': (num_nodes,), 'units': 'm/s'},
                                    has_diag_partials=True)

phase = dm.Phase(ode_class=ode, transcription=t)
```

Note the use of `has_diag_partials=True` to provide more efficient graph coloring for the derivatives.

In theory, this also means you can implement Python's `__call__` method for an ODE.
The following code will return a copy of the brachistochrone ODE with the appropriate number of nodes.
Note that the implementation below does not deal with any options provided via the `ode_init_kwargs`.

 ```python
 class CallableBrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def __call__(self, num_nodes, **kwargs):
        from copy import deepcopy
        ret = deepcopy(self)
        ret.options['num_nodes'] = num_nodes
        return ret

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665, desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s',
                        tags=['state_rate_source:x', 'state_units:m'])

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s',
                        tags=['state_rate_source:y', 'state_units:m'])

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['state_rate_source:v', 'state_units:m/s'])

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', tol=1.0E-12)
 ```

 An instance of the above ODE can then be provided to the phase upon instantiation.

```python
ode = CallableBrachistochroneODE(num_nodes=1)
phase = dm.Phase(ode_class=ode, transcription=t)
```

This can potentially lead to unintended behavior if multiple copeis of the ODE are intended to share data.
See [the Python docs](https://docs.python.org/3/library/copy.html) for some of the limitations of deepcopy.

