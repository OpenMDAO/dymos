import numpy as np
import openmdao.api as om

from ...options import options as dymos_options


class PathConstraintComp(om.ExplicitComponent):
    """
    Component that computes path constraint values.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self._path_constraints = []
        self._vars = []
        self.options.declare('num_nodes', types=(int,), desc='The number of nodes in the phase '
                                                             'at which the path constraint is to '
                                                             'be evaluated')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        for (name, kwargs) in self._path_constraints:
            self._add_path_constraint_configure(name=name, shape=kwargs['shape'],
                                                units=kwargs['units'], desc=kwargs['desc'],
                                                indices=kwargs['indices'], lower=kwargs['lower'],
                                                upper=kwargs['upper'], equals=kwargs['equals'],
                                                scaler=kwargs['scaler'], adder=kwargs['adder'],
                                                ref0=kwargs['ref0'], ref=kwargs['ref'],
                                                linear=kwargs['linear'])

    def _add_path_constraint_configure(self, name, shape=None, units=None,
                                       desc='', indices=None, lower=None, upper=None, equals=None,
                                       scaler=None, adder=None, ref=None, ref0=None, linear=False):
        """
        Add a path constraint to this component during the configure portion of the setup stack.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        indices : tuple or list or ndarray or None
            Indices that specify which elements in shape are to be path constrained.  If None,
            then the constraint will apply to all values and lower/upper/equals must be scalar
            or of the same shape.  Indices assumes C-order flattening.  For instance, if
            constraining element [0, 1] of a variable with shape [2, 2], indices=[3].
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        desc : str
            description of the variable
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        linear : bool
            True if the *total* derivative of the constrained variable is linear, otherwise False.
        """
        num_nodes = self.options['num_nodes']

        input_name = f'all_values:{name}'
        self.add_input(input_name, shape=(num_nodes,) + shape, units=units, desc=desc)

        output_name = f'path:{name}'
        self.add_output(output_name, shape=(num_nodes,) + shape, units=units, desc=desc)

        # Convert indices from those in one time instance to those in all time instances
        template = np.zeros(np.prod(shape), dtype=int)
        template[indices] = 1
        template = np.tile(template, num_nodes)
        indices = np.nonzero(template)[0]

        self.add_constraint(output_name, lower=lower, upper=upper, equals=equals, ref0=ref0,
                            ref=ref, scaler=scaler, adder=adder, indices=indices, linear=linear)

        self._vars.append((input_name, output_name, shape))

        # Setup partials
        all_shape = (num_nodes,) + shape
        var_size = np.prod(shape)
        all_size = np.prod(all_shape)

        all_row_starts = np.arange(num_nodes, dtype=int) * var_size
        all_rows = []
        for i in all_row_starts:
            all_rows.extend(range(i, i + var_size))
        all_rows = np.asarray(all_rows, dtype=int)

        self.declare_partials(of=output_name, wrt=input_name, rows=all_rows,
                              cols=np.arange(all_size), val=1.0)

    def compute(self, inputs, outputs):
        """
        Compute path constraint outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
