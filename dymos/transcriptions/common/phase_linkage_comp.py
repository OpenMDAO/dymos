import numpy as np
import openmdao.api as om


class PhaseLinkageComp(om.ExplicitComponent):
    """
    Provides a 'linkage' capability between two phases to provide
    continuity in states, time, and other variables between two
    phases.

    Conceptually, each linkage can be thought of as a set of compatibility constraints involving
    one or more variables.
    """
    def initialize(self):
        self.options.declare('linkages', default=[])

    def setup(self):

        for lnk in self.options['linkages']:
            self.add_input(name=lnk['cond0_name'], shape=lnk['shape'],
                           val=np.zeros(lnk['shape']), units=lnk['units'])

            self.add_input(name=lnk['cond1_name'], shape=lnk['shape'],
                           val=np.zeros(lnk['shape']), units=lnk['units'])

            self.add_output(name=lnk['name'], shape=lnk['shape'],
                            val=np.zeros(lnk['shape']), units=lnk['units'])

            self.add_constraint(name=lnk['name'], equals=lnk['equals'],
                                lower=lnk['lower'], upper=lnk['upper'],
                                ref=lnk['ref'], ref0=lnk['ref0'],
                                scaler=lnk['scaler'], adder=lnk['adder'],
                                linear=lnk['linear'])

            shape = lnk['shape']
            ar = np.arange(np.prod(shape))

            self.declare_partials(of=lnk['name'], wrt=lnk['cond1_name'], rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=lnk['name'], wrt=lnk['cond0_name'], rows=ar, cols=ar, val=-1.0)

    def add_linkage(self, name, vars, shape=(1,), equals=None, lower=None, upper=None, units=None,
                    scaler=None, adder=None, ref0=None, ref=None, linear=False):
        """
        Add a linkage constraint to be managed by this component.

        .. math ::

            C_n = y_{n1} - y_{n0}

        where :math:`y_1` is the value of the variable at the beginning or end of phase 1,
        and :math:`y_0` is the value of the variable at the beginning or end of phase 0.
        The location of the source of the constraint can be set by the user based on
        connected indices.

        The name of each linkage constraint will be LNK_var where LNK is the name of the linkage
        and var are the vars in that linkage.

        Parameters
        ----------
        name : str.
            The name of one or more linkage constraints to be added.
        vars : str or iterable
            The name of one or more linked variables to be added.
        shape : tuple or dict
            The shape of the constraint being formed.  Must be compliant with the shape
            of the variable.  If given as a dict, it should be keyed
            with variables in var, and the associated value being the corresponding units.
        units : str, dict, or None
            The units of the linkage constraint.  If given as a string, the units will
            apply to each variable in vars.  If given as a dict, it should be keyed
            with variables in var, and the associated value being the corresponding units.
            Default is None.
        lower : float or ndarray
            The minimum allowable difference of y_1 - y_0, enforced by the optimizer.
        upper : float or ndarray
            The minimum allowable difference of y_1 - y_0, enforced by the optimizer.
        equals : float or ndarray
            The prescribed difference of y_1 - y_0, enforced bt the optimizer.
        scaler : float, ndarray, or None
            The scalar applied to this constraint by the driver.
        adder : float, ndarray, or None
            The adder applied to this constraint by the driver.
        ref0 : float, ndarray, or None
            The zero-reference value of this constraint, used for scaling by the driver.
        ref : float, ndarray, or None
            The one-reference value of this constraint, used for scaling by the driver.
        linear : bool
            If True, this constraint will be treated as a linear constraint by the optimizer.
            This should only be done if the *total derivative* of the constraint is linear.
            That is, the affected variables in each phase are design
            variables or linear functions of design variables.  Default is False.
        """
        if equals is None and lower is None and upper is None:
            equals = 0.0

        if isinstance(vars, str):
            _vars = (vars,)
        else:
            _vars = vars

        if isinstance(units, str) or units is None:
            _units = {}
            for var in _vars:
                _units[var] = units
        else:
            _units = units

        if isinstance(shape, tuple):
            _shapes = {}
            for var in _vars:
                _shapes[var] = shape
        else:
            _shapes = shape

        for var in _vars:

            lnk = om.OptionsDictionary()

            lnk.declare('name', types=(str,))
            lnk.declare('equals', types=(float, np.ndarray), allow_none=True)
            lnk.declare('lower', types=(float, np.ndarray), allow_none=True)
            lnk.declare('upper', types=(float, np.ndarray), allow_none=True)
            lnk.declare('units', types=str, allow_none=True)
            lnk.declare('scaler', types=(float, np.ndarray), allow_none=True)
            lnk.declare('adder', types=(float, np.ndarray), allow_none=True)
            lnk.declare('ref0', types=(float, np.ndarray), allow_none=True)
            lnk.declare('ref', types=(float, np.ndarray), allow_none=True)
            lnk.declare('linear', types=bool)
            lnk.declare('shape', types=tuple)
            lnk.declare('cond0_name', types=str)
            lnk.declare('cond1_name', types=str)

            lnk['name'] = '{0}_{1}'.format(name, var)
            lnk['equals'] = equals
            lnk['lower'] = lower
            lnk['upper'] = upper
            lnk['scaler'] = scaler
            lnk['adder'] = adder
            lnk['ref0'] = ref0
            lnk['ref'] = ref
            lnk['shape'] = _shapes.get(var, (1,))
            lnk['linear'] = linear
            lnk['units'] = _units.get(var, None)
            lnk['cond0_name'] = '{0}:lhs'.format(lnk['name'])
            lnk['cond1_name'] = '{0}:rhs'.format(lnk['name'])

            self.options['linkages'].append(lnk)

    def compute(self, inputs, outputs):

        for lnk in self.options['linkages']:
            outputs[lnk['name']] = inputs[lnk['cond1_name']] - inputs[lnk['cond0_name']]
