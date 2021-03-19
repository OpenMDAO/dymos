import numpy as np

import openmdao.api as om

from ..options import options as dymos_options


class PhaseLinkageComp(om.ExplicitComponent):
    """
    Component that provides a constraint between end values in two connected phases.

    Provides a 'linkage' capability between two phases to provide
    continuity in states, time, and other variables between two
    phases.

    Conceptually, each linkage can be thought of as a set of compatibility constraints involving
    one or more variables.

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
        self.options.declare('linkages', default=[])
        self._io_names = {}

    def add_linkage_configure(self, lnk):
        """
        Add a linkage constraint to be managed by this component.

        Each constraint equation consists of a variable in the first phase (var_a) at a given
        location in the first phase (loc_a - either 'initial' or 'final') and a variable in the
        second phase (var_b), with corresponding location (loc_b).

        The resulting linkage equation of constraint is:

        C_linkage = sign_a * vars_a + sign_b * vars_b

        Constraining this linkage to a value of zero, with the default signs
        (sign_a = 1, sign_b = -1) will result in the two variables having the same value at the
        given locations.

        Parameters
        ----------
        lnk : LinkageOptionsDictionary
            The linkage options dictionary defining the given linkage constraint.
        """
        if lnk['connected']:
            return

        self.options['linkages'].append(lnk)

        phase_a = lnk['phase_a']
        phase_b = lnk['phase_b']
        shape = lnk['shape']
        units = lnk['units']

        var_a = lnk['constraint_name'] if lnk['constraint_name'] else lnk['var_a'].split('.')[-1]
        var_b = lnk['constraint_name'] if lnk['constraint_name'] else lnk['var_b'].split('.')[-1]

        loc_a = lnk['loc_a']
        loc_b = lnk['loc_b']

        input_a = f'{phase_a}:{var_a}'
        input_b = f'{phase_b}:{var_b}'
        output = f'{phase_a}:{var_a}_{loc_a}|{phase_b}:{var_b}_{loc_b}'
        ishape = (2,) + shape

        lnk._input_a = input_a
        lnk._input_b = input_b
        lnk._idxs_a = (0, ...) if loc_a == 'initial' else (-1, ...)
        lnk._idxs_b = (0, ...) if loc_b == 'initial' else (-1, ...)
        lnk._output = output

        try:
            self.add_input(name=input_a, shape=ishape, val=np.zeros(ishape), units=units)
        except ValueError as e:
            pass

        try:
            self.add_input(name=input_b, shape=ishape, val=np.zeros(ishape), units=units)
        except ValueError as e:
            pass

        self.add_output(name=output, shape=shape, val=np.zeros(shape), units=units)

        if lnk['equals'] is None and lnk['lower'] is None and lnk['upper'] is None:
            lnk['equals'] = 0.0

        self.add_constraint(name=output, equals=lnk['equals'], lower=lnk['lower'],
                            upper=lnk['upper'], ref=lnk['ref'], ref0=lnk['ref0'],
                            scaler=lnk['scaler'], adder=lnk['adder'], linear=lnk['linear'])

        size = np.prod(lnk['shape'])
        rs = np.arange(size)
        cs_a = rs if loc_a == 'initial' else size + rs
        cs_b = rs if loc_b == 'initial' else size + rs

        self.declare_partials(of=output, wrt=input_a, rows=rs, cols=cs_a, val=lnk['sign_a'])
        self.declare_partials(of=output, wrt=input_b, rows=rs, cols=cs_b, val=lnk['sign_b'])

    def compute(self, inputs, outputs):
        """
        Compute the linkage constraint values.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        for lnk in self.options['linkages']:
            input_a = lnk._input_a
            input_b = lnk._input_b
            idxs_a = lnk._idxs_a
            idxs_b = lnk._idxs_b
            output = lnk._output

            outputs[output] = lnk['sign_a'] * inputs[input_a][idxs_a] + lnk['sign_b'] * inputs[input_b][idxs_b]
