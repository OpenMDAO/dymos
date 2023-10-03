import numpy as np
import openmdao.api as om

from openmdao.utils.general_utils import ensure_compatible

from ...._options import options as dymos_options


class BirkhoffStateResidComp(om.ImplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']
        self._io_pairs = []

    def add_implicit_output(self, name, resid_input, **kwargs):
        """
        Adds an implicit output with the given name whose resids are given by given input name.

        Parameters
        ----------
        name : str
            The name of the implicit output.
        resid_input : str
            The name of the input providing the residuals for the given output
        kwargs
            Additional keyword arguments for super.add_input and super.add_output.
        """
        val = kwargs['val'] if 'val' in kwargs else 1.0
        shape = kwargs['shape'] if 'shape' in kwargs else None
        val, shape = ensure_compatible(name, value=val, shape=shape)

        self.add_input(resid_input, **kwargs)
        self.add_output(name, **kwargs)
        self._io_pairs.append((name, resid_input))

        size = np.prod(shape, dtype=int)
        ar = np.arange(size, dtype=int)

        self.declare_partials(of=name, wrt=resid_input, rows=ar, cols=ar, val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        for name, resid_input in self._io_pairs:
            residuals[name] = inputs[resid_input]
