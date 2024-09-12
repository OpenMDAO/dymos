import numpy as np

import openmdao.api as om

from openmdao.utils.general_utils import ensure_compatible

from dymos._options import options as dymos_options
from dymos.utils.misc import om_version


class BirkhoffStateResidComp(om.ImplicitComponent):
    """
    Class definition for the BirkhoffStateResidComp.

    Generates the residuals for any states that are solved for implicitly.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']
        self._io_pairs = []

    def add_residual_from_input(self, name, **kwargs):
        """
        Adds a residual whose value is given by resid_input.

        Parameters
        ----------
        name : str
            The name of the input providing the residuals for the given output.
        **kwargs : dict
            Additional keyword arguments for add_input and add_residual.
        """
        val = kwargs['val'] if 'val' in kwargs else 1.0
        shape = kwargs['shape'] if 'shape' in kwargs else None
        val, shape = ensure_compatible(name, value=val, shape=shape)
        resid_name = 'resid_' + name

        self._io_pairs.append((resid_name, name))

        size = np.prod(shape, dtype=int)
        ar = np.arange(size, dtype=int)

        self.add_input(name, **kwargs)
        self.add_residual(resid_name, **kwargs)

        if om_version()[0] > (3, 31, 1):
            self.declare_partials(of=resid_name, wrt=name, rows=ar, cols=ar, val=1.0)
        else:
            self.declare_partials(of='*', wrt='*', method='fd')

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
        residuals.set_val(inputs.asarray())
