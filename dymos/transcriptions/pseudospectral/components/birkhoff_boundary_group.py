import numpy as np
import openmdao.api as om

from .birkhoff_collocation_comp import BirkhoffCollocationComp
from .birkhoff_state_resid_comp import BirkhoffStateResidComp

from ...grid_data import GridData
from ....phase.options import TimeOptionsDictionary
from dymos._options import options as dymos_options


class BirkhoffBoundaryMuxComp(om.ExplicitComponent):
    """
    Class definition of the BirtkhoffBoundaryMuxComp.

    This component takes the initial and final values of states
    and muxes them into a single output.

    For a state of a given `shape` in a phase with `num_seg` segments,
    the shape of `initial_states:{state_name}` and `final_states:{state_name}`
    are both `(num_seg,) + shape` and the shape of the resulting
    `states:{state_name}` is `(2,) + shape`.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional phase arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._io_names = {}
        self._no_check_partials = not dymos_options['include_check_partials']

    def configure_io(self, state_options):
        """
        I/O creation is delayed until configure so that we can determine shape and units for the states.

        Parameters
        ----------
        state_options : StateOptionsDictionary
            The phase object to which this transcription instance applies.
        """
        self._io_names = {}
        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape, dtype=int)
            units = options['units']
            self._io_names[state_name] = {'initial': f'initial_states:{state_name}',
                                          'final': f'final_states:{state_name}',
                                          'boundary': state_name}

            iname = self._io_names[state_name]['initial']
            fname = self._io_names[state_name]['final']
            bname = self._io_names[state_name]['boundary']

            self.add_input(iname, shape=shape, units=units)
            self.add_input(fname, shape=shape, units=units)
            self.add_output(bname, shape=(2,) + shape, units=units)

            ar = np.arange(size, dtype=int)

            self.declare_partials(of=bname, wrt=iname, rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=bname, wrt=fname, rows=ar + size, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        for state_name, io_names in self._io_names.items():
            outputs[io_names['boundary']][0] = inputs[io_names['initial']]
            outputs[io_names['boundary']][1] = inputs[io_names['final']]


class BirkhoffBoundaryGroup(om.Group):
    """
    Class definition for the BirkhoffBoundaryEvalGroup.

    This group accepts values for initial and final times, states, controls, and parameters
    and evaluates the ODE with those in order to compute the boundary values and
    objectives.

    Note that in the Birkhoff transcription, the initial and final state values are
    decoupled from the initial and final states in the interpolating polynomial.

    Dymos uses the Birkhoff LGL or CGL approaches so that the control values are provided
    at the endpoints of the phase without the need for extrapolation (unlike the classical
    Radau approach in Dymos)

    Parameters
    ----------
    ode_sys : System
        The OpenMDAO system serving as the ODE.
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def __init__(self, ode_sys, **kwargs):
        super().__init__(**kwargs)
        self._ode_sys = ode_sys

    def setup(self):
        """
        Define the structure of the BirkhoffBoundaryGroup.
        """
        self.add_subsystem('boundary_mux', subsys=BirkhoffBoundaryMuxComp(),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('boundary_ode', subsys=self._ode_sys,
                           promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so that we can determine shape and units for the states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        self._get_subsystem('boundary_mux').configure_io(state_options=phase.state_options)

        for state_name, options in phase.state_options.items():
            for tgt in options['targets']:
                self.promotes('boundary_ode', inputs=[(tgt, state_name)])
