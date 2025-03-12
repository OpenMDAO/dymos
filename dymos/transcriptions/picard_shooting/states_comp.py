"""Definition of the States passthru component."""
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from dymos.transcriptions.grid_data import GridData
from dymos._options import options as dymos_options


class StatesComp(ExplicitComponent):
    """
    This component provides inputs for states and outputs their values during Picard iteration.

    This component serves to accept variables 'states:{state_name}' and echo them out as
    'state_val:{state_name}'.  This is necessary when using NonlinearBlockGS to converge
    the states.

    Parameters
    ----------
    **kwargs : dict
        Arguments to be passed to the component initialization method.

    Attributes
    ----------
    _vars : dict
        Container mapping name of variables to be muxed with additional data.
    _input_names : dict
        Container mapping name of variables to be muxed with associated inputs.
    """

    def __init__(self, **kwargs):
        """
        Instantiate MuxComp and populate private members.
        """
        super().__init__(**kwargs)

        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so we can determine shape and units.

        Parameters
        ----------
        phase : Phase
            The phase object that contains this collocation comp.
        """
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']
        state_options = self.options['state_options']

        self.var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'current_state': f'state_val:{state_name}',
                'next_state': f'states:{state_name}'
            }

        ar = np.arange(num_nodes, dtype=int)
        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['next_state'],
                shape=(num_nodes,) + shape,
                desc=f'Value of state {state_name} available as an input.',
                units=units)

            self.add_output(
                name=var_names['current_state'],
                shape=(num_nodes,) + shape,
                desc=f'Value of the state {state_name} to be used in the ODE evaluation.',
                units=units
            )

            self.declare_partials(of=var_names['current_state'],
                                  wrt=var_names['next_state'],
                                  rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Mux the inputs into the appropriate outputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : Vector
            Discrete input variables read via discrete_inputs[key].
        discrete_outputs : Vector
            Discrete output variables read via discrete_outputs[key].
        """
        outputs.set_val(inputs.asarray())
