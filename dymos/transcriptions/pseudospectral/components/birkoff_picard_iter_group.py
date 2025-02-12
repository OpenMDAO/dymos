import openmdao.api as om

from .birkhoff_picard_update_comp import PicardUpdateComp
from .states_comp import StatesComp

from ...grid_data import GridData


class BirkhoffPicardIterGroup(om.Group):
    """
    Class definition for the BirkhoffIterGroup.

    This group allows for iteration of the state variables and initial _or_ final value of the state
    depending on the direction of the solve.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare group options.
        """
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of options for the states.')
        self.options.declare('time_units', types=str,
                             desc='Units for time in the phase.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info.')
        self.options.declare('ode_class', default=None,
                             desc='Callable that instantiates the ODE system.',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        state_options = self.options['state_options']
        time_units = self.options['time_units']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']

        self.add_subsystem('states_comp',
                           subsys=StatesComp(grid_data=gd, state_options=state_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('ode_all', subsys=ode_class(num_nodes=nn, **ode_init_kwargs))

        self.add_subsystem('picard_update_comp',
                           subsys=PicardUpdateComp(grid_data=gd,
                                                   state_options=state_options,
                                                   time_units=time_units),
                            promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so that we can determine shape and units for the states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        states_comp = self._get_subsystem('states_comp')
        states_comp.configure_io(phase)

        picard_update_comp = self._get_subsystem('picard_update_comp')
        picard_update_comp.configure_io(phase)

        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']

        state_options = self.options['state_options']

        for name, options in state_options.items():
            units = options['units']
            rate_source = options['rate_source']
            shape = options['shape']

            for tgt in options['targets']:
                self.promotes('ode_all', [(tgt, f'state_val:{name}')])
                self.set_input_defaults(f'state_val:{name}', val=1.0, units=units, src_shape=(nn,) + shape)
                # self.connect(f'state_val:{name}', tgt)
            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            # Note the rate source must be shape-compatible with the state
            var_type = phase.classify_var(rate_source_var)

            if var_type == 'ode':
                self.connect(f'ode_all.{rate_source}', f'state_rates:{name}')
