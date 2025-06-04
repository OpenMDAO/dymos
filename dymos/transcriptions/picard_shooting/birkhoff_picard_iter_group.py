import openmdao.api as om

from .birkhoff_picard_update_comp import PicardUpdateComp
from .states_comp import StatesComp

from ..grid_data import GridData
from ...utils.ode_utils import _make_ode_system


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._implicit_outputs = set()

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
        self.options.declare('calc_exprs', types=dict, default={},
                             desc='phase calculation expressions.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='phase parameter options')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        self.add_subsystem('states_comp',
                           subsys=StatesComp(grid_data=gd, state_options=state_options))

        ode = _make_ode_system(ode_class=self.options['ode_class'],
                               num_nodes=nn,
                               calc_exprs=self.options['calc_exprs'],
                               ode_init_kwargs=self.options['ode_init_kwargs'],
                               parameter_options=self.options['parameter_options'])

        self.add_subsystem('ode_all', subsys=ode)

        self.add_subsystem('picard_update_comp',
                           subsys=PicardUpdateComp(grid_data=gd,
                                                   state_options=state_options,
                                                   time_units=time_units))

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

        self.promotes('states_comp', inputs=['*'], outputs=['*'])

        picard_update_comp = self._get_subsystem('picard_update_comp')
        picard_update_comp.configure_io(phase)

        state_options = self.options['state_options']

        for name, options in state_options.items():
            rate_source = options['rate_source']

            for tgt in options['targets']:
                self.connect(f'state_val:{name}', f'ode_all.{tgt}')

            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            # Note the rate source must be shape-compatible with the state
            rate_src_type = phase.classify_var(rate_source_var)

            if rate_src_type == 'ode':
                self.connect(f'ode_all.{rate_source}', f'picard_update_comp.f_computed:{name}')
            elif rate_src_type == 'state':
                self.connect(f'state_val:{rate_source}', f'picard_update_comp.f_computed:{name}')

            promotes = [f'states:{name}']
            if options['solve_segments'] == 'forward':
                promotes += [f'final_states:{name}']
            else:
                promotes += [f'initial_states:{name}']

            self.promotes('picard_update_comp', any=promotes)
