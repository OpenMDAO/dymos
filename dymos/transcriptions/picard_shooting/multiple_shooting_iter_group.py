import openmdao.api as om

from ..grid_data import GridData
from .birkhoff_picard_iter_group import BirkhoffPicardIterGroup
from .multiple_shooting_update_comp import MultipleShootingUpdateComp


class MultipleShootingIterGroup(om.Group):
    """
    Class definition for the MultipleShootingIterGroup.

    This group allows for iteration of the initial or final state variable values in each segment
    (in forward shooting or backward shooting, respectively). This group feeds back new guesses
    for these "starting" state values based on the results of integration the causal segment

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self.options.declare('ode_nonlinear_solver', default=om.NonlinearBlockGS(maxiter=100, use_aitken=True, iprint=0),
                             desc='Nonlinear solver used to resolve Picard iteration.', recordable=False)
        self.options.declare('ode_linear_solver', default=om.DirectSolver(),
                             desc='Linear solver used to linearize the picard iteration subsystem.', recordable=False)
        self.options.declare('ms_nonlinear_solver', default=om.NonlinearBlockGS(maxiter=100, use_aitken=True, iprint=0),
                             desc='Nonlinear solver used to resolve Picard iteration.', recordable=False)
        self.options.declare('ms_linear_solver', default=om.DirectSolver(),
                             desc='Linear solver used to linearize the picard iteration subsystem.', recordable=False)
        self.options.declare('calc_exprs', types=dict, default={},
                             desc='phase calculation expressions.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='phase parameter options')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        state_options = self.options['state_options']
        time_units = self.options['time_units']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        calc_exprs = self.options['calc_exprs']
        parameter_options = self.options['parameter_options']

        seg_prop_group = self.add_subsystem('segment_prop_group',
                                            subsys=BirkhoffPicardIterGroup(grid_data=gd,
                                                                           state_options=state_options,
                                                                           time_units=time_units,
                                                                           ode_class=ode_class,
                                                                           ode_init_kwargs=ode_init_kwargs,
                                                                           calc_exprs=calc_exprs,
                                                                           parameter_options=parameter_options),
                                            promotes_inputs=['*'], promotes_outputs=['*'])

        seg_prop_group.nonlinear_solver = self.options['ode_nonlinear_solver']
        seg_prop_group.linear_solver = self.options['ode_linear_solver']

        self.nonlinear_solver = self.options['ms_nonlinear_solver']
        self.linear_solver = self.options['ms_linear_solver']

        self.add_subsystem('ms_update_comp',
                           MultipleShootingUpdateComp(grid_data=gd,
                                                      state_options=state_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so that we can determine shape and units for the states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        state_options = self.options['state_options']

        segment_prop_group = self._get_subsystem('segment_prop_group')
        segment_prop_group.configure_io(phase)

        ms_update_comp = self._get_subsystem('ms_update_comp')
        ms_update_comp.configure_io(phase)

        for state_name, options in state_options.items():
            if options['solve_segments'] == 'forward':
                self.connect(f'seg_initial_states:{state_name}',
                             f'picard_update_comp.seg_initial_states:{state_name}')
            else:
                self.connect(f'seg_final_states:{state_name}',
                             f'picard_update_comp.seg_final_states:{state_name}')
