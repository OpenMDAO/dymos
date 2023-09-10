import numpy as np
import openmdao.api as om

from .birkhoff_collocation_comp import BirkhoffCollocationComp
from .birkhoff_state_resid_comp import BirkhoffStateResidComp

from ...grid_data import GridData
from ....phase.options import TimeOptionsDictionary


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
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare group options.
        """
        # self.options.declare('state_options', types=dict,
        #                      desc='Dictionary of options for the states.')
        # self.options.declare('control_options', types=dict,
        #                      desc='Dictionary of options for the controls.')
        # self.options.declare('polynomial_control_options', types=dict,
        #                      desc='Dictionary of options for the polynomial controls.')
        # self.options.declare('parameters', types=dict,
        #                      desc='Dictionary of options for the parameters.')
        # self.options.declare('time_options', types=TimeOptionsDictionary,
        #                      desc='Options for time in the phase.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info.')
        self.options.declare('ode_class', default=None,
                             desc='Callable that instantiates the ODE system.',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={}, recordable=False,
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('initial_boundary_constraints', types=list, recordable=False,
                             desc='Initial boundary constraints from the containing phase.')
        self.options.declare('final_boundary_constraints', types=list, recordable=False,
                             desc='Final boundary constraints from the containing phase.')
        self.options.declare('objectives', types=dict, recordable=False,
                             desc='Objectives from the containing phase.')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        ibcs = self.options['initial_boundary_constraints']
        fbcs = self.options['final_boundary_constraints']
        objs = [meta for meta in self.options['objectives'].values()]

        self.add_subsystem('boundary_ode', subsys=ode_class(num_nodes=2, **ode_init_kwargs),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if any([response['is_expr'] for response in ibcs + fbcs + objs]):
            print([response['is_expr'] for response in ibcs + fbcs + objs])
            self.add_subsystem('boundary_constraint_exec_comp', subsys=om.ExecComp(),
                               promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_io(self, phase):
        grid_data = phase.options['transcription'].grid_data
        nn = grid_data.subset_num_nodes['all']
        # for name, options in phase.state_options.items():
        #     units = options['units']
        #     rate_source = options['rate_source']
        #     shape = options['shape']
        #
        #     for tgt in options['targets']:
        #         self.promotes('boundary_ode', [(tgt, f'states:{name}')],
        #                        src_indices=om.slicer[[0, -1,], ...],
        #                        src_shape=(nn,) + shape,
        #                        flat_src_indices=True)
