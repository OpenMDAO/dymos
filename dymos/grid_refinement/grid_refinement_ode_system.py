import numpy as np
import openmdao.api as om

from ..phase.options import TimeOptionsDictionary
from ..utils.misc import get_rate_units
from ..utils.introspection import get_targets, _get_targets_metadata
from ..utils.ode_utils import _make_ode_system
from ..transcriptions.grid_data import GridData


class GridRefinementODESystem(om.Group):
    """
    Defines a group that performs grid refinement on an ODE.

    The Grid Refinement algorithms in Dymos use the following approach for computing
    errors in the transcription:

    1. The phase segmentation is reproduced, except each segment is of one polynomial order higher
    (for Radau) or two orders higher (for GaussLobatto).
    2. An interpolation matrix (L) and an integration matrix (I) is developed that takes the solution
    from 'all' nodes of the existing solution and interpolates it onto 'all' nodes of the higher-order segmentation.
    For states, this provides a reference set of interpolated values (x_hat).
    3. The ODE is evaluated at all nodes of the new, higher-order grid.  The resulting state-rate
    output values are used to integrate the state values onto 'all' nodes in the higher-order segmentation.
    These states are a new estimate of the state values at each node (x_prime).
    4. The error is computed by comparing x_hat to x_prime.

    This system is used to evaluate the ODE at all nodes within the higher-order segmentation.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare options for this Group.
        """
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

        self.options.declare('time', types=TimeOptionsDictionary,
                             desc='Time options for the phase')

        self.options.declare('states', types=dict,
                             desc='Dictionary of state names/options for the segments parent Phase')

        self.options.declare('controls', default=None, types=dict, allow_none=True,
                             desc='Dictionary of control names/options for the segments parent Phase.')

        self.options.declare('parameters', default=None, types=dict, allow_none=True,
                             desc='Dictionary of parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('ode_class',
                             desc='System defining the ODE')

        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

        self.options.declare('calc_exprs', types=dict, default={},
                             desc='Calculation expressions from the parent phase.')

    def setup(self):
        """
        Add the ODE subsystem.
        """
        grid_data = self.options['grid_data']
        num_nodes = grid_data.num_nodes
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        calc_exprs = self.options['calc_exprs']

        # The ODE System
        if ode_class is not None:
            ode_sys = _make_ode_system(ode_class=ode_class,
                                       num_nodes=num_nodes,
                                       ode_init_kwargs=ode_init_kwargs,
                                       calc_exprs=calc_exprs,
                                       parameter_options=self.options['parameters'])
            self.add_subsystem('ode', subsys=ode_sys)

    def configure(self):
        """
        Promote variables from the ODE.
        """
        grid_data = self.options['grid_data']
        num_nodes = grid_data.num_nodes

        # Configure time
        options = self.options['time']

        # time
        targets = get_targets(self.ode, 'time', options['targets'])
        for tgt in targets:
            self.promotes('ode', inputs=[(tgt, 'time')])
        if targets:
            self.set_input_defaults(name='time', val=np.ones(num_nodes), units=options['units'])

        # time_phase
        targets = get_targets(self.ode, 't_phase', options['time_phase_targets'])
        for tgt in targets:
            self.promotes('ode', inputs=[(tgt, 't_phase')])
        if targets:
            self.set_input_defaults(name='t_phase', val=np.ones(num_nodes), units=options['units'])

        # t_initial
        targets = get_targets(self.ode, 't_initial', options['t_initial_targets'])
        for tgt in targets:
            self.promotes('ode', inputs=[(tgt, 't_initial')])
        if targets:
            self.set_input_defaults(name='t_initial', val=np.ones(num_nodes), units=options['units'])

        # t_duration
        targets = get_targets(self.ode, 't_duration', options['t_duration_targets'])
        for tgt in targets:
            self.promotes('ode', inputs=[(tgt, 't_duration')])
        if targets:
            self.set_input_defaults(name='t_duration', val=np.ones(num_nodes), units=options['units'])

        # Configure the states
        for name, options in self.options['states'].items():
            targets = get_targets(self.ode, name, options['targets'])
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, f'states:{name}')])
            if targets:
                self.set_input_defaults(name=f'states:{name}',
                                        val=np.ones(num_nodes),
                                        units=options['units'])

        # Configure the controls
        for name, options in self.options['controls'].items():
            rate_units = get_rate_units(units=options['units'],
                                        time_units=self.options['time']['units'])
            rate2_units = get_rate_units(units=options['units'],
                                         time_units=self.options['time']['units'],
                                         deriv=2)

            targets = get_targets(self.ode, name, options['targets'])
            if targets:
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, f'controls:{name}')])
                self.set_input_defaults(name=f'controls:{name}',
                                        val=np.ones(num_nodes),
                                        units=options['units'])

            targets = get_targets(self.ode, f'{name}_rate', options['rate_targets'])
            if targets:
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, f'control_rates:{name}_rate')])
                self.set_input_defaults(name=f'control_rates:{name}_rate',
                                        val=np.ones(num_nodes),
                                        units=rate_units)

            targets = get_targets(self.ode, f'{name}_rate2', options['rate2_targets'])
            if targets:
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, f'control_rates:{name}_rate2')])
                self.set_input_defaults(name=f'control_rates:{name}_rate2',
                                        val=np.ones(num_nodes),
                                        units=rate2_units)

        # Configure the parameters
        for name, options in self.options['parameters'].items():
            static_targets = options['static_targets']
            shape = options['shape']
            prom_name = f'parameters:{name}'
            targets = _get_targets_metadata(self.ode, name, options['targets'])
            for tgt, meta in targets.items():
                if tgt in static_targets:
                    self.promotes('ode', inputs=[(tgt, prom_name)])
                else:
                    self.promotes('ode', inputs=[(tgt, prom_name)],
                                  src_indices=om.slicer[np.zeros(num_nodes, dtype=int), ...])
            if targets:
                self.set_input_defaults(name=prom_name,
                                        src_shape=shape,
                                        units=options['units'])
