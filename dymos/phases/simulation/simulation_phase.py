
from __future__ import print_function, division, absolute_import

import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, OptionsDictionary

from .segment_simulation_comp import SegmentSimulationComp
from ...utils.indexing import get_src_indices_by_row


class SimulationPhase(Group):
    """
    SimulationPhase is an instance that resembles a Phase in structure but is intended for
    use with scipy.solve_ivp to verify the accuracy of the implicit solutions of Dymos.

    This phase is not currently a fully-fledged phase.  It does not support constraints or
    objectives (or anything used by run_driver in general).  It does not accurately compute
    derivatives across the model and should only be used via run_model to verify the accuracy
    of solutions achieved via the other Phase classes.
    """

    def initialize(self):
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('time_options', types=OptionsDictionary)
        self.options.declare('state_options', types=dict)
        self.options.declare('control_options', types=dict)
        self.options.declare('design_parameter_options', types=dict)
        self.options.declare('input_parameter_options', types=dict)
        # self.options.declare('timeseries_outputs', type=dict, default={})

    def _setup_time(self, ivc):
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        num_seg = gd.num_segments
        time_units = self.options['time_options']['units']

        ivc.add_output('time', val=np.ones(nn), units=time_units)
        ivc.add_output('t_initial', val=np.ones(nn), units=time_units)
        ivc.add_output('t_duration', val=np.ones(nn), units=time_units)

        for i in range(num_seg):
            i1, i2 = gd.subset_segment_indices['all'][i, :]
            seg_idxs = gd.subset_node_indices['all'][i1:i2]
            self.connect(src_name='time',
                         tgt_name='segment_{0}.time'.format(i),
                         src_indices=seg_idxs)

    def _setup_states(self, ivc):
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        num_seg = gd.num_segments

        for name, options in iteritems(self.options['state_options']):
            ivc.add_output('implicit_states:{0}'.format(name),
                           val=np.ones((nn,) + options['shape']),
                           units=options['units'])
            size = np.prod(options['shape'])
            for i in range(num_seg):
                if i == 0:
                    src_idxs = np.arange(size, dtype=int)
                    self.connect(src_name='implicit_states:{0}'.format(name),
                                 tgt_name='segment_{0}.initial_states:{1}'.format(i, name),
                                 src_indices=src_idxs, flat_src_indices=True)
                else:
                    src_idxs = np.arange(-size, 0, dtype=int)
                    self.connect(src_name='segment_{0}.states:{1}'.format(i-1, name),
                                 tgt_name='segment_{0}.initial_states:{1}'.format(i, name),
                                 src_indices=src_idxs, flat_src_indices=True)

    def _setup_controls(self, ivc):
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        num_seg = gd.num_segments

        for name, options in iteritems(self.options['control_options']):
            ivc.add_output('implicit_controls:{0}'.format(name),
                           val=np.ones((nn,) + options['shape']),
                           units=options['units'])
            for i in range(num_seg):
                i1, i2 = gd.subset_segment_indices['all'][i, :]
                seg_idxs = gd.subset_node_indices['all'][i1:i2]
                src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
                self.connect(src_name='implicit_controls:{0}'.format(name),
                             tgt_name='segment_{0}.controls:{1}'.format(i, name),
                             src_indices=src_idxs, flat_src_indices=True)

    def _setup_design_parameters(self, ivc):
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        for name, options in iteritems(self.options['design_parameter_options']):
            ivc.add_output('design_parameters:{0}'.format(name),
                           val=np.ones((1,) + options['shape']),
                           units=options['units'])

            for i in range(num_seg):
                self.connect(src_name='design_parameters:{0}'.format(name),
                             tgt_name='segment_{0}.design_parameters:{1}'.format(i, name))

    def _setup_input_parameters(self, ivc):
        for name, options in iteritems(self.options['input_parameter_options']):
            ivc.add_output('input_parameters:{0}'.format(name),
                           val=np.ones((1,) + options['shape']),
                           units=options['units'])

    def setup(self):
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        ivc = self.add_subsystem(name='ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        self._setup_time(ivc)
        self._setup_states(ivc)
        self._setup_controls(ivc)
        self._setup_design_parameters(ivc)
        self._setup_input_parameters(ivc)

        segments_group = self.add_subsystem(name='segments', subsys=Group(),
                                            promotes_outputs=['*'], promotes_inputs=['*'])

        for i in range(num_seg):
            i1, i2 = gd.subset_segment_indices['all'][i, :]
            seg_idxs = gd.subset_node_indices['all'][i1:i2]

            seg_i_comp = SegmentSimulationComp(index=i,
                                               grid_data=self.options['grid_data'],
                                               ode_class=self.options['ode_class'],
                                               ode_init_kwargs=self.options['ode_init_kwargs'],
                                               time_options=self.options['time_options'],
                                               state_options=self.options['state_options'],
                                               control_options=self.options['control_options'],
                                               design_parameter_options=self.options['design_parameter_options'],
                                               input_parameter_options=self.options['input_parameter_options'])

            segments_group.add_subsystem('segment_{0}'.format(i), subsys=seg_i_comp)
