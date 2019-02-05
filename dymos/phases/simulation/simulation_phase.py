
from __future__ import print_function, division, absolute_import

from collections import Sequence

import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, OptionsDictionary

from ...utils.indexing import get_src_indices_by_row
from ...utils.misc import get_rate_units
from .segment_simulation_comp import SegmentSimulationComp
from .simulation_state_mux_comp import SimulationStateMuxComp
from .interp_comp import InterpComp
from .simulation_timeseries_comp import SimulationTimeseriesOutputComp


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
        self.options.declare('times', types=(Sequence, np.ndarray, int, str),
                             desc='number of times to include in timeseries output or values of'
                                  'time for timeseries output')
        self.options.declare('t_initial', desc='initial time of the phase')
        self.options.declare('t_duration', desc='time duration of the phase')
        self.options.declare('timeseries_outputs', types=dict, default={})

    def _setup_time(self, ivc):
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        time_units = self.options['time_options']['units']

        # Figure out the times at which each segment needs to output data.
        t_initial = self.options['t_initial']
        t_duration = self.options['t_duration']
        node_ptau = gd.node_ptau
        times_all = t_initial + 0.5 * (node_ptau + 1) * t_duration  # times at all nodes

        if isinstance(self.options['times'], int):
            times = np.linspace(t_initial, t_initial + t_duration, self.options['times'])
        elif isinstance(self.options['times'], str):
            times = np.unique(times_all[gd.subset_node_indices[self.options['times']]])
        else:
            times = self.options['times']
            if np.any(times < t_initial) or np.any(times > t_initial + t_duration):
                raise ValueError('simulation times contains one or more values outside of the '
                                 'range [t_initial, t_initial + t_duration]')

        # Now we have the times, bin them into their segments
        times_seg_ends = np.concatenate((times_all[gd.subset_node_indices['segment_ends']][::2],
                                         [times_all[-1]]))
        segment_for_times = np.clip(np.digitize(times, times_seg_ends) - 1, 0, num_seg - 1)

        # Now for each segment we can get t_eval
        self.t_eval_per_seg = dict([(i, times[np.where(segment_for_times == i)[0]])
                                    for i in range(num_seg)])

        # Finally, make sure t_eval_per_seg contains the segment endpoints.
        time_seg_ends = np.reshape(times_all[gd.subset_node_indices['segment_ends']], (num_seg, 2))
        for i in range(num_seg):
            self.t_eval_per_seg[i] = np.unique(np.concatenate((self.t_eval_per_seg[i],
                                                               time_seg_ends[i, :])))

        time_vals = np.concatenate(list(self.t_eval_per_seg.values()))

        ivc.add_output('time',
                       val=time_vals,
                       units=time_units)

        ivc.add_output('time_phase',
                       val=time_vals - time_vals[0],
                       units=time_units)

        if self.options['time_options']['targets']:
            self.connect('time', ['ode.{0}'.format(tgt) for tgt in self.options['time_options']['targets']])

    def _setup_states(self, ivc):
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        for name, options in iteritems(self.options['state_options']):
            ivc.add_output('initial_states:{0}'.format(name),
                           val=np.ones(options['shape']),
                           units=options['units'])

            size = np.prod(options['shape'])

            for i in range(num_seg):
                if i == 0:
                    self.connect(src_name='initial_states:{0}'.format(name),
                                 tgt_name='segment_{0}.initial_states:{1}'.format(i, name))
                else:
                    src_idxs = np.arange(-size, 0, dtype=int)
                    self.connect(src_name='segment_{0}.states:{1}'.format(i-1, name),
                                 tgt_name='segment_{0}.initial_states:{1}'.format(i, name),
                                 src_indices=src_idxs, flat_src_indices=True)

                self.connect(src_name='segment_{0}.states:{1}'.format(i, name),
                             tgt_name='state_mux_comp.segment_{0}_states:{1}'.format(i, name))

            if options['targets']:
                self.connect(src_name='state_mux_comp.states:{0}'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']])

    def _setup_controls(self, ivc):
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        num_seg = gd.num_segments

        for name, options in iteritems(self.options['control_options']):
            ivc.add_output('implicit_controls:{0}'.format(name),
                           val=np.ones((nn,) + options['shape']),
                           units=options['units'])

            for i in range(num_seg):
                i1, i2 = gd.subset_segment_indices['control_disc'][i, :]
                seg_idxs = gd.subset_node_indices['control_disc'][i1:i2]
                src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
                self.connect(src_name='implicit_controls:{0}'.format(name),
                             tgt_name='segment_{0}.controls:{1}'.format(i, name),
                             src_indices=src_idxs, flat_src_indices=True)

            # connect the control to the interpolator
            self.connect(src_name='implicit_controls:{0}'.format(name),
                         tgt_name='interp_comp.controls:{0}'.format(name))

            if options['targets']:
                self.connect(src_name='interp_comp.control_values:{0}'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']])

            if options['rate_param']:
                self.connect(src_name='interp_comp.control_rates:{0}_rate'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['rate_param']])

            if options['rate2_param']:
                self.connect(src_name='interp_comp.control_rates:{0}_rate2'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['rate2_param']])

            # if options['targets']:
            #     self.connect('time', ['ode.{0}'.format(tgt) for tgt in options['targets']])

    def _setup_design_parameters(self, ivc):
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_points = sum([len(a) for a in list(self.t_eval_per_seg.values())])

        for name, options in iteritems(self.options['design_parameter_options']):
            ivc.add_output('design_parameters:{0}'.format(name),
                           val=np.ones((1,) + options['shape']),
                           units=options['units'])

            for i in range(num_seg):
                self.connect(src_name='design_parameters:{0}'.format(name),
                             tgt_name='segment_{0}.design_parameters:{1}'.format(i, name))

            if options['targets']:
                self.connect(src_name='design_parameters:{0}'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']],
                             src_indices=np.zeros(num_points, dtype=int))

    def _setup_input_parameters(self, ivc):
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_points = sum([len(a) for a in list(self.t_eval_per_seg.values())])

        for name, options in iteritems(self.options['input_parameter_options']):
            ivc.add_output('input_parameters:{0}'.format(name),
                           val=np.ones((1,) + options['shape']),
                           units=options['units'])

            for i in range(num_seg):
                self.connect(src_name='input_parameters:{0}'.format(name),
                             tgt_name='segment_{0}.input_parameters:{1}'.format(i, name))

            if options['targets']:
                self.connect(src_name='input_parameters:{0}'.format(name),
                             tgt_name=['ode.{0}'.format(tgt) for tgt in options['targets']],
                             src_indices=np.zeros(num_points, dtype=int))

    def _setup_segments(self):
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        segments_group = self.add_subsystem(name='segments', subsys=Group(),
                                            promotes_outputs=['*'], promotes_inputs=['*'])

        for i in range(num_seg):
            seg_i_comp = SegmentSimulationComp(index=i,
                                               grid_data=self.options['grid_data'],
                                               ode_class=self.options['ode_class'],
                                               ode_init_kwargs=self.options['ode_init_kwargs'],
                                               time_options=self.options['time_options'],
                                               state_options=self.options['state_options'],
                                               control_options=self.options['control_options'],
                                               design_parameter_options=self.options['design_parameter_options'],
                                               input_parameter_options=self.options['input_parameter_options'],
                                               t_eval=self.t_eval_per_seg[i])
            segments_group.add_subsystem('segment_{0}'.format(i), subsys=seg_i_comp)

    def _setup_ode(self):
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        nn = sum([len(self.t_eval_per_seg[i]) for i in range(num_seg)])

        ode = self.options['ode_class'](num_nodes=nn, **self.options['ode_init_kwargs'])
        self.add_subsystem(name='ode', subsys=ode)

    def _setup_timeseries_outputs(self):

        gd = self.options['grid_data']
        time_units = self.options['time_options']['units']
        num_points = sum([len(a) for a in list(self.t_eval_per_seg.values())])
        timeseries_comp = SimulationTimeseriesOutputComp(grid_data=gd, num_times=num_points)
        self.add_subsystem('timeseries', subsys=timeseries_comp)

        timeseries_comp._add_timeseries_output('time',
                                               var_class='time',
                                               units=time_units)
        self.connect(src_name='time', tgt_name='timeseries.all_values:time')

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class='time_phase',
                                               units=time_units)
        self.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')

        for name, options in iteritems(self.options['state_options']):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class='state',
                                                   units=options['units'],
                                                   shape=options['shape'])
            self.connect(src_name='state_mux_comp.states:{0}'.format(name),
                         tgt_name='timeseries.all_values:states:{0}'.format(name))

        for name, options in iteritems(self.options['control_options']):
            control_units = options['units']

            # Control values
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class='control',
                                                   units=control_units)
            self.connect(src_name='interp_comp.control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class='control_rate',
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='interp_comp.control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.all_values:control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class='control_rate2',
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='interp_comp.control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.all_values:control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(self.options['design_parameter_options']):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class='design_parameter',
                                                   units=units)

            if self.options['ode_class'].ode_options._parameters[name]['dynamic']:
                src_idxs_raw = np.zeros(num_points, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='design_parameters:{0}'.format(name),
                         tgt_name='timeseries.all_values:design_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.options['input_parameter_options']):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class='input_parameter',
                                                   units=units)

            if self.options['ode_class'].ode_options._parameters[name]['dynamic']:
                src_idxs_raw = np.zeros(num_points, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='input_parameters:{0}'.format(name),
                         tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(self.options['timeseries_outputs']):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = 'ode'

            # Failed to find variable, assume it is in the RHS
            self.connect(src_name='ode.{0}'.format(var),
                         tgt_name='timeseries.all_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def setup(self):

        ivc = self.add_subsystem(name='ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        self._setup_time(ivc)
        self._setup_states(ivc)
        self._setup_controls(ivc)
        self._setup_design_parameters(ivc)
        self._setup_input_parameters(ivc)
        self._setup_segments()

        self.add_subsystem('state_mux_comp',
                           SimulationStateMuxComp(grid_data=self.options['grid_data'],
                                                  times_per_seg=self.t_eval_per_seg,
                                                  state_options=self.options['state_options']))

        self.add_subsystem('interp_comp',
                           InterpComp(control_options=self.options['control_options'],
                                      time_units=self.options['time_options']['units'],
                                      grid_data=self.options['grid_data'],
                                      t_eval_per_seg=self.t_eval_per_seg,
                                      t_initial=self.options['t_initial'],
                                      t_duration=self.options['t_duration']))

        self._setup_ode()

        self._setup_timeseries_outputs()
