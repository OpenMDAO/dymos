from __future__ import print_function, division

from six import string_types, iteritems

import numpy as np
from scipy.linalg import block_diag

from openmdao.api import ExplicitComponent

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos.utils.lagrange import lagrange_matrices


def _phase_lagrange_matrices(grid_data, t_eval_per_seg, t_initial, t_duration):
    """
    Compute the matrices mapping values at some nodes to values and derivatives at new nodes.

    Parameters
    ----------
    grid_data : GridData
        GridData object representing the grid to be interpolated.
    t_eval_per_seg : dict of {int: np.ndarray}
        Times at which interpolated values and derivatives are to be computed.
    t_initial : float
        Initial phase time.
    t_duration : float
        Phase time duration.

    Returns
    -------
    ndarray[num_eval_set, num_given_set]
        Matrix that yields the values at the new nodes.
    ndarray[num_eval_set, num_given_set]
        Matrix that yields the time derivatives at the new nodes.
    ndarray[num_eval_set, num_given_set]
        Matrix that yields the second time derivatives at the new nodes.

    Notes
    -----
    The values are mapped using the equation:

    .. math::

        x_{eval} = \\left[ L \\right] x_{given}

    And the derivatives are mapped with the equation:

    .. math::

        \\dot{x}_{eval} = \\left[ D \\right] x_{given} \\frac{d \\tau}{dt}

    """
    L_blocks = []
    D_blocks = []
    Daa_blocks = []

    node_ptau = grid_data.node_ptau
    times_all = t_initial + 0.5 * (node_ptau + 1) * t_duration  # times at all nodes
    time_seg_ends = np.reshape(times_all[grid_data.subset_node_indices['segment_ends']],
                               (grid_data.num_segments, 2))

    for iseg in range(grid_data.num_segments):

        #
        # 1. Get the segment tau values of the given nodes.
        #
        i1, i2 = grid_data.subset_segment_indices['all'][iseg, :]
        indices = grid_data.subset_node_indices['all'][i1:i2]
        tau_s_given = grid_data.node_stau[indices]

        #
        # 2. Get the segment tau values of the evaluation nodes.
        #
        t_eval_iseg = t_eval_per_seg[iseg]
        t0_iseg, tf_iseg = time_seg_ends[iseg, :]
        tau_s_eval = 2.0 * (t_eval_iseg - t0_iseg) / (tf_iseg - t0_iseg) - 1

        L_block, D_block = lagrange_matrices(tau_s_given, tau_s_eval)
        _, Daa_block = lagrange_matrices(tau_s_given, tau_s_given)

        L_blocks.append(L_block)
        D_blocks.append(D_block)
        Daa_blocks.append(Daa_block)

    L_ae = block_diag(*L_blocks)
    D_ae = block_diag(*D_blocks)

    D_aa = block_diag(*Daa_blocks)
    D2_ae = np.dot(D_ae, D_aa)

    return L_ae, D_ae, D2_ae


class InterpComp(ExplicitComponent):
    """
    Compute the approximated control values and rates given the values of a control at an arbitrary
    set of points (known a priori) given the values at all nodes.

    Notes
    -----
    .. math::

        u = \\left[ L \\right] u_d

        \\dot{u} = \\frac{d\\tau_s}{dt} \\left[ D \\right] u_d

        \\ddot{u} = \\left( \\frac{d\\tau_s}{dt} \\right)^2 \\left[ D_2 \\right] u_d

    where
    :math:`u_d` are the values of the control at the control discretization nodes,
    :math:`u` are the values of the control at all nodes,
    :math:`\\dot{u}` are the time-derivatives of the control at all nodes,
    :math:`\\ddot{u}` are the second time-derivatives of the control at all nodes,
    :math:`L` is the Lagrange interpolation matrix,
    :math:`D` is the Lagrange differentiation matrix,
    and :math:`\\frac{d\\tau_s}{dt}` is the ratio of segment duration in segment tau space
    [-1 1] to segment duration in time.
    """

    def initialize(self):
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')
        self.options.declare('t_eval_per_seg', types=dict,
                             desc='Times within each segment at which interpolation is desired')
        self.options.declare('t_initial', types=(float, np.ndarray),
                             desc='Initial time of the phase.')
        self.options.declare('t_duration', types=(float, np.ndarray),
                             desc='Time duration of the phase.')

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def _setup_controls(self):
        control_options = self.options['control_options']
        num_nodes = self.options['grid_data'].subset_num_nodes['all']
        num_output_points = sum([len(a) for a in list(self.options['t_eval_per_seg'].values())])
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
        time_units = self.options['time_units']

        for name, options in iteritems(control_options):
            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            input_shape = (num_nodes,) + shape
            output_shape = (num_output_points,) + shape

            units = options['units']
            rate_units = get_rate_units(units, time_units)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=output_shape,
                            units=rate2_units)

    def setup(self):
        num_nodes = self.options['grid_data'].num_nodes
        num_seg = self.options['grid_data'].num_segments
        gd = self.options['grid_data']

        # Find the dt_dstau for each point in t_eval
        node_ptau = gd.node_ptau
        times_all = self.options['t_initial'] + 0.5 * (node_ptau + 1) * self.options['t_duration']
        time_seg_ends = np.reshape(times_all[gd.subset_node_indices['segment_ends']],
                                   (gd.num_segments, 2))

        dt_dstau = []
        for i in range(num_seg):
            t_eval_seg_i = self.options['t_eval_per_seg'][i]
            num_points_seg_i = len(t_eval_seg_i)
            t0_seg_i = time_seg_ends[i, 0]
            tf_seg_i = time_seg_ends[i, 1]
            dt_dstau.extend(num_points_seg_i * [(0.5 * (tf_seg_i - t0_seg_i)).tolist()])
        self.dt_dstau = np.array(dt_dstau)

        self.sizes = {}
        self.num_nodes = num_nodes

        self.L, self.D, self.D2 = _phase_lagrange_matrices(gd,
                                                           self.options['t_eval_per_seg'],
                                                           self.options['t_initial'],
                                                           self.options['t_duration'])

        self._setup_controls()

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        control_options = self.options['control_options']

        for name, options in iteritems(control_options):

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T
            b = np.tensordot(self.D2, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_val_names[name]] = np.tensordot(self.L, u, axes=(1, 0))
            outputs[self._output_rate_names[name]] = (a / self.dt_dstau).T
            outputs[self._output_rate2_names[name]] = (b / self.dt_dstau ** 2).T
