import numpy as np
import openmdao.api as om


class LagrangeBarycentricInterpComp(om.ExplicitComponent):
    """
    A segmented component which divides the interpolated region into some number of segments
    num_seg.
    In each of these segments (i) the interpolant is provided for some number of nodes (n_segi)
    """

    def initialize(self):
        self.options.declare('grid_data', recordable=False)
        self.options.declare('time_units', default='s')

    def setup(self):
        self.add_input('t0', val=0.0, units=self.options['time_units'])
        self.add_input('time', val=0.0, units=self.options['time_units'])
        self.add_input('u', val=0.0, units=self.options['time_units'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gd = self.options['grid_data']

        t = inputs['time']
        t_0 = inputs['t_initial']
        t_d = inputs['t_duration']

        # bin the time to the appropriate segment
        tau = -1.0 + 2.0 * (t - t_0) / t_d

        # dtau_dt = 2.0 / t_d
        # dtau_dt0 = - 2.0 / t_d
        # dtau_dtd = -2.0 * (t - t_0) / t_d ** 2

        seg_idx = np.digitize(tau, self.options['segment_ends'])

        # For each control obtain its parameterization on the given segment
        # Number of control discretization nodes per segment
        ncdsps = gd.subset_num_nodes_per_segment['control_disc'][seg_idx]

        # Indices of the control disc nodes belonging to the current segment
        control_disc_seg_idxs = gd.subset_segment_indices['control_disc'][seg_idx]

        # Segment tau values for the control disc nodes in the phase
        control_disc_stau = gd.node_stau[gd.subset_node_indices['control_disc']]

        # Segment tau values for the control disc nodes in the current segment
        control_disc_seg_stau = control_disc_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]]



    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pass


