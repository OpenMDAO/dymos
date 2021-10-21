import numpy as np

import openmdao.api as om

from ...options import options as dymos_options


class TauComp(om.ExplicitComponent):
    """
    Component that computes the phase tau, segment tau, and current segment index based on time.

    Note that stau is differentiable within a segment but non-differentiable at the segment bounds.

    Parameters
    ----------
    grid_data : GridData
        The GridData object which provides the dicretization of the Phase.
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, grid_data=None, **kwargs):
        super().__init__(**kwargs)
        self._grid_data = grid_data
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('vec_size', types=int, default=1, desc='number of nodes at which to compute time')
        self.options.declare('segment_index', types=int, desc='index of the current segment')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time (or the integration variable)')

    def setup(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        vec_size = self.options['vec_size']
        time_units = self.options['time_units']

        self.add_input('t_initial', val=0.0, units=time_units)
        self.add_input('t_duration', val=1.0, units=time_units)
        self.add_input('time', shape=(vec_size,), units=time_units)
        self.add_output('ptau', units=None, shape=(vec_size,))
        self.add_output('stau', units=None, shape=(vec_size,))
        self.add_output('dstau_dt', units=f'1/{time_units}', val=1.0)
        self.add_output('time_phase', units=time_units, shape=(vec_size,))
        # self.add_discrete_output('segment_index', val=0)

        # Setup partials
        ar = np.arange(vec_size, dtype=int)
        self.declare_partials(of='ptau', wrt='t_initial')
        self.declare_partials(of='ptau', wrt='t_duration')
        self.declare_partials(of='ptau', wrt='time', rows=ar, cols=ar)

        self.declare_partials(of='stau', wrt='t_initial')
        self.declare_partials(of='stau', wrt='t_duration')
        self.declare_partials(of='stau', wrt='time', rows=ar, cols=ar)

        self.declare_partials(of='time_phase', wrt='time', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='time_phase', wrt='t_initial', val=-1.0)

        self.declare_partials(of='dstau_dt', wrt='t_duration', val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute time component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete outputs.
        """
        gd = self._grid_data
        seg_idx = self.options['segment_index']

        time = inputs['time']
        t_initial = inputs['t_initial'].copy()
        t_duration = inputs['t_duration'].copy()

        outputs['ptau'] = ptau = 2.0 * (time - t_initial) / t_duration - 1.0

        ptau0_seg = gd.segment_ends[seg_idx]
        ptauf_seg = gd.segment_ends[seg_idx + 1]

        td_seg = ptauf_seg - ptau0_seg

        outputs['stau'] = 2.0 * (ptau - ptau0_seg) / td_seg - 1.0
        outputs['dstau_dt'] = 4 / (t_duration * td_seg)
        outputs['time_phase'] = time - t_initial

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        gd = self._grid_data
        seg_idx = self.options['segment_index']

        time = inputs['time']
        t_initial = inputs['t_initial']
        t_duration = inputs['t_duration']

        ptau0_seg = gd.segment_ends[seg_idx]
        ptauf_seg = gd.segment_ends[seg_idx + 1]

        td_seg = ptauf_seg - ptau0_seg

        partials['ptau', 'time'] = 2.0 / t_duration
        partials['ptau', 't_initial'] = -2.0 / t_duration
        partials['ptau', 't_duration'] = -2.0 * (time - t_initial) / (t_duration ** 2)

        dstau_dptau = 2.0 / td_seg

        # Note that these derivatives ignore the effect of changing segments.
        # The derivatives of stau are discontinuous at segment boundaries.
        partials['stau', 'time'] = dstau_dptau * partials['ptau', 'time']
        partials['stau', 't_initial'] = dstau_dptau * partials['ptau', 't_initial']
        partials['stau', 't_duration'] = dstau_dptau * partials['ptau', 't_duration']

        partials['dstau_dt', 't_duration'] = -4 / (t_duration**2 * td_seg)
