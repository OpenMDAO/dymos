import numpy as np

import openmdao.api as om

from ..._options import options as dymos_options


class TimeComp(om.ExplicitComponent):
    """
    Class definition of the TimeComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        # Required
        self.options.declare('num_nodes', types=int,
                             desc='The total number of points at which times are required in the'
                                  'phase.')

        self.options.declare('node_ptau', types=(np.ndarray,),
                             desc='The locations of all nodes in non-dimensional phase tau space.')

        self.options.declare('node_dptau_dstau', types=(np.ndarray,),
                             desc='For each node, the ratio of the total phase length to the length'
                                  ' of the nodes containing segment.')

        # Optional
        self.options.declare('units', default=None, allow_none=True, types=str,
                             desc='Units of time (or the integration variable)')

        self.options.declare('initial_val', default=0.0, types=(int, float),
                             desc='default value of initial time')

        self.options.declare('duration_val', default=1.0, types=(int, float),
                             desc='default value of duration')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        time_units = self.options['units']
        num_nodes = self.options['num_nodes']
        node_ptau = self.options['node_ptau']
        node_dptau_dstau = self.options['node_dptau_dstau']

        self.add_input('t_initial', val=self.options['initial_val'], units=time_units)
        self.add_input('t_duration', val=self.options['duration_val'], units=time_units)
        self.add_output('t', units=time_units, val=np.ones(num_nodes))
        self.add_output('t_phase', units=time_units, val=np.ones(num_nodes))
        self.add_output('dt_dstau', units=time_units, val=np.ones(num_nodes))

        # Setup partials
        rs = np.arange(num_nodes)
        cs = np.zeros(num_nodes)

        dtime_dduration = 0.5 * (node_ptau + 1)

        self.declare_partials(of='t', wrt='t_initial', rows=rs, cols=cs, val=1.0)
        self.declare_partials(of='t', wrt='t_duration', rows=rs, cols=cs, val=dtime_dduration)
        self.declare_partials(of='t_phase', wrt='t_duration', rows=rs, cols=cs, val=dtime_dduration)
        self.declare_partials(of='dt_dstau', wrt='t_duration', rows=rs, cols=cs, val=0.5 * node_dptau_dstau)

    def compute(self, inputs, outputs):
        """
        Compute time component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        node_ptau = self.options['node_ptau']
        node_dptau_dstau = self.options['node_dptau_dstau']

        t_initial = inputs['t_initial']
        t_duration = inputs['t_duration']

        outputs['t'][:] = t_initial + 0.5 * (node_ptau + 1) * t_duration
        outputs['t_phase'][:] = 0.5 * (node_ptau + 1) * t_duration
        outputs['dt_dstau'][:] = 0.5 * t_duration * node_dptau_dstau
