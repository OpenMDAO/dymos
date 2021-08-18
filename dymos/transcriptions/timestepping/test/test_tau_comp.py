import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.timestepping.test.test_euler_integration_comp import SimpleODE
from dymos.transcriptions.timestepping.tau_comp import TauComp


class TestTauComp(unittest.TestCase):

    def test_eval(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=3, transcription='gauss-lobatto',
                                                         transcription_order=3)

        p = om.Problem()
        p.model.add_subsystem('tau_comp', TauComp(grid_data=grid_data, time_units='s'))
        p.setup(force_alloc_complex=True)

        p.set_val('tau_comp.t_initial', 0)
        p.set_val('tau_comp.t_duration', 10)

        t = np.linspace(0, 10, 500)
        ptau = np.zeros_like(t)
        stau = np.zeros_like(t)

        for i, time in enumerate(t):
            p.set_val('tau_comp.time', time)
            p.run_model()
            ptau[i] = p.get_val('tau_comp.ptau')
            stau[i] = p.get_val('tau_comp.stau')


        # x = p.get_val('ode_eval.states:x')
        # t = p.get_val('ode_eval.time')
        # xdot_check = x - t**2 + 1
        #
        # assert_near_equal(p.get_val('ode_eval.state_rate_collector.state_rates:x_rate'), xdot_check)
        #

        p.set_val('tau_comp.time', 1.5)
        p.run_model()
        cpd = p.check_partials(method='cs')
        # assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    grid_data = dm.transcriptions.grid_data.GridData(num_segments=1, transcription='gauss-lobatto',
                                                     transcription_order=3)

    time_options = dm.phase.options.TimeOptionsDictionary()

    time_options['units'] = 's'

    control_options = {'u1': dm.phase.options.ControlOptionsDictionary()}

    control_options['u1']['shape'] = (1,)
    control_options['u1']['units'] = 'rad'

    polynomial_control_options = {}

    p = om.Problem()
    p.model.add_subsystem('interp', ControlInterpolationComp(grid_data=grid_data,
                                                             control_options=control_options,
                                                             time_units='s'))
    p.setup(force_alloc_complex=True)

    p.set_val('interp.segment_index', 0)
    p.set_val('interp.controls:u1', [0.0, 4.0, 4.0])

    x = np.linspace(-1, 1, 20000)
    y = np.zeros_like(x)

    for i, tau in enumerate(x):
        p.set_val('interp.tau', tau)
        p.run_model()
        y[i] = p.get_val('interp.control_values:u1')