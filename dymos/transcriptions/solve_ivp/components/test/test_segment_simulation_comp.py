import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from dymos.transcriptions.solve_ivp.components.segment_simulation_comp import SegmentSimulationComp
from dymos.transcriptions.runge_kutta.test.rk_test_ode import TestODE
from dymos.phase.options import TimeOptionsDictionary, StateOptionsDictionary
from dymos.transcriptions.grid_data import GridData

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
SegmentSimulationComp = CompWrapperConfig(SegmentSimulationComp)


class TestSegmentSimulationComp(unittest.TestCase):

    def test_simple_integration(self):

        p = om.Problem(model=om.Group())

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'
        time_options['targets'] = 't'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'
        state_options['y']['targets'] = 'y'
        state_options['y']['rate_source'] = 'ydot'

        # Non-standard way to assign state options, so we need this
        state_options['y']['shape'] = (1, )

        gd = GridData(num_segments=4, transcription='gauss-lobatto', transcription_order=3)

        seg0_comp = SegmentSimulationComp(index=0, grid_data=gd, method='RK45',
                                          atol=1.0E-9, rtol=1.0E-9,
                                          ode_class=TestODE, time_options=time_options,
                                          state_options=state_options)

        p.model.add_subsystem('segment_0', subsys=seg0_comp)

        p.setup(check=True)

        p.set_val('segment_0.time', [0, 0.25, 0.5])
        p.set_val('segment_0.initial_states:y', 0.5)

        p.run_model()

        assert_near_equal(p.get_val('segment_0.states:y', units='m')[-1, ...],
                          1.425639364649936,
                          tolerance=1.0E-6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
