import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.explicit_shooting.test.test_rk_integration_comp import SimpleODE
from dymos.transcriptions.explicit_shooting.ode_evaluation_group import ODEEvaluationGroup


class TestODEEvaluationGroup(unittest.TestCase):

    def test_eval(self):
        ode_class = SimpleODE
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        param_options = {'p': dm.phase.options.ParameterOptionsDictionary()}

        param_options['p']['shape'] = (1,)
        param_options['p']['units'] = 's**2'
        param_options['p']['targets'] = ['p']

        control_options = {}
        polynomial_control_options = {}

        p = om.Problem()

        tx = dm.GaussLobatto(num_segments=1, order=3)

        p.model.add_subsystem('ode_eval', ODEEvaluationGroup(ode_class, time_options, state_options,
                                                             param_options, control_options,
                                                             polynomial_control_options,
                                                             grid_data=tx.grid_data,
                                                             ode_init_kwargs=None))
        p.setup(force_alloc_complex=True)

        p.model.ode_eval.set_segment_index(0)
        p.set_val('ode_eval.states:x', [1.25])
        p.set_val('ode_eval.time', [2.2])
        p.set_val('ode_eval.parameters:p', [1.0])

        p.run_model()

        x = p.get_val('ode_eval.states:x')
        t = p.get_val('ode_eval.time')
        xdot_check = x - t**2 + 1

        assert_near_equal(p.get_val('ode_eval.state_rate_collector.state_rates:x_rate'), xdot_check)

        cpd = p.check_partials(compact_print=True, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
