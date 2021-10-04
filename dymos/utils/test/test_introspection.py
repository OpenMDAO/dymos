import unittest


class TestIntrospection(unittest.TestCase):

    def test_filter_outputs(self):
        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        import openmdao.api as om

        from dymos.utils.introspection import filter_outputs

        p = om.Problem()
        p.model.add_subsystem('ode', MinTimeClimbODE(num_nodes=1))

        p.setup()

        outputs = filter_outputs(['atmos.*', 'aero.*'], p.model.ode)

        expected = 'atmos.drhos_dh atmos.temp atmos.pres atmos.rho atmos.viscosity atmos.sos ' \
                   'aero.mach aero.CD0 aero.kappa aero.CLa aero.CL aero.CD aero.q aero.f_lift ' \
                   'aero.f_drag'

        self.assertSetEqual(set(outputs.keys()), set(expected.split()))
