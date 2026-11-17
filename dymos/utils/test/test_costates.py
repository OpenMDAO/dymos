import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.utils.costates import estimate_costates


class _GargODE(om.ExplicitComponent):
    """
    The ODE of example 1 of Garg et al., with the running cost as a second output.

    dy/dt = 2 y + 2 u sqrt(y) and L = (y + u^2) / 2. The state may be vector valued, in
    which case each entry follows the same equation and L is the sum over the entries.

    Garg, D., et al., "Direct trajectory optimization and costate estimation of finite-horizon
    and infinite-horizon optimal control problems using a Radau pseudospectral method",
    Computational Optimization and Applications 49, 2011.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('size', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        size = self.options['size']
        self.add_input('y', shape=(nn, size))
        self.add_input('u', shape=(nn, size))
        self.add_output('y_dot', shape=(nn, size), units='1/s')
        self.add_output('L', shape=(nn,), units='1/s')

        ar = np.arange(nn * size)
        self.declare_partials('y_dot', ['y', 'u'], rows=ar, cols=ar)
        self.declare_partials('L', 'y', rows=np.repeat(np.arange(nn), size), cols=ar, val=0.5)
        self.declare_partials('L', 'u', rows=np.repeat(np.arange(nn), size), cols=ar)

    def compute(self, inputs, outputs):
        y, u = inputs['y'], inputs['u']
        outputs['y_dot'] = 2 * y + 2 * u * np.sqrt(y)
        outputs['L'] = 0.5 * np.sum(y + u ** 2, axis=-1)

    def compute_partials(self, inputs, partials):
        y, u = inputs['y'], inputs['u']
        partials['y_dot', 'y'] = (2 + u / np.sqrt(y)).ravel()
        partials['y_dot', 'u'] = (2 * np.sqrt(y)).ravel()
        partials['L', 'u'] = u.ravel()


def _garg_exact_costate(t, y0=2.0, yf=1.0, tf=5.0):
    """
    Return the exact costate of y in the example of Garg et al.

    With x = sqrt(y) the problem is linear-quadratic: dx/dt = x + u, L = (x^2 + u^2) / 2.
    Its solution is x = a exp(sqrt(2) t) + b exp(-sqrt(2) t) with the costate p = x - dx/dt,
    and the costate of y is p / (2 x).
    """
    r = np.sqrt(2.0)
    den = np.exp(r * tf) - np.exp(-r * tf)
    a = (np.sqrt(yf) - np.sqrt(y0) * np.exp(-r * tf)) / den
    b = (np.sqrt(y0) * np.exp(r * tf) - np.sqrt(yf)) / den
    x = a * np.exp(r * t) + b * np.exp(-r * t)
    p = a * (1 - r) * np.exp(r * t) + b * (1 + r) * np.exp(-r * t)
    return p / (2 * x)


def _solve(p, obj):
    """
    Run the driver.

    The multipliers are only as accurate as the optimality conditions at the final point, so
    the tests use a tight tolerance for SLSQP.
    """
    dm.run_problem(p, simulate=False, make_plots=False)
    return p


def _garg_problem(tx, size=1, defect_ref=None, obj_ref=None):
    p = om.Problem()
    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1.0E-12, maxiter=500)
    p.driver.options['disp'] = False
    p.driver.declare_coloring()

    phase = dm.Phase(ode_class=_GargODE, ode_init_kwargs={'size': size}, transcription=tx)
    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, fix_duration=True)
    kwargs = {} if defect_ref is None else {'defect_ref': defect_ref}
    phase.add_state('y', shape=(size,), fix_initial=True, rate_source='y_dot', targets=['y'],
                    lower=1.0E-3, **kwargs)
    phase.add_state('J', fix_initial=True, rate_source='L', **kwargs)
    phase.add_control('u', shape=(size,), targets=['u'])
    phase.add_boundary_constraint('y', loc='final', equals=1.0)
    phase.add_objective('J', loc='final', **({} if obj_ref is None else {'ref': obj_ref}))

    p.setup()

    phase.set_time_val(initial=0.0, duration=5.0)
    phase.set_state_val('y', [2.0 * np.ones(size), np.ones(size)])
    phase.set_state_val('J', [0.0, 1.0])
    phase.set_control_val('u', [-np.ones(size), np.ones(size)])

    return _solve(p, 'phase0.timeseries.J')


@use_tempdirs
class TestEstimateCostates(unittest.TestCase):

    # transcription, tolerance at the nodes, tolerance at the ends of the phase
    cases = {'radau': (lambda: dm.Radau(num_segments=10, order=3), 1.0E-2, 5.0E-3),
             'gauss_lobatto': (lambda: dm.GaussLobatto(num_segments=10, order=3), 2.0E-2, None),
             'birkhoff_lgl': (lambda: dm.Birkhoff(num_nodes=20, grid_type='lgl'), 5.0E-3, 5.0E-3),
             'birkhoff_cgl': (lambda: dm.Birkhoff(num_nodes=20, grid_type='cgl'), 2.0E-2, 5.0E-3)}

    def test_against_exact_costates(self):
        for name, (make_tx, tol, end_tol) in self.cases.items():
            with self.subTest(transcription=name):
                p = _garg_problem(make_tx())
                result = estimate_costates(p)['phase0']

                lam_y = result['costates']['y']
                self.assertEqual(lam_y.shape, (len(result['time']), 1))
                assert_near_equal(lam_y.ravel(), _garg_exact_costate(result['time']), tolerance=tol)

                # The costate of the integrated cost is one.
                assert_near_equal(result['costates']['J'].ravel(), np.ones(len(result['time'])),
                                  tolerance=1.0E-6)

                if end_tol is None:
                    self.assertEqual(result['initial_costates'], {})
                    self.assertEqual(result['final_costates'], {})
                else:
                    assert_near_equal(result['initial_costates']['y'], _garg_exact_costate(0.0),
                                      tolerance=end_tol)
                    assert_near_equal(result['final_costates']['y'], _garg_exact_costate(5.0),
                                      tolerance=end_tol)

    def test_vector_valued_state(self):
        p = _garg_problem(dm.Radau(num_segments=10, order=3), size=2)
        result = estimate_costates(p)['phase0']

        lam_y = result['costates']['y']
        self.assertEqual(lam_y.shape, (len(result['time']), 2))
        exact = _garg_exact_costate(result['time'])
        assert_near_equal(lam_y[:, 0], exact, tolerance=1.0E-2)
        assert_near_equal(lam_y[:, 1], exact, tolerance=1.0E-2)
        self.assertEqual(result['final_costates']['y'].shape, (2,))

    def test_scaling_does_not_change_the_costates(self):
        nominal = estimate_costates(_garg_problem(dm.Radau(num_segments=6, order=3)))['phase0']
        scaled_defects = estimate_costates(_garg_problem(dm.Radau(num_segments=6, order=3),
                                                         defect_ref=37.0))['phase0']
        scaled_objective = estimate_costates(_garg_problem(dm.Radau(num_segments=6, order=3),
                                                           obj_ref=5.0))['phase0']

        assert_near_equal(scaled_defects['costates']['y'], nominal['costates']['y'],
                          tolerance=1.0E-6)
        assert_near_equal(scaled_objective['costates']['y'], nominal['costates']['y'],
                          tolerance=1.0E-6)

    def test_brachistochrone_hamiltonian(self):
        # For the minimum time problem the costates of x and y are constant, and the
        # Hamiltonian is constant and equal to -1.
        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1.0E-12, maxiter=500)
        p.driver.options['disp'] = False
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10, order=3))
        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0))
        phase.add_state('x', fix_initial=True)
        phase.add_state('y', fix_initial=True)
        phase.add_state('v', fix_initial=True)
        phase.add_control('theta', units='rad', lower=1.0E-3, upper=np.pi - 1.0E-3)
        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)
        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_boundary_constraint('y', loc='final', equals=5.0)
        phase.add_objective('time', loc='final')

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0.09, 1.75])

        _solve(p, 'phase0.timeseries.time')

        result = estimate_costates(p)['phase0']
        idxs = result['node_idxs']
        lam_x, lam_y, lam_v = (result['costates'][s].ravel() for s in ('x', 'y', 'v'))

        assert_near_equal(lam_x, lam_x[0] * np.ones(len(idxs)), tolerance=1.0E-5)
        assert_near_equal(lam_y, lam_y[0] * np.ones(len(idxs)), tolerance=1.0E-5)

        v = p.get_val('phase0.timeseries.v')[idxs, 0]
        theta = p.get_val('phase0.timeseries.theta', units='rad')[idxs, 0]
        hamiltonian = lam_x * v * np.sin(theta) - lam_y * v * np.cos(theta) + \
            lam_v * 9.80665 * np.cos(theta)

        assert_near_equal(hamiltonian, -np.ones(len(idxs)), tolerance=1.0E-3)

        # The final speed is free, so its costate vanishes at the end of the phase.
        assert_near_equal(result['final_costates']['v'], 0.0, tolerance=1.0E-6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
