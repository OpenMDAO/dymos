import unittest

import numpy as np

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.utils.lgl import lgl


@use_tempdirs
class TestPhaseInterp(unittest.TestCase):

    def test_linear_state(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_state('x', fix_initial=True, fix_final=True)
        gd = tx.grid_data
        state_input_nodes = gd.node_ptau[gd.subset_node_indices['state_input']]
        expected = np.atleast_2d(100 * 0.5 * (state_input_nodes + 1)).T
        assert_near_equal(phase.interp('x', [0, 100]), expected)

    def test_quadratic_state(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_state('x', fix_initial=True, fix_final=True)
        gd = tx.grid_data
        state_input_nodes = gd.node_ptau[gd.subset_node_indices['state_input']]
        expected = np.atleast_2d((2 * state_input_nodes) ** 2).T
        assert_near_equal(phase.interp('x', xs=[-2, 0, 2], ys=[4, 0, 4], kind='quadratic'), expected)

    def test_linear_control(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)
        gd = tx.grid_data
        input_nodes = gd.node_ptau[gd.subset_node_indices['control_input']]
        expected = np.atleast_2d(100 * 0.5 * (input_nodes + 1)).T
        assert_near_equal(phase.interp('u', [0, 100]), expected)

    def test_quadratic_control(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)
        gd = tx.grid_data
        input_nodes = gd.node_ptau[gd.subset_node_indices['control_input']]
        expected = np.atleast_2d((2 * input_nodes) ** 2).T
        assert_near_equal(phase.interp('u', xs=[-2, 0, 2], ys=[4, 0, 4], kind='quadratic'), expected)

    def test_polynomial_control(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_polynomial_control('u', fix_initial=True, fix_final=True, order=3)
        xs = np.linspace(-10, 10, 100)
        ys = xs**3

        input_nodes, _ = lgl(4)
        expected = np.atleast_2d((10 * input_nodes) ** 3).T

        assert_near_equal(phase.interp('u', ys=ys, xs=xs, kind='cubic'), expected)

    def test_invalid_var(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        with self.assertRaises(ValueError) as e:
            phase.interp('x', [0, 100])

        expected = 'Could not find a state, control, or polynomial control named x to be ' \
                   'interpolated.\nPlease explicitly specified the node subset onto which this ' \
                   'value should be interpolated.'

        self.assertEqual(str(e.exception), expected)

    def test_invalid_var_explicit_nodes(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)
        gd = tx.grid_data
        state_input_nodes = gd.node_ptau[gd.subset_node_indices['control_input']]
        expected = np.atleast_2d(100 * 0.5 * (state_input_nodes + 1)).T
        assert_near_equal(phase.interp('foo', [0, 100], nodes='control_input'), expected)

    def test_invalid_ys(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)

        with self.assertRaises(ValueError) as e:
            phase.interp('foo', [0, 5, 10], nodes='control_input')

        expected = 'xs may only be unspecified when len(ys)=2'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_kind(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)

        with self.assertRaises(ValueError) as e:
            phase.interp('u', [0, 5], nodes='control_input', kind='quadratic')

        expected = 'kind must be linear when xs is unspecified.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_xs(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)

        with self.assertRaises(ValueError) as e:
            phase.interp('u', xs=[[0, 1, 2], [3, 4, 5]], ys=[0, 1, 2, 3, 4, 5])

        expected = 'xs must be viewable as a 1D array'
        self.assertEqual(str(e.exception), expected)

    def test_underspecified(self):
        tx = dm.GaussLobatto(num_segments=8, order=5, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.add_control('u', fix_initial=True, fix_final=True)

        with self.assertRaises(ValueError) as e:
            phase.interp(ys=[0, 5])

        expected = 'nodes for interpolation were not specified but the name of the variable ' \
                   'to be interpolated was not provided.\nPlease specify the name of the ' \
                   'interpolated variable or a node subset.'

        self.assertEqual(str(e.exception), expected)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
