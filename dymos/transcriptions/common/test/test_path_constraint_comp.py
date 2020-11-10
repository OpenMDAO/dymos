import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om

import dymos as dm
from dymos.transcriptions.common import PathConstraintComp
from dymos.transcriptions.grid_data import GridData
from dymos.phase.options import ControlOptionsDictionary
from dymos.utils.testing_utils import assert_check_partials


class TestPathConstraintComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        transcription = 'radau-ps'

        self.gd = gd = GridData(num_segments=2,
                                transcription_order=3,
                                segment_ends=[0.0, 3.0, 10.0],
                                transcription=transcription)

        ndn = gd.subset_num_nodes['state_disc']

        self.p = om.Problem(model=om.Group())

        controls = {'a': ControlOptionsDictionary(),
                    'b': ControlOptionsDictionary(),
                    'c': ControlOptionsDictionary(),
                    'd': ControlOptionsDictionary()}

        controls['a'].update({'units': 'm', 'shape': (1,), 'opt': False})
        controls['b'].update({'units': 's', 'shape': (3,), 'opt': False})
        controls['c'].update({'units': 'kg', 'shape': (3, 3), 'opt': False})

        ivc = om.IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('a_disc', val=np.zeros((ndn, 1)), units='m')
        ivc.add_output('b_disc', val=np.zeros((ndn, 3)), units='s')
        ivc.add_output('c_disc', val=np.zeros((ndn, 3, 3)), units='kg')

        path_comp = PathConstraintComp(num_nodes=gd.num_nodes)

        self.p.model.add_subsystem('path_constraints', subsys=path_comp)

        path_comp._add_path_constraint_configure('a', shape=(1,),
                                                 lower=0, upper=10, units='m')
        path_comp._add_path_constraint_configure('b', shape=(3,),
                                                 lower=0, upper=10, units='s')
        path_comp._add_path_constraint_configure('c', shape=(3, 3),
                                                 lower=0, upper=10, units='kg')

        self.p.model.connect('a_disc', 'path_constraints.all_values:a')
        self.p.model.connect('b_disc', 'path_constraints.all_values:b')
        self.p.model.connect('c_disc', 'path_constraints.all_values:c')

        self.p.setup()

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):
        p = self.p
        gd = self.gd
        assert_almost_equal(p['a_disc'],
                            p['path_constraints.path:a'][gd.subset_node_indices['state_disc'], ...])

        assert_almost_equal(p['b_disc'],
                            p['path_constraints.path:b'][gd.subset_node_indices['state_disc'], ...])

        assert_almost_equal(p['c_disc'],
                            p['path_constraints.path:c'][gd.subset_node_indices['state_disc'], ...])

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
