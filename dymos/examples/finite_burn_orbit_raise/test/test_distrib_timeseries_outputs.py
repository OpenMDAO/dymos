import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm

try:
    from mpi4py import MPI
except:
    MPI = None

from dymos.transcriptions.explicit_shooting.explicit_shooting import ExplicitShooting


class OscillatorODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("parallel", types=bool, default=False)

    def setup(self):
        nn = self.options["num_nodes"]

        # If doing parallel, determine the subsection of the array this proc will work on
        if self.options["parallel"]:
            sizes, offsets = evenly_distrib_idxs(self.comm.size, nn)
            self.start_idx = offsets[self.comm.rank]
            self.io_size = sizes[self.comm.rank]
            self.end_idx = self.start_idx + self.io_size
        else:
            self.start_idx = 0
            self.io_size = nn
            self.end_idx = nn

        # Inputs
        self.add_input("x", shape=(nn,), desc="displacement", units="m")
        self.add_input("v", shape=(nn,), desc="velocity", units="m/s")
        self.add_input("k", shape=(nn,), desc="spring constant", units="N/m")
        self.add_input("c", shape=(nn,), desc="damping coefficient", units="N*s/m")
        self.add_input("m", shape=(nn,), desc="mass", units="kg")

        # self.add_output("x_dot", val=np.zeros(nn), desc="rate of change of displacement", units="m/s")
        self.add_output(
            "v_dot",
            val=np.zeros(self.io_size),
            desc="rate of change of velocity",
            units="m/s**2",
            distributed=self.options["parallel"],
        )

        # Add an auxiliary output to constrain
        self.add_output(
            "v_squared",
            val=np.zeros(self.io_size),
            units="(m/s)**2",
            distributed=self.options["parallel"],
        )

        r = np.arange(self.io_size, dtype=int)
        c = r + self.start_idx
        self.declare_partials(of="*", wrt="*", rows=r, cols=c, method="fd")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        v = inputs["v"]
        k = inputs["k"]
        c = inputs["c"]
        m = inputs["m"]

        # Pretend this is an expensive computation that operates only on scalars
        for i_out, i_in in enumerate(np.arange(self.start_idx, self.end_idx)):
            f_spring = -k[i_in] * x[i_in]
            f_damper = -c[i_in] * v[i_in]

            outputs["v_dot"][i_out] = (f_spring + f_damper) / m[i_in]

            # Compute auxiliary output
            outputs["v_squared"][i_out] = v[i_in]**2


class TestDistributedODEOutputsToTimeseries(unittest.TestCase):

    N_PROC = 3

    @unittest.skipIf(MPI is None, 'this test requires MPI')
    def test_ditrib_ode_output_to_timeseries(self):
        transcriptions = {'GaussLobatto': dm.GaussLobatto(num_segments=15, order=7, compressed=True),
                          'Radau': dm.Radau(num_segments=15, order=7, compressed=True),
                          'Birkhoff': dm.Birkhoff(num_nodes=80),
                          'ExplicitShooting': dm.ExplicitShooting(num_segments=15, order=3),
                          'PicardShooting': dm.PicardShooting(num_segments=15, nodes_per_seg=7)}

        for tx_name, tx in transcriptions.items():

            with self.subTest(f'{tx_name=}'):
                # define the OpenMDAO problem
                p = om.Problem(model=om.Group())

                p.driver = om.ScipyOptimizeDriver()
                p.driver.options["optimizer"] = "SLSQP"
                p.driver.declare_coloring()

                # define a Trajectory object and add to model
                traj = dm.Trajectory()
                p.model.add_subsystem("traj", subsys=traj)

                # define a Phase as specified above and add to Phase
                phase = dm.Phase(
                    ode_class=OscillatorODE,
                    transcription=tx,
                    ode_init_kwargs={"parallel": MPI.COMM_WORLD.Get_size() > 1},  # use distributed comp if running in parallel
                )
                traj.add_phase(name="phase0", phase=phase)

                # Tell Dymos that the duration of the phase is bounded.
                phase.set_time_options(fix_initial=True, fix_duration=True)

                # Tell Dymos the states to be propagated using the given ODE.
                phase.add_state("x", fix_initial=True, rate_source="v", targets=["x"], units="m")
                phase.add_state("v", fix_initial=True, rate_source="v_dot", targets=["v"], units="m/s")

                # The spring constant, damping coefficient, and mass are inputs to the system that are
                # constant throughout the phase.
                phase.add_parameter("k", units="N/m", targets=["k"])
                phase.add_parameter("c", units="N*s/m", targets=["c"])
                phase.add_parameter("m", units="kg", targets=["m"])

                # Since we're using an optimization driver, an objective is required.  We"ll minimize
                # the final time in this case.
                phase.add_objective("time", loc="final")

                # EDIT: Suppose we want to adjust the damping to satisfy
                #       a constraint based on an auxiliary parameter
                phase.set_parameter_options("c", opt=True)
                phase.add_path_constraint("v_squared", upper=30.0)

                # Setup the OpenMDAO problem
                # Unless compute_jacvec_product is defined by the ODE component, we must use forward mode
                p.setup(mode="fwd")

                # Assign values to the times and states
                phase.set_time_val(0.0, 15.0)
                phase.set_state_val("x", 10.0)
                phase.set_state_val("v", 0.0)
                phase.set_parameter_val("k", 1.0)
                phase.set_parameter_val("c", 0.5)
                phase.set_parameter_val("m", 1.0)

                dm.run_problem(p, run_driver=True, simulate=False, make_plots=True)

                v_squared = p.get_val('traj.phase0.timeseries.v_squared')
                x = p.get_val('traj.phase0.timeseries.x')

                assert_near_equal(v_squared[-1, ...], 0.0, tolerance=1.0E-4)
                assert_near_equal(x[-1, ...], 0.0, tolerance=1.0E-2)
