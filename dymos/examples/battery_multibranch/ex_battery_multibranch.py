"""
Example that shows how to use multiple phases in Dymos to model failure of a battery cell
in a simple electrical system.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, IndepVarComp

from dymos import Phase
from dymos.utils.lgl import lgl

from battery_multibranch_ode import BatteryODE


def run_example(optimizer='SLSQP', transcription='gauss-lobatto'):

    prob = Problem(model=Group())

    opt = prob.driver = pyOptSparseDriver()

    opt.options['optimizer'] = optimizer
    opt.options['dynamic_simul_derivs'] = True
    if optimizer == 'SNOPT':
        opt.opt_settings['Major iterations limit'] = 1000
        opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
        opt.opt_settings['Major optimality tolerance'] = 1.0E-6
        opt.opt_settings["Linesearch tolerance"] = 0.10
        opt.opt_settings['iSumm'] = 6

    num_seg = 5
    seg_ends, _ = lgl(num_seg + 1)

    phase = Phase(transcription,
                  ode_class=BatteryODE,
                  num_segments=num_seg,
                  segment_ends=seg_ends,
                  transcription_order=5,
                  compressed=False)

    prob.model.add_subsystem('phase0', phase)

    # Sim for 2.5 hours. Battery runs out in 3.
    phase.set_time_options(fix_initial=True, fix_duration=True, initial=0.0, duration=2*3600.0)

    phase.set_state_options('state_of_charge', fix_initial=True, fix_final=False)

    #phase.add_objective('range', loc='final', ref=-1.0)

    prob.setup()

    prob.run_driver()


if __name__ == '__main__':
    run_example(optimizer='SNOPT', transcription='radau-ps')