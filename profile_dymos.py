#!/usr/bin/env python
"""
Comprehensive profiling script for Dymos codebase.
Profiles key components and identifies performance bottlenecks.
"""

import cProfile
import pstats
import io
import sys
import os
from pstats import SortKey

# Add dymos to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
import dymos as dm
from dymos.examples.brachistochrone import BrachistochroneODE


def profile_function(func, func_name, *args, **kwargs):
    """Profile a function and return statistics."""
    print(f"\n{'='*80}")
    print(f"Profiling: {func_name}")
    print(f"{'='*80}\n")

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"Error during profiling: {e}")
        profiler.disable()
        return None

    profiler.disable()

    # Create statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    # Sort by cumulative time
    ps.sort_stats(SortKey.CUMULATIVE)

    print("\n--- Top 30 functions by CUMULATIVE time ---")
    ps.print_stats(30)

    # Sort by internal time
    ps.sort_stats(SortKey.TIME)
    print("\n--- Top 30 functions by INTERNAL (self) time ---")
    ps.print_stats(30)

    # Print to stdout
    print(s.getvalue())

    # Save to file
    stats_file = f"profile_{func_name.replace(' ', '_')}.stats"
    ps.dump_stats(stats_file)
    print(f"\nProfile stats saved to: {stats_file}")

    return result


def brachistochrone_min_time_simple():
    """Simple brachistochrone problem for profiling."""
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    # Use smaller problem for faster profiling
    t = dm.GaussLobatto(num_segments=10, order=3, compressed=True)
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='xdot')
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='ydot')
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m/s', rate_source='vdot')

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()

    # Profile setup
    p.setup(check=True)

    phase.set_time_val(initial=0.0, duration=2.0)
    phase.set_state_val('x', [0, 10])
    phase.set_state_val('y', [10, 5])
    phase.set_state_val('v', [0, 9.9])
    phase.set_control_val('theta', [5, 100])
    phase.set_parameter_val('g', 9.80665)

    # Profile run
    p.run_driver()

    return p


def brachistochrone_setup_only():
    """Profile just the setup phase."""
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'

    t = dm.GaussLobatto(num_segments=20, order=3, compressed=True)
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')
    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='xdot')
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='ydot')
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m/s', rate_source='vdot')
    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)
    phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])
    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    return p


def brachistochrone_radau():
    """Radau transcription for comparison."""
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    t = dm.Radau(num_segments=10, order=3, compressed=True)
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')
    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='xdot')
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='ydot')
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m/s', rate_source='vdot')
    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)
    phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])
    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    phase.set_time_val(initial=0.0, duration=2.0)
    phase.set_state_val('x', [0, 10])
    phase.set_state_val('y', [10, 5])
    phase.set_state_val('v', [0, 9.9])
    phase.set_control_val('theta', [5, 100])
    phase.set_parameter_val('g', 9.80665)

    p.run_driver()
    return p


def main():
    """Run all profiling tests."""
    print("="*80)
    print("DYMOS PERFORMANCE PROFILING")
    print("="*80)

    # Profile different aspects
    tests = [
        (brachistochrone_setup_only, "Brachistochrone_Setup_Phase"),
        (brachistochrone_min_time_simple, "Brachistochrone_GaussLobatto_Full"),
        (brachistochrone_radau, "Brachistochrone_Radau_Full"),
    ]

    for test_func, test_name in tests:
        try:
            profile_function(test_func, test_name)
        except Exception as e:
            print(f"\nError running {test_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nProfile files generated:")
    print("  - profile_Brachistochrone_Setup_Phase.stats")
    print("  - profile_Brachistochrone_GaussLobatto_Full.stats")
    print("  - profile_Brachistochrone_Radau_Full.stats")
    print("\nTo analyze further, use:")
    print("  python -m pstats profile_<name>.stats")


if __name__ == '__main__':
    main()
