from __future__ import print_function, division, absolute_import

from openmdao.api import CaseReader

from .phase_simulation_results import PhaseSimulationResults
from .trajectory_simulation_results import TrajectorySimulationResults


def load_simulation_results(f):
    """
    Load PhaseSimulationResults or TrajectorySimulationResults from the given file

    Parameters
    ----------
    f : str
        The file path from which to load the simulation results.

    Returns
    -------
    res : PhaseSimulationResults or TrajectorySimulationResults
        The PhaseSimulationResults or TrajectorySimulationResults loaded from the given file.

    """
    cr = CaseReader(f)

    system_cases = cr.list_cases('root')

    try:
        case = cr.get_case(system_cases[-1])
    except IndexError:
        raise RuntimeError('Did not find a valid simulation in file: {0}'.format(f))

    loaded_outputs = case.list_outputs(explicit=True, implicit=True, values=True,
                                       out_stream=None)

    if len([s for s in loaded_outputs if s[0].startswith('phases.')]) > 0:
        return TrajectorySimulationResults(f)
    else:
        return PhaseSimulationResults(f)
