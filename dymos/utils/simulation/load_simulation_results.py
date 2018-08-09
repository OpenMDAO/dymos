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

    try:
        case = cr.system_cases.get_case(-1)
    except IndexError:
        raise RuntimeError('Did not find a valid simulation in file: {0}'.format(f))

    loaded_outputs = cr.list_outputs(case=case, explicit=True, implicit=True, values=True,
                                     out_stream=None)

    if len([s for s in loaded_outputs if s[0].startswith('phases.')]) > 0:
        return TrajectorySimulationResults(f)
    else:
        return PhaseSimulationResults(f)
