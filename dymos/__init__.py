__version__ = '1.8.1-dev'

from .phase import Phase, AnalyticPhase
from .transcriptions import GaussLobatto, Radau, ExplicitShooting, Analytic
from .transcriptions.grid_data import GaussLobattoGrid, RadauGrid, UniformGrid
from .trajectory.trajectory import Trajectory
from .run_problem import run_problem
from ._options import options
