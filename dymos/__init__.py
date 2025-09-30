__version__ = '1.15.0'


from openmdao.utils.general_utils import env_truthy as _env_truthy


from dymos.phase import Phase, AnalyticPhase
from dymos.transcriptions import GaussLobatto, ExplicitShooting, Analytic, \
    Birkhoff, PicardShooting

if _env_truthy('DYMOS_2'):
    from dymos.transcriptions import RadauNew as Radau
else:
    from dymos.transcriptions import Radau

from dymos.transcriptions.grid_data import GaussLobattoGrid, ChebyshevGaussLobattoGrid, \
    RadauGrid, UniformGrid, BirkhoffGrid
from dymos.trajectory.trajectory import Trajectory
from dymos.run_problem import run_problem
from ._options import options
