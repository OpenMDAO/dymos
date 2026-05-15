import os
from openmdao.utils.general_utils import is_truthy

from .analytic.analytic import Analytic
from .explicit_shooting import ExplicitShooting
from .pseudospectral.gauss_lobatto import GaussLobatto

if is_truthy(os.environ.get('DYMOS_LEGACY_RADAU', '0')):
    from .pseudospectral.radau_pseudospectral import Radau
else:
    from .pseudospectral.radau_new import RadauNew as Radau

from .pseudospectral.birkhoff import Birkhoff
from .picard_shooting.picard_shooting import PicardShooting
