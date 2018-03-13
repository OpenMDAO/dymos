from dymos.glm.ozone.methods.runge_kutta.explicit_runge_kutta import \
    ForwardEuler, ExplicitMidpoint, ExplicitMidpointST, HeunsMethod, RalstonsMethod, \
    KuttaThirdOrder, KuttaThirdOrderST, RK4, RK4ST, RK6, RK6ST
from dymos.glm.ozone.methods.runge_kutta.implicit_runge_kutta import \
    BackwardEuler, ImplicitMidpoint, TrapezoidalRule
from dymos.glm.ozone.methods.runge_kutta.gauss_legendre import GaussLegendre
from dymos.glm.ozone.methods.runge_kutta.lobatto import LobattoIIIA
from dymos.glm.ozone.methods.runge_kutta.radau import Radau

from dymos.glm.ozone.utils.misc import _get_class


method_classes = {
    'ForwardEuler': ForwardEuler(),
    'BackwardEuler': BackwardEuler(),
    'ExplicitMidpoint': ExplicitMidpoint(),
    'ImplicitMidpoint': ImplicitMidpoint(),
    'KuttaThirdOrder': KuttaThirdOrder(),
    'RK4': RK4(),
    'RK6': RK6(),
    'RalstonsMethod': RalstonsMethod(),
    'HeunsMethod': HeunsMethod(),
    'GaussLegendre2': GaussLegendre(2),
    'GaussLegendre4': GaussLegendre(4),
    'GaussLegendre6': GaussLegendre(6),
    'Lobatto2': LobattoIIIA(2),
    'Lobatto4': LobattoIIIA(4),
    'RadauI3': Radau('I', 3),
    'RadauI5': Radau('I', 5),
    'RadauII3': Radau('II', 3),
    'RadauII5': Radau('II', 5),
    'Trapezoidal': TrapezoidalRule(),
}


family_names = [
    'ExplicitRungeKutta',
    'ImplicitRungeKutta',
    'GaussLegendre',
    'Lobatto',
    'Radau',
]


method_families = {}
method_families['ExplicitRungeKutta'] = [
    'ForwardEuler',
    'ExplicitMidpoint',
    'KuttaThirdOrder',
    'RK4',
    'RK6',
]
method_families['ImplicitRungeKutta'] = [
    'BackwardEuler',
    'ImplicitMidpoint',
    'Trapezoidal',
]
method_families['GaussLegendre'] = [
    'GaussLegendre2',
    'GaussLegendre4',
    'GaussLegendre6',
]
method_families['Lobatto'] = [
    'Lobatto2',
    'Lobatto4',
]
method_families['Radau'] = [
    'RadauI3',
    'RadauI5',
    'RadauII3',
    'RadauII5',
]


def get_method(method_name):
    return _get_class(method_name, method_classes, 'Method')
