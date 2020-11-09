from numbers import Number
from collections.abc import Iterable

import openmdao.api as om
from ..utils.misc import _unspecified


class LinkageOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for Phase linkages.
    """

    def __init__(self):
        super().__init__()

        self.declare(name='phase_a', types=str,
                     desc='name of the first phase in the linkage')

        self.declare(name='phase_b', types=str,
                     desc='name of the second phase in the linkage')

        self.declare(name='var_a', types=str,
                     desc='name of the first variable in the linkage')

        self.declare(name='var_b', types=str,
                     desc='name of the second variable in the linkage')

        self.declare(name='loc_a', values=('initial', 'final', '--', '-+', '+-', '++'),
                     desc='location of the first variable in the linkage (\'initial\' or \'final\')')

        self.declare(name='loc_b', values=('initial', 'final', '--', '-+', '+-', '++'),
                     desc='location of the second variable in the linkage (\'initial\' or \'final\')')

        self.declare(name='sign_a', types=Number, default=1.0,
                     desc='sign of the first variable in the linkage (\'initial\' or \'final\')')

        self.declare(name='sign_b', types=Number, default=-1.0,
                     desc='sign of the second variable in the linkage (\'initial\' or \'final\')')

        self.declare(name='units', default=_unspecified,
                     allow_none=True, desc='units in which the linkage constraint is defined')

        self.declare(name='shape', types=Iterable, allow_none=True, default=None,
                     desc='shape of the state variable, as determined by introspection')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True, desc='Lower bound of the resulting constraint.')

        self.declare(name='upper',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Upper bound of the resulting constraint.')

        self.declare(name='equals',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Equality constraint value of the resulting constraint.')

        self.declare(name='scaler',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Scaler of the resulting constraint.')

        self.declare(name='adder',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Adder of the resulting constraint.')

        self.declare(name='ref0',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Zero-reference of the resulting constraint.')

        self.declare(name='ref',
                     types=(Iterable, Number), default=None,
                     allow_none=True, desc='Unit-reference of the resulting constraint.')

        self.declare(name='linear', types=bool, default=False,
                     desc='If True, treat the resulting constraint as a linear constraint. This '
                          'option should only be applied to linked design variables and time.')

        self.declare(name='connected', types=bool, default=False,
                     desc='If True, this linkage is handled as a direct connection rather than'
                          'by a constraint.  This option only applies to time and states.')

        self.declare(name='constraint_name', types=str, default=None, allow_none=True,
                     desc='optional alternative constraint name to override name conflicts.')

        self._src_a = None
        self._src_b = None
        self._input_a = None
        self._input_b = None
        self._output = None
