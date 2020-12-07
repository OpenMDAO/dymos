from openmdao.api import OptionsDictionary

options = OptionsDictionary()

options.declare('include_check_partials', default=True, types=bool,
                desc='If True, include dymos components when checking partials.')
