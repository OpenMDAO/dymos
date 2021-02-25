from pyxdsm.XDSM import XDSM, OPT, SUBOPT, SOLVER, DOE, IFUNC, FUNC, GROUP, IGROUP, METAMODEL


x = XDSM(use_sfmath=False)

x.add_system('OPT', OPT, r"\text{Optimizer}")
x.add_system('ODE', GROUP, r"\text{ODE or DAE}")

x.connect('ODE', 'OPT', ["J", r"\bar{g}_0", r"\bar{g}_f", r"\bar{p}"],  label_width=4)
x.connect('OPT', 'ODE', ["t", r"\bar{x}", r"\bar{u}", r"\bar{d}"],  label_width=4)

x.write('opt_control')


x = XDSM(use_sfmath=False)

x.add_system('OPT_static', OPT, r"\text{Static Optimizer}")
x.add_system('static', GROUP, r"\text{Static System Model}")

x.add_system('OPT_dynamic', OPT, r"\text{Dynamic Optimizer}")
x.add_system('dynamic', GROUP, r"\text{ODE or DAE}")

x.connect('dynamic', 'OPT_dynamic', [r"J_\text{dynamic}", r"\bar{g}_0", r"\bar{g}_f", r"\bar{p}"],  label_width=4)
x.connect('OPT_dynamic', 'dynamic', ["t", r"\bar{x}", r"\bar{u}"],  label_width=3)

x.connect('static', 'OPT_static', [r"J_\text{static}", r"g_\text{static}"],  label_width=3)
x.connect('OPT_static', 'static', [r"\bar{d}"],  label_width=3)

x.connect('OPT_static', 'OPT_dynamic', [r'\text{static optimization}', r'\text{outputs}'])
x.connect('OPT_dynamic', 'OPT_static', [r'\text{dynamic optimization}', r'\text{outputs}'])

x.write('sequential_co_design')


x = XDSM(use_sfmath=False)

x.add_system('OPT', OPT, r"\text{Optimizer}")
x.add_system('static', GROUP, r"\text{Static System Model}")
x.add_system('dynamic', GROUP, r"\text{ODE or DAE}")

x.connect('dynamic', 'OPT', [r"J", r"\bar{g}_0", r"\bar{g}_f", r"\bar{p}"],  label_width=4)
x.connect('OPT', 'dynamic', ["t", r"\bar{x}", r"\bar{u}"],  label_width=3)

x.connect('static', 'OPT', [r"g_\text{static}"],  label_width=3)
x.connect('OPT', 'static', [r"\bar{d}"],  label_width=3)

x.connect('static', 'dynamic', [r'\text{static}', r'\text{outputs}'])
x.connect('dynamic', 'static', [r'\text{dynamic}', r'\text{outputs}'])

x.write('coupled_co_design')
