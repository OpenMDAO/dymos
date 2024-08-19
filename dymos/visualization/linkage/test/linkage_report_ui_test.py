"""Test Dymos Linkage Reports GUI using Playwright."""
import asyncio
from packaging.version import Version
from playwright.async_api import async_playwright
from aiounittest import async_test
import unittest

from openmdao.utils.gui_testing_utils import _GuiTestCase
import sys

import openmdao.api as om
import openmdao
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp
from dymos.visualization.linkage.report import create_linkage_report

if 'win32' in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

my_loop = asyncio.new_event_loop()
asyncio.set_event_loop(my_loop)

""" A set of toolbar tests that runs on each model. """
toolbar_script = [
    {
        "desc": "Uncollapse All button",
        "id": "expand-all",
    },
    {
        "desc": "Collapse Outputs in View Only button",
        "id": "collapse-element-2",
    },
    {
        "desc": "Uncollapse In View Only button",
        "id": "expand-element",
    },
    {
        "desc": "Show Legend (off) button",
        "id": "legend-button",
    },
    {
        "desc": "Show Legend (on) button",
        "id": "legend-button",
    },
    {
        "desc": "Show Path (on) button",
        "id": "info-button",
    },
    {
        "desc": "Show Path (off) button",
        "id": "info-button",
    },
    {
        "desc": "Clear Arrows and Connection button",
        "id": "hide-connections",
    },
    {
        "desc": "Help (on) button",
        "id": "question-button",
    },
    {
        "desc": "Help (off) button",
        "id": "question-button",
    },
    {
        "desc": "Collapse All Outputs button",
        "id": "collapse-all",
    }
]

""" A dictionary of tests script with an array for each model."""
gui_test_script = [
    {"test": "toolbar"},
    {
        "desc": "Hover on matrix cell and check arrow count",
        "test": "hoverArrow",
        "selector": "g#n2elements rect#cellShape_node_41.vMid",
        "arrowCount": 2
    },
    {
        "desc": "Left-click on model tree element to zoom",
        "test": "click",
        "selector": "g#tree g.phase rect#climb",
        "button": "left"
    },
    # TODO: This particular test is broken due to timeout, not sure why yet.
    # {
    #     "desc": "Hover on matrix cell and check arrow count",
    #     "test": "hoverArrow",
    #     "selector": "g#n2elements rect#cellShape_node_128.vMid",
    #     "arrowCount": 1
    # },
    {"test": "root"},
    {
        "desc": "Right-click on model tree element to collapse",
        "test": "click",
        "selector": "g#tree rect#br_to_v1",
        "button": "right"
    },
    {
        "desc": "Hover over collapsed matrix cell and check arrow count",
        "test": "hoverArrow",
        "selector": "g#n2elements rect#cellShape_node_18.gMid",
        "arrowCount": 19
    },
    {
        "desc": "Right-click on model tree element to uncollapse",
        "test": "click",
        "selector": "g#tree rect#br_to_v1",
        "button": "right"
    },
    {"test": "root"},
    {
        "desc": "Check the number of cells in the matrix grid",
        "test": "count",
        "selector": "g#n2elements > g.n2cell",
        "count": 196
    },
    {
        "desc": "Perform a search on states:v",
        "test": "search",
        "searchString": "states:v",
        "diagElementCount": 25
    },
    {"test": "root"},
    {
        "desc": "Check that home button works after search",
        "test": "count",
        "selector": "g#n2elements > g.n2cell",
        "count": 196
    },
    {
        "desc": "Expand toolbar connections menu",
        "test": "hover",
        "selector": ".group-3 > div.expandable:first-child"
    },
    {
        "desc": "Press toolbar show all connections button",
        "test": "click",
        "selector": "#show-all-connections",
        "button": "left"
    },
    {
        "desc": "Check number of arrows",
        "test": "count",
        "selector": "g#n2arrows > g",
        "count": 154
    },
    {
        "desc": "Expand toolbar connections menu",
        "test": "hover",
        "selector": ".group-3 > div.expandable:first-child"
    },
    {
        "desc": "Press toolbar hide all connections button",
        "test": "click",
        "selector": "#hide-connections-2",
        "button": "left"
    },
    {
        "desc": "Check number of arrows",
        "test": "count",
        "selector": "g#n2arrows > g",
        "count": 0
    },
    {"test": "root"},
    {
        "desc": "Alt-right-click the climb.initial component",
        "test": "click",
        "selector": "rect#climb_initial",
        "button": "right",
        "modifiers": ["Alt"]
    },
    {
        "desc": "Check that variable selection dialog appears",
        "test": "count",
        "selector": "#childSelect-climb_initial",
        "count": 1
    },
    {
        "desc": "Select a variable to hide",
        "test": "click",
        "selector": "input#climb_initial_time-visible-check",
        "button": "left"
    },
    {
        "desc": "Click the Apply button",
        "test": "click",
        "selector": ".button-container button:last-child",
        "button": "left"
    },
    {
        "desc": "Check that the time variable is no longer displayed",
        "test": "count",
        "selector": "rect#climb_initial_time",
        "count": 0
    },
    {
        "desc": "Alt-right-click the climb.initial component",
        "test": "click",
        "selector": "rect#climb_initial",
        "button": "right",
        "modifiers": ["Alt"]
    },
    {
        "desc": "Click the Clear Search button",
        "test": "click",
        "selector": ".search-clear",
        "button": "left"
    },
    {
        "desc": "Perform a search in the variable selection dialog",
        "test": "var_select_search",
        "searchString": "gam",
        "foundVariableCount": 1
    },
    {
        "desc": "Click the Apply button",
        "test": "click",
        "selector": ".button-container button:last-child",
        "button": "left"
    },
    {
        "desc": "Check the number of cells in the matrix grid",
        "test": "count",
        "selector": "g#n2elements > g.n2cell",
        "count": 192
    },

]

resize_dirs = {
    'top': [0, -1],
    'top-right': [1, -1],
    'right': [1, 0],
    'bottom-right': [1, 1],
    'bottom': [0, 1],
    'bottom-left': [-1, 1],
    'left': [-1, 0],
    'top-left': [-1, -1]
}

current_test = 1


@unittest.skipUnless(Version(openmdao.__version__) > Version("3.19"), "reports API is too old")
@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class dymos_linkage_gui_test_case(_GuiTestCase):
    def setUp(self):
        p = om.Problem()

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['derivative_test'] = 'first-order'

        p.driver.declare_coloring()

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)
        br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)
        br_to_v1.add_state('v', fix_initial=True, lower=0.0001, ref=100.0, defect_ref=100.0)
        br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
        br_to_v1.add_timeseries_output('*')

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                       ode_init_kwargs={'mode': 'runway'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rto.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        rto.add_parameter('alpha', val=0.0, opt=False, units='deg')
        rto.add_timeseries_output('*')

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
        v1_to_vr.add_timeseries_output('*')

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                          ode_init_kwargs={'mode': 'runway'})
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
        rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rotate.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        rotate.add_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10,
                           ref=10, val=[0, 10], control_type='polynomial')
        rotate.add_timeseries_output('*')

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=5),
                         ode_init_kwargs={'mode': 'climb'})
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)
        climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        climb.add_state('h', fix_initial=True, lower=0.0, ref=1.0, defect_ref=1.0)
        climb.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        climb.add_state('gam', fix_initial=True, lower=0.0, ref=0.05, defect_ref=0.05)
        climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10)
        climb.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('climb', climb)

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={'br_to_v1': ['m'], 'v1_to_vr': ['m'], 'rto': ['m'],
                                    'rotate': ['m'], 'climb': ['m']})

        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust',
                           targets={'br_to_v1': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={'v1_to_vr': ['T'], 'rotate': ['T'], 'climb': ['T']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coeffcient',
                           targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'], 'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft', static_target=False,
                           desc='runway altitude',
                           targets={'br_to_v1': ['h'], 'v1_to_vr': ['h'], 'rto': ['h'],
                                    'rotate': ['h']})

        traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', static_target=True,
                           desc='atmospheric density',
                           targets={'br_to_v1': ['rho'], 'v1_to_vr': ['rho'], 'rto': ['rho'],
                                    'rotate': ['rho']})

        traj.add_parameter('S', val=124.7, opt=False, units='m**2', static_target=True,
                           desc='aerodynamic reference area',
                           targets={'br_to_v1': ['S'], 'v1_to_vr': ['S'], 'rto': ['S'],
                                    'rotate': ['S'], 'climb': ['S']})

        traj.add_parameter('CD0', val=0.03, opt=False, units=None, static_target=True,
                           desc='zero-lift drag coefficient',
                           targets={f'{phase}': ['CD0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate', 'climb']})

        traj.add_parameter('AR', val=9.45, opt=False, units=None, static_target=True,
                           desc='wing aspect ratio',
                           targets={f'{phase}': ['AR'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                     'rto', 'rotate', 'climb']})

        traj.add_parameter('e', val=801, opt=False, units=None, static_target=True,
                           desc='Oswald span efficiency factor',
                           targets={f'{phase}': ['e'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                    'rto', 'rotate', 'climb']})

        traj.add_parameter('span', val=35.7, opt=False, units='m', static_target=True,
                           desc='wingspan',
                           targets={f'{phase}': ['span'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                       'rto', 'rotate', 'climb']})

        traj.add_parameter('h_w', val=1.0, opt=False, units='m', static_target=True,
                           desc='height of wing above CG',
                           targets={f'{phase}': ['h_w'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate', 'climb']})

        traj.add_parameter('CL0', val=0.5, opt=False, units=None, static_target=True,
                           desc='zero-alpha lift coefficient',
                           targets={f'{phase}': ['CL0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate', 'climb']})

        traj.add_parameter('CL_max', val=2.0, opt=False, units=None, static_target=True,
                           desc='maximum lift coefficient for linear fit',
                           targets={f'{phase}': ['CL_max'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                         'rto', 'rotate', 'climb']})

        # Standard "end of first phase to beginning of second phase" linkages
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

        # Less common "final value of r must be the match at ends of two phases".
        traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                                    phase_b='climb', var_b='r', loc_b='final',
                                    ref=1000,
                                    connected=False)

        # Define the constraints and objective for the optimal control problem
        rto.add_boundary_constraint('v', loc='final', upper=0.001, ref=100, linear=True)

        rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

        rto.add_objective('r', loc='final', ref=1000.0)

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p.set_val('traj.br_to_v1.t_initial', 0)
        p.set_val('traj.br_to_v1.t_duration', 35)
        p.set_val('traj.br_to_v1.states:r', br_to_v1.interp('r', [0, 2500.0]))
        p.set_val('traj.br_to_v1.states:v', br_to_v1.interp('v', [0.0001, 100.0]))
        p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')

        p.set_val('traj.v1_to_vr.t_initial', 35)
        p.set_val('traj.v1_to_vr.t_duration', 35)
        p.set_val('traj.v1_to_vr.states:r', v1_to_vr.interp('r', [2500, 300.0]))
        p.set_val('traj.v1_to_vr.states:v', v1_to_vr.interp('v', [100, 110.0]))
        p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rto.t_initial', 35)
        p.set_val('traj.rto.t_duration', 1)
        p.set_val('traj.rto.states:r', rto.interp('r', [2500, 5000.0]))
        p.set_val('traj.rto.states:v', rto.interp('v', [110, 0.0001]))
        p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rotate.t_initial', 35)
        p.set_val('traj.rotate.t_duration', 5)
        p.set_val('traj.rotate.states:r', rotate.interp('r', [1750, 1800.0]))
        p.set_val('traj.rotate.states:v', rotate.interp('v', [80, 85.0]))
        p.set_val('traj.rotate.controls:alpha', 0.0, units='deg')

        p.set_val('traj.climb.t_initial', 30)
        p.set_val('traj.climb.t_duration', 20)
        p.set_val('traj.climb.states:r', climb.interp('r', [5000, 5500.0]), units='ft')
        p.set_val('traj.climb.states:v', climb.interp('v', [160, 170.0]), units='kn')
        p.set_val('traj.climb.states:h', climb.interp('h', [0, 35.0]), units='ft')
        p.set_val('traj.climb.states:gam', climb.interp('gam', [0, 5.0]), units='deg')
        p.set_val('traj.climb.controls:alpha', 5.0, units='deg')

        create_linkage_report(traj)

    def log_test(self, msg):
        global current_test

        """ Print a description and index for the test about to run. """
        print("  Test {:04}".format(current_test) + ": " + msg)
        current_test += 1

    async def load_test_page(self):
        """ Load the specified HTML file from the local filesystem. """
        url = f'file://{self.tempdir}/linkage_report.html'

        # Without wait_until: 'networkidle', processing will begin before
        # the page is fully rendered
        await self.page.goto(url, wait_until='networkidle')

    async def generic_toolbar_tests(self):
        """ Click most of the toolbar buttons to see if an error occurs """
        for test in toolbar_script:
            with self.subTest(test['desc']):
                self.log_test("[Toolbar] " + test['desc'])

                btnHandle = await self.get_handle('#' + test['id'])
                await btnHandle.click(button='left', timeout=3333, force=True)

        await self.page.reload(wait_until='networkidle')

    async def assert_element_count(self, selector, expected_found):
        """
        Count the number of elements located by the selector and make
        sure it exactly matches the supplied value. Try several times
        because sometimes transition animations throw things off.
        """
        max_tries = 3  # Max number of times to attempt to find a selector
        max_time = 2000  # The timeout in ms for each search

        if (expected_found > 0):
            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, {expected_found})'
                try:
                    await self.page.wait_for_selector(nth_selector, state='attached',
                                                      timeout=max_time)
                    found = True
                except Exception:
                    num_tries += 1

            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, {expected_found + 1})'
                try:
                    await self.page.wait_for_selector(nth_selector, state='detached',
                                                      timeout=max_time)
                    found = True
                except Exception:
                    num_tries += 1

        else:
            num_tries = 0
            found = False
            while (not found and num_tries < max_tries):
                nth_selector = f':nth-match({selector}, 1)'
                try:
                    await self.page.wait_for_selector(nth_selector, state='detached',
                                                      timeout=max_time)
                    found = True
                except Exception:
                    num_tries += 1

        hndl_list = await self.page.query_selector_all(selector)
        if (len(hndl_list) > expected_found):
            global current_test
            await self.page.screenshot(path=f'shot_{current_test}.png')

        self.assertEqual(len(hndl_list), expected_found,
                         'Found ' + str(len(hndl_list)) +
                         ' elements, expected ' + str(expected_found))

    async def assert_arrow_count(self, expected_arrows):
        """
        Count the number of path elements in the n2arrows < div > and make
        sure it matches the specified value.
        """
        await self.assert_element_count('g#n2arrows > g', expected_arrows)

    async def hover(self, options, log_test=True):
        """
        Hover over the specified element.
        """
        if log_test:
            self.log_test(options['desc'] if 'desc' in options else
                          "Hover over '" + options['selector'] + "'")

        hndl = await self.get_handle(options['selector'])

        await hndl.hover(force=False)

    async def hover_and_check_arrow_count(self, options):
        """
        Hover over a matrix cell, make sure the number of expected arrows
        are there, then move off and make sure the arrows go away.
        """
        await self.hover(options)

        # Make sure there are enough arrows
        await self.assert_arrow_count(options['arrowCount'])
        await self.page.mouse.move(0, 0)  # Get the mouse off the element
        await self.assert_arrow_count(0)  # Make sure no arrows left

    async def click(self, options):
        """
        Perform a click of the type specified by options.button on the
        element specified by options.selector.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      options['button'] + "-click on '" +
                      options['selector'] + "'")

        hndl = await self.get_handle(options['selector'])

        mod_keys = [] if 'modifiers' not in options else options['modifiers']
        await hndl.click(button=options['button'], modifiers=mod_keys)

    async def drag(self, options):
        """
        Hover over the element, perform a mousedown event, move the mouse to the
        specified location, and perform a mouseup. Check to make sure the element
        moved in at least one direction.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Dragging '" + options['selector'] + "' to " + options['x'] + "," + options['y'])

        hndl = await self.get_handle(options['selector'])

        pre_drag_bbox = await hndl.bounding_box()

        await hndl.hover(force=True)
        await self.page.mouse.down()
        await self.page.mouse.move(options['x'], options['y'])
        await self.page.mouse.up()

        post_drag_bbox = await hndl.bounding_box()

        moved = ((pre_drag_bbox['x'] != post_drag_bbox['x']) or
                 (pre_drag_bbox['y'] != post_drag_bbox['y']))

        self.assertIsNot(moved, False,
                         "The '" + options['selector'] + "' element did not move.")

    async def resize_window(self, options):
        """
        Drag an edge/corner of a WindowResizable obj and check that the size changed
        or didn't change as expected.
        """
        self.log_test(options['desc'] if 'desc' in options else
                      "Resizing '" + options['selector'] + "' window.")

        win_hndl = await self.get_handle(options['selector'])
        pre_resize_bbox = await win_hndl.bounding_box()

        edge_hndl = await self.get_handle(options['selector'] + ' div.rsz-' + options['side'])
        edge_bbox = await edge_hndl.bounding_box()

        new_x = edge_bbox['x'] + resize_dirs[options['side']][0] * options['distance']
        new_y = edge_bbox['y'] + resize_dirs[options['side']][1] * options['distance']

        await edge_hndl.hover()
        await self.page.mouse.down()
        await self.page.mouse.move(new_x, new_y)
        await self.page.mouse.up()

        post_resize_bbox = await win_hndl.bounding_box()
        dw = post_resize_bbox['width'] - pre_resize_bbox['width']
        dh = post_resize_bbox['height'] - pre_resize_bbox['height']

        resized = ((dw != 0) or (dh != 0))
        if options['expectChange']:
            self.assertIsNot(resized, False,
                             "The '" + options['selector'] + "' element was NOT resized and should have been.")
        else:
            self.assertIsNot(resized, True,
                             "The '" + options['selector'] + "' element was resized and should NOT have been.")

    async def return_to_root(self):
        """
        Left-click the home button and wait for the transition to complete.
        """

        self.log_test("Return to root")
        hndl = await self.get_handle("#reset-graph")
        await hndl.click()

    async def search_and_check_result(self, options):
        """
        Enter a string in the search textbox and check that the expected
        number of elements are shown in the N2 matrix.
        """
        searchString = options['searchString']
        self.log_test(options['desc'] if 'desc' in options else
                      "Searching for '" + options['searchString'] +
                      "' and checking for " +
                      str(options['diagElementCount']) + " diagram elements after.")

        await self.page.click("#searchbar-container")

        searchbar = await self.page.wait_for_selector('#awesompleteId', state='visible')
        await searchbar.type(searchString + "\n", delay=50)

        await self.assert_element_count("g.n2cell", options['diagElementCount'])

    async def var_select_search_and_check_result(self, options):
        """
        Enter a string in the variable selection search textbox and check the result.
        """
        searchString = options['searchString']
        self.log_test(options['desc'] if 'desc' in options else
                      "Searching for '" + options['searchString'] +
                      "' and checking for " +
                      str(options['foundVariableCount']) + " table rows after.")

        searchbar = await self.page.wait_for_selector('.search-container input', state='visible')
        await searchbar.type(searchString + "\n", delay=50)

        await self.assert_element_count("td.varname", options['foundVariableCount'])

    async def run_model_script(self, script):
        """
        Iterate through the supplied script array and perform each
        action/test.
        """

        print("Running tests from model script...")

        for script_item in script:
            if 'test' not in script_item:
                continue

            test_type = script_item['test']
            if test_type == 'hoverArrow':
                await self.hover_and_check_arrow_count(script_item)
            elif test_type == 'hover':
                await self.hover(script_item)
            elif test_type == 'click':
                await self.click(script_item)
            elif test_type == 'drag':
                await self.drag(script_item)
            elif test_type == 'resize':
                await self.resize_window(script_item)
            elif test_type == 'root':
                await self.return_to_root()
            elif test_type == 'search':
                await self.search_and_check_result(script_item)
            elif test_type == 'var_select_search':
                await self.var_select_search_and_check_result(script_item)
            elif test_type == 'toolbar':
                await self.generic_toolbar_tests()
            elif test_type == 'count':
                self.log_test(script_item['desc'] if 'desc' in script_item
                              else "Checking for " + str(script_item['count']) +
                                   "' instances of '" + script_item['selector'] + "'")
                await self.assert_element_count(script_item['selector'],
                                                script_item['count'])

    async def run_gui_tests(self, playwright):
        """ Execute all of the tests in an async event loop. """
        await self.setup_browser(playwright)
        await self.load_test_page()
        await self.run_model_script(gui_test_script)

        await self.browser.close()

        if self.console_error:
            msg = "Console log contains errors."
            print(msg)
            self.fail(msg)

        if self.page_error:
            msg = "There were errors on the page."
            print(msg)
            self.fail(msg)

    @async_test(loop=my_loop)
    async def test_gui(self):
        print("\n" + '-' * 78 + "\n" + '-' * 78 + "\n" + '-' * 78)

        self.current_test_desc = ''
        self.current_model = 'bfl'

        async with async_playwright() as playwright:
            await self.run_gui_tests(playwright)
