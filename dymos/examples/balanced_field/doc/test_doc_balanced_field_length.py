import unittest

import numpy as np
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
import dymos as dm
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp


regression_data = {}
regression_data[dm.Radau] = {
    "constraints": {
        "traj.linkages.br_to_v1:time_final|v1_to_vr:time_initial": [0.0],
        "traj.linkages.br_to_v1:r_final|v1_to_vr:r_initial": [0.0],
        "traj.linkages.br_to_v1:v_final|v1_to_vr:v_initial": [0.0],
        "traj.linkages.v1_to_vr:time_final|rotate:time_initial": [0.0],
        "traj.linkages.v1_to_vr:r_final|rotate:r_initial": [-1450.0],
        "traj.linkages.v1_to_vr:v_final|rotate:v_initial": [30.0],
        "traj.linkages.v1_to_vr:alpha_final|rotate:alpha_initial": [0.0],
        "traj.linkages.rotate:time_final|climb:time_initial": [0.0],
        "traj.linkages.rotate:r_final|climb:r_initial": [276.0],
        "traj.linkages.rotate:v_final|climb:v_initial": [2.688888888888883],
        "traj.linkages.rotate:alpha_final|climb:alpha_initial": [-5.0],
        "traj.linkages.br_to_v1:time_final|rto:time_initial": [0.0],
        "traj.linkages.br_to_v1:r_final|rto:r_initial": [0.0],
        "traj.linkages.br_to_v1:v_final|rto:v_initial": [-10.0],
        "traj.linkages.rto:r_final|climb:r_final": [3.3236],
        "traj.br_to_v1.collocation_constraint.defects:r": [
            0.4166666666666665,
            0.3476289672207841,
            0.2523710327792159,
            0.22222222222222143,
            0.1531845227763402,
            0.05792658833477156,
            0.027777777777776274,
            -0.04125992166810405,
            -0.1365178561096728,
        ],
        "traj.br_to_v1.collocation_constraint.defects:v": [
            0.00649795093524439,
            0.006616445486851315,
            0.007169036357321012,
            0.007437927094391042,
            0.0082239006449162,
            0.009697476299502095,
            0.010257855571831287,
            0.01171130812127473,
            0.01410586855997708,
        ],
        "traj.climb.h[final]": [1.0],
        "traj.climb.gam[final]": [1.0],
        "traj.climb.v_over_v_stall[final]": [0.9823389802049667],
        "traj.climb.gam[path]": [
            0.0,
            0.07101020514433644,
            0.1689897948556636,
            0.20000000000000004,
            0.20000000000000004,
            0.27101020514433644,
            0.36898979485566363,
            0.4000000000000001,
            0.4000000000000001,
            0.4710102051443366,
            0.5689897948556638,
            0.6000000000000002,
            0.6000000000000002,
            0.6710102051443366,
            0.7689897948556637,
            0.8000000000000002,
            0.8000000000000002,
            0.8710102051443366,
            0.9689897948556637,
        ],
        "traj.climb.h[path]": [
            0.0,
            2.4853571800517744,
            5.914642819948225,
            6.999999999999999,
            6.999999999999999,
            9.485357180051773,
            12.914642819948225,
            13.999999999999998,
            13.999999999999998,
            16.48535718005178,
            19.91464281994823,
            21.0,
            21.0,
            23.48535718005178,
            26.914642819948227,
            27.999999999999996,
            27.999999999999996,
            30.485357180051775,
            33.91464281994823,
        ],
        "traj.climb.collocation_constraint.defects:r": [
            -0.10822666666666667,
            -0.10877224764873458,
            -0.10951713737628663,
            -0.1097509603715007,
            -0.11028284972038041,
            -0.11100855808579062,
            -0.11123624046202378,
            -0.11175394240945394,
            -0.11245978793155431,
            -0.11268111463694755,
            -0.11318413790684131,
            -0.11386944544945637,
            -0.1140842034756845,
            -0.1145720615353963,
            -0.11523616265315353,
        ],
        "traj.climb.collocation_constraint.defects:h": [
            1.0667999999999995,
            0.29830989982523587,
            -0.7731445565941426,
            -1.1149253287248142,
            -1.902381792745922,
            -2.9998640073265532,
            -3.349847639357176,
            -4.156023740270561,
            -5.279186548509148,
            -5.637261541533632,
            -6.461899026513403,
            -7.610379407812043,
            -7.976429256188403,
            -8.81925842906678,
            -9.992677623677165,
        ],
        "traj.climb.collocation_constraint.defects:v": [
            -0.01461939424215987,
            -0.013680311182054173,
            -0.012385122427714935,
            -0.01197529115802313,
            -0.01103693473646186,
            -0.009742414304801463,
            -0.0093327538694019,
            -0.008394772109415277,
            -0.007100805084518509,
            -0.006691339937664971,
            -0.005753853875203694,
            -0.00446069768091129,
            -0.004051522420108261,
            -0.0031147634445885157,
            -0.0018227595387125509,
        ],
        "traj.climb.collocation_constraint.defects:gam": [
            0.716834308339309,
            0.6879445904659692,
            0.6480853509270524,
            0.6354706790667543,
            0.6065855235688165,
            0.566732541890516,
            0.5541198471763983,
            0.5252392227663774,
            0.48539252338321354,
            0.4727818295474824,
            0.4439058189556195,
            0.4040655795810557,
            0.3914569579135891,
            0.3625857510819955,
            0.32275229358217394,
        ],
        "traj.climb.continuity_comp.defect_control_rates:alpha_rate": [
            2.220446049250313e-15,
            2.220446049250313e-15,
            2.2204460492503127e-15,
            2.220446049250314e-15,
        ],
        "traj.climb.continuity_comp.defect_controls:alpha": [0.0, 0.0, 0.0, 0.0],
        "traj.rotate.F_r[final]": [4.989619710033791],
        "traj.rotate.collocation_constraint.defects:r": [
            -0.05833333333333443,
            -0.058826459757946446,
            -0.059506873575385956,
            -0.05972222222222309,
            -0.0602153486468354,
            -0.060895762464275166,
            -0.06111111111111143,
            -0.06160423753572436,
            -0.06228465135316393,
        ],
        "traj.rotate.collocation_constraint.defects:v": [
            -0.0011079918722422847,
            -0.001096507055634512,
            -0.00108052139116765,
            -0.0010754284124435546,
            -0.001063705210478725,
            -0.0010473906228746822,
            -0.0010421935411017993,
            -0.0010302319537806714,
            -0.0010135884430394586,
        ],
        "traj.rto.v[final]": [0.0],
        "traj.rto.collocation_constraint.defects:r": [
            -0.22500000000000214,
            -0.1490585306095287,
            -0.04427480272380285,
            -0.011111111111113143,
            0.06483035827935983,
            0.1696140861650864,
            0.2027777777777753,
            0.2787192471682481,
            0.3835029750539747,
        ],
        "traj.rto.collocation_constraint.defects:v": [
            -0.09358803170582275,
            -0.07535584468522619,
            -0.05396466020985954,
            -0.048104102054440434,
            -0.03633156138447398,
            -0.023853370440509947,
            -0.020813744263609962,
            -0.01550084994427481,
            -0.011935652531713638,
        ],
        "traj.v1_to_vr.v_over_v_stall[final]": [0.015444600279553376],
        "traj.v1_to_vr.collocation_constraint.defects:r": [
            -0.950000000000002,
            -0.9569037699445878,
            -0.9664295633887448,
            -0.969444444444446,
            -0.9763482143890325,
            -0.9858740078331893,
            -0.9888888888888886,
            -0.9957926588334773,
            -1.0053184522776337,
        ],
        "traj.v1_to_vr.collocation_constraint.defects:v": [
            -0.04637708701672346,
            -0.04617565837153162,
            -0.045893837027591915,
            -0.04580370155964456,
            -0.04559559812446296,
            -0.04530456693268199,
            -0.04521151657938215,
            -0.044996738354211245,
            -0.044696497314589134,
        ],
    },
    "design_vars": {
        "traj.br_to_v1.t_duration": [3.5],
        "traj.br_to_v1.states:r": [
            0.29587585476806844,
            0.7041241452319315,
            0.8333333333333333,
            1.1292091881014017,
            1.5374574785652648,
            1.6666666666666665,
            1.962542521434735,
            2.3707908118985985,
            2.5,
        ],
        "traj.br_to_v1.states:v": [
            0.11835034190722737,
            0.2816496580927726,
            0.3333333333333333,
            0.4516836752405607,
            0.6149829914261059,
            0.6666666666666666,
            0.785017008573894,
            0.9483163247594394,
            1.0,
        ],
        "traj.climb.t_initial": [75.0],
        "traj.climb.t_duration": [15.0],
        "traj.climb.states:r": [
            1.524,
            1.5348219552639972,
            1.5497540447360032,
            1.55448,
            1.565301955263997,
            1.5802340447360035,
            1.5849600000000001,
            1.5957819552639971,
            1.6107140447360035,
            1.61544,
            1.6262619552639968,
            1.6411940447360034,
            1.64592,
            1.6567419552639973,
            1.6716740447360035,
            1.6764000000000001,
        ],
        "traj.climb.states:h": [
            0.757536868479781,
            1.802783131520219,
            2.1336,
            2.891136868479781,
            3.9363831315202193,
            4.2672,
            5.024736868479782,
            6.069983131520221,
            6.400800000000001,
            7.158336868479783,
            8.20358313152022,
            8.5344,
            9.291936868479782,
            10.33718313152022,
            10.668000000000001,
        ],
        "traj.climb.states:v": [
            0.8231111111111112,
            0.8267641916646475,
            0.8318046972242413,
            0.8334,
            0.8370530805535364,
            0.8420935861131303,
            0.8436888888888889,
            0.8473419694424255,
            0.8523824750020192,
            0.853977777777778,
            0.8576308583313142,
            0.8626713638909083,
            0.8642666666666667,
            0.8679197472202032,
            0.872960252779797,
            0.8745555555555556,
        ],
        "traj.climb.states:gam": [
            0.12393618822852859,
            0.2949428322501105,
            0.3490658503988659,
            0.4730020386273945,
            0.6440086826489765,
            0.6981317007977318,
            0.8220678890262606,
            0.9930745330478425,
            1.0471975511965979,
            1.1711337394251264,
            1.3421403834467083,
            1.3962634015954636,
            1.5201995898239922,
            1.6912062338455742,
            1.7453292519943295,
        ],
        "traj.climb.controls:alpha": [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        "traj.rotate.t_initial": [70.0],
        "traj.rotate.t_duration": [5.0],
        "traj.rotate.states:r": [
            1.75,
            1.7559175170953614,
            1.7640824829046386,
            1.7666666666666668,
            1.772584183762028,
            1.7807491495713053,
            1.7833333333333332,
            1.7892508504286948,
            1.797415816237972,
            1.8,
        ],
        "traj.rotate.states:v": [
            0.8,
            0.8059175170953614,
            0.8140824829046388,
            0.8166666666666668,
            0.822584183762028,
            0.8307491495713053,
            0.8333333333333333,
            0.8392508504286947,
            0.847415816237972,
            0.85,
        ],
        "traj.rotate.controls:alpha": [0.0, 0.0],
        "traj.rto.t_initial": [35.0],
        "traj.rto.t_duration": [35.0],
        "traj.rto.states:r": [
            2.5,
            2.7958758547680684,
            3.2041241452319316,
            3.3333333333333335,
            3.629209188101402,
            4.037457478565265,
            4.166666666666667,
            4.462542521434735,
            4.870790811898598,
            5.0,
        ],
        "traj.rto.states:v": [
            1.1,
            0.96981462390205,
            0.7901853760979503,
            0.7333333333333334,
            0.6031479572353832,
            0.4235187094312835,
            0.3666666666666667,
            0.2364812905687166,
            0.05685204276461679,
            0.0,
        ],
        "traj.v1_to_vr.t_initial": [35.0],
        "traj.v1_to_vr.t_duration": [35.0],
        "traj.v1_to_vr.states:r": [
            2.5,
            2.2396292478040998,
            1.8803707521959006,
            1.7666666666666668,
            1.5062959144707664,
            1.147037418862567,
            1.0333333333333332,
            0.7729625811374332,
            0.4137040855292336,
            0.3,
        ],
        "traj.v1_to_vr.states:v": [
            1.0,
            1.0118350341907227,
            1.0281649658092773,
            1.0333333333333334,
            1.0451683675240562,
            1.0614982991426107,
            1.0666666666666667,
            1.0785017008573894,
            1.094831632475944,
            1.1,
        ],
    },
    "objectives": {"traj.rto.states:r": [5000.0]},
}

regression_data[dm.GaussLobatto] = {
    "constraints": {
        "traj.linkages.br_to_v1:time_final|v1_to_vr:time_initial": [0.0],
        "traj.linkages.br_to_v1:r_final|v1_to_vr:r_initial": [0.0],
        "traj.linkages.br_to_v1:v_final|v1_to_vr:v_initial": [0.0],
        "traj.linkages.v1_to_vr:time_final|rotate:time_initial": [0.0],
        "traj.linkages.v1_to_vr:r_final|rotate:r_initial": [-1450.0],
        "traj.linkages.v1_to_vr:v_final|rotate:v_initial": [30.0],
        "traj.linkages.v1_to_vr:alpha_final|rotate:alpha_initial": [0.0],
        "traj.linkages.rotate:time_final|climb:time_initial": [0.0],
        "traj.linkages.rotate:r_final|climb:r_initial": [276.0],
        "traj.linkages.rotate:v_final|climb:v_initial": [2.688888888888883],
        "traj.linkages.rotate:alpha_final|climb:alpha_initial": [-5.0],
        "traj.linkages.br_to_v1:time_final|rto:time_initial": [0.0],
        "traj.linkages.br_to_v1:r_final|rto:r_initial": [0.0],
        "traj.linkages.br_to_v1:v_final|rto:v_initial": [-10.0],
        "traj.linkages.rto:r_final|climb:r_final": [3.3236],
        "traj.br_to_v1.collocation_constraint.defects:r": [
            0.47902958681012453,
            0.1870887604303732,
            -0.10485206594937782,
        ],
        "traj.br_to_v1.collocation_constraint.defects:v": [
            0.010217577615992882,
            0.013042811161855497,
            0.018693278253580733,
        ],
        "traj.climb.h[final]": [1.0],
        "traj.climb.gam[final]": [1.0],
        "traj.climb.v_over_v_stall[final]": [0.9823389802049667],
        "traj.climb.gam[path]": [
            0.0,
            0.0883455185920389,
            0.20000000000000004,
            0.20000000000000004,
            0.2883473516820108,
            0.4000000000000001,
            0.4000000000000001,
            0.4883491871897563,
            0.6000000000000002,
            0.6000000000000002,
            0.688351070214837,
            0.8000000000000002,
            0.8000000000000002,
            0.8883530431698609,
        ],
        "traj.climb.h[path]": [
            0.0,
            1.7105271253897536,
            6.999999999999999,
            6.999999999999999,
            8.666894430255608,
            13.999999999999998,
            13.999999999999998,
            15.623840303332958,
            21.0,
            21.0,
            22.581391310158487,
            27.999999999999996,
            27.999999999999996,
            29.539573821693008,
        ],
        "traj.climb.collocation_constraint.defects:r": [
            -0.1635881987944115,
            -0.1658477115826216,
            -0.16804769765480093,
            -0.17018604309853758,
            -0.17226068025412108,
        ],
        "traj.climb.collocation_constraint.defects:h": [
            0.09618576144272717,
            -3.2166033112704633,
            -6.608701632651881,
            -10.079026976699794,
            -13.626448340964746,
        ],
        "traj.climb.collocation_constraint.defects:v": [
            -0.020091291412078167,
            -0.016125860961723488,
            -0.01216255501671909,
            -0.008201285233091979,
            -0.00424289445112884,
        ],
        "traj.climb.collocation_constraint.defects:gam": [
            1.0090664482860263,
            0.8871644579785876,
            0.7652774854957041,
            0.643407653401527,
            0.5215561354920641,
        ],
        "traj.climb.continuity_comp.defect_control_rates:alpha_rate": [0.0, 0.0, 0.0, 0.0],
        "traj.rotate.F_r[final]": [4.989619710033791],
        "traj.rotate.collocation_constraint.defects:r": [
            -0.08854234507207909,
            -0.090625692393153,
            -0.09270903971422675,
        ],
        "traj.rotate.collocation_constraint.defects:v": [
            -0.0016376332341922898,
            -0.0015882838231674338,
            -0.0015379272813052295,
        ],
        "traj.rto.v[final]": [0.0],
        "traj.rto.collocation_constraint.defects:r": [
            -0.18371640640749326,
            0.13977015615550417,
            0.4632567187185013,
        ],
        "traj.rto.collocation_constraint.defects:v": [
            -0.1054141896634857,
            -0.04992513115327707,
            -0.022180601898172673,
        ],
        "traj.v1_to_vr.v_over_v_stall[final]": [0.015444600279553376],
        "traj.v1_to_vr.collocation_constraint.defects:r": [
            -1.439666952045824,
            -1.4688363603096213,
            -1.4980057685734194,
        ],
        "traj.v1_to_vr.collocation_constraint.defects:v": [
            -0.06913547541722186,
            -0.06826113323636616,
            -0.06735858646903126,
        ],
    },
    "design_vars": {
        "traj.br_to_v1.t_duration": [3.5],
        "traj.br_to_v1.states:r": [0.8333333333333333, 1.6666666666666665, 2.5],
        "traj.br_to_v1.states:v": [0.3333333333333333, 0.6666666666666666, 1.0],
        "traj.climb.t_initial": [75.0],
        "traj.climb.t_duration": [15.0],
        "traj.climb.states:r": [
            1.524,
            1.55448,
            1.5849600000000001,
            1.61544,
            1.64592,
            1.6764000000000001,
        ],
        "traj.climb.states:h": [2.1336, 4.2672, 6.400800000000001, 8.5344, 10.668000000000001],
        "traj.climb.states:v": [
            0.8231111111111112,
            0.8334,
            0.8436888888888889,
            0.853977777777778,
            0.8642666666666667,
            0.8745555555555556,
        ],
        "traj.climb.states:gam": [
            0.3490658503988659,
            0.6981317007977318,
            1.0471975511965979,
            1.3962634015954636,
            1.7453292519943295,
        ],
        "traj.climb.controls:alpha": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "traj.rotate.t_initial": [70.0],
        "traj.rotate.t_duration": [5.0],
        "traj.rotate.states:r": [1.75, 1.7666666666666668, 1.7833333333333332, 1.8],
        "traj.rotate.states:v": [0.8, 0.8166666666666668, 0.8333333333333333, 0.85],
        "traj.rotate.controls:alpha": [0.0, 0.0],
        "traj.rto.t_initial": [35.0],
        "traj.rto.t_duration": [35.0],
        "traj.rto.states:r": [2.5, 3.3333333333333335, 4.166666666666667, 5.0],
        "traj.rto.states:v": [1.1, 0.7333333333333334, 0.3666666666666667, 0.0],
        "traj.v1_to_vr.t_initial": [35.0],
        "traj.v1_to_vr.t_duration": [35.0],
        "traj.v1_to_vr.states:r": [2.5, 1.7666666666666668, 1.0333333333333332, 0.3],
        "traj.v1_to_vr.states:v": [1.0, 1.0333333333333334, 1.0666666666666667, 1.1],
    },
    "objectives": {"traj.rto.states:r": [5000.0]},
}


@use_tempdirs
class TestBalancedFieldLengthForDocs(unittest.TestCase):
    def _make_problem(self, tx, optimizer='IPOPT'):
        p = om.Problem()

        # Use IPOPT if available, with fallback to SLSQP
        if optimizer is not None:
            p.driver = om.pyOptSparseDriver(optimizer="IPOPT")
            p.driver.options["optimizer"] = optimizer
            p.driver.declare_coloring()
            p.driver.options["print_results"] = False
            if optimizer == "IPOPT":
                p.driver.opt_settings["print_level"] = 0

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(
            ode_class=BalancedFieldODEComp,
            transcription=tx(num_segments=3),
            ode_init_kwargs={"mode": "runway"},
        )
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)
        br_to_v1.add_state("r", fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)
        br_to_v1.add_state("v", fix_initial=True, lower=0, ref=100.0, defect_ref=100.0)
        br_to_v1.add_parameter("alpha", val=0.0, opt=False, units="deg")
        br_to_v1.add_timeseries_output("*")

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(
            ode_class=BalancedFieldODEComp,
            transcription=tx(num_segments=3),
            ode_init_kwargs={"mode": "runway"},
        )
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state("r", fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rto.add_state("v", fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rto.add_parameter("alpha", val=0.0, opt=False, units="deg")
        rto.add_timeseries_output("*")

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(
            ode_class=BalancedFieldODEComp,
            transcription=tx(num_segments=3),
            ode_init_kwargs={"mode": "runway"},
        )
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state("r", fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        v1_to_vr.add_state("v", fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        v1_to_vr.add_parameter("alpha", val=0.0, opt=False, units="deg")
        v1_to_vr.add_timeseries_output("*")

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(
            ode_class=BalancedFieldODEComp,
            transcription=tx(num_segments=3),
            ode_init_kwargs={"mode": "runway"},
        )
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
        rotate.add_state("r", fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rotate.add_state("v", fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rotate.add_control(
            "alpha",
            order=1,
            opt=True,
            units="deg",
            lower=0,
            upper=10,
            ref=10,
            val=[0, 10],
            control_type="polynomial",
        )
        rotate.add_timeseries_output("*")

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(
            ode_class=BalancedFieldODEComp,
            transcription=tx(num_segments=5),
            ode_init_kwargs={"mode": "climb"},
        )
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)
        climb.add_state("r", fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        climb.add_state("h", fix_initial=True, lower=0, ref=1.0, defect_ref=1.0)
        climb.add_state("v", fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        climb.add_state("gam", fix_initial=True, lower=0, ref=0.05, defect_ref=0.05)
        climb.add_control("alpha", opt=True, units="deg", lower=-10, upper=15, ref=10)
        climb.add_timeseries_output("*")

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem("traj", traj)
        traj.add_phase("br_to_v1", br_to_v1)
        traj.add_phase("rto", rto)
        traj.add_phase("v1_to_vr", v1_to_vr)
        traj.add_phase("rotate", rotate)
        traj.add_phase("climb", climb)

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter(
            "m",
            val=174200.0,
            opt=False,
            units="lbm",
            desc="aircraft mass",
            targets={
                "br_to_v1": ["m"],
                "v1_to_vr": ["m"],
                "rto": ["m"],
                "rotate": ["m"],
                "climb": ["m"],
            },
        )

        traj.add_parameter(
            "T_nominal",
            val=27000 * 2,
            opt=False,
            units="lbf",
            static_target=True,
            desc="nominal aircraft thrust",
            targets={"br_to_v1": ["T"]},
        )

        traj.add_parameter(
            "T_engine_out",
            val=27000,
            opt=False,
            units="lbf",
            static_target=True,
            desc="thrust under a single engine",
            targets={"v1_to_vr": ["T"], "rotate": ["T"], "climb": ["T"]},
        )

        traj.add_parameter(
            "T_shutdown",
            val=0.0,
            opt=False,
            units="lbf",
            static_target=True,
            desc="thrust when engines are shut down for rejected takeoff",
            targets={"rto": ["T"]},
        )

        traj.add_parameter(
            "mu_r_nominal",
            val=0.03,
            opt=False,
            units=None,
            static_target=True,
            desc="nominal runway friction coefficient",
            targets={"br_to_v1": ["mu_r"], "v1_to_vr": ["mu_r"], "rotate": ["mu_r"]},
        )

        traj.add_parameter(
            "mu_r_braking",
            val=0.3,
            opt=False,
            units=None,
            static_target=True,
            desc="runway friction coefficient under braking",
            targets={"rto": ["mu_r"]},
        )

        traj.add_parameter(
            "h_runway",
            val=0.0,
            opt=False,
            units="ft",
            desc="runway altitude",
            targets={"br_to_v1": ["h"], "v1_to_vr": ["h"], "rto": ["h"], "rotate": ["h"]},
        )

        traj.add_parameter(
            "rho",
            val=1.225,
            opt=False,
            units="kg/m**3",
            static_target=True,
            desc="atmospheric density",
            targets={"br_to_v1": ["rho"], "v1_to_vr": ["rho"], "rto": ["rho"], "rotate": ["rho"]},
        )

        traj.add_parameter(
            "S",
            val=124.7,
            opt=False,
            units="m**2",
            static_target=True,
            desc="aerodynamic reference area",
            targets={
                "br_to_v1": ["S"],
                "v1_to_vr": ["S"],
                "rto": ["S"],
                "rotate": ["S"],
                "climb": ["S"],
            },
        )

        traj.add_parameter(
            "CD0",
            val=0.03,
            opt=False,
            units=None,
            static_target=True,
            desc="zero-lift drag coefficient",
            targets={
                f"{phase}": ["CD0"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "AR",
            val=9.45,
            opt=False,
            units=None,
            static_target=True,
            desc="wing aspect ratio",
            targets={
                f"{phase}": ["AR"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "e",
            val=801,
            opt=False,
            units=None,
            static_target=True,
            desc="Oswald span efficiency factor",
            targets={
                f"{phase}": ["e"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "span",
            val=35.7,
            opt=False,
            units="m",
            static_target=True,
            desc="wingspan",
            targets={
                f"{phase}": ["span"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "h_w",
            val=1.0,
            opt=False,
            units="m",
            static_target=True,
            desc="height of wing above CG",
            targets={
                f"{phase}": ["h_w"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "CL0",
            val=0.5,
            opt=False,
            units=None,
            static_target=True,
            desc="zero-alpha lift coefficient",
            targets={
                f"{phase}": ["CL0"] for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "CL_max",
            val=2.0,
            opt=False,
            units=None,
            static_target=True,
            desc="maximum lift coefficient for linear fit",
            targets={
                f"{phase}": ["CL_max"]
                for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        traj.add_parameter(
            "alpha_max",
            val=10.0,
            opt=False,
            units="deg",
            static_target=True,
            desc="angle of attack at maximum lift",
            targets={
                f"{phase}": ["alpha_max"]
                for phase in ["br_to_v1", "v1_to_vr", "rto", "rotate", "climb"]
            },
        )

        # Standard "end of first phase to beginning of second phase" linkages
        # Alpha changes from being a parameter in v1_to_vr to a polynomial control
        # in rotate, to a dynamic control in `climb`.
        traj.link_phases(["br_to_v1", "v1_to_vr"], vars=["time", "r", "v"])
        traj.link_phases(["v1_to_vr", "rotate"], vars=["time", "r", "v", "alpha"])
        traj.link_phases(["rotate", "climb"], vars=["time", "r", "v", "alpha"])
        traj.link_phases(["br_to_v1", "rto"], vars=["time", "r", "v"])

        # Less common "final value of r must be the match at ends of two phases".
        traj.add_linkage_constraint(
            phase_a="rto",
            var_a="r",
            loc_a="final",
            phase_b="climb",
            var_b="r",
            loc_b="final",
            ref=1000,
        )

        # Define the constraints and objective for the optimal control problem
        v1_to_vr.add_boundary_constraint("v_over_v_stall", loc="final", lower=1.2, ref=100)

        rto.add_boundary_constraint("v", loc="final", equals=0.0, ref=100, linear=True)

        rotate.add_boundary_constraint("F_r", loc="final", equals=0, ref=100000)

        climb.add_boundary_constraint("h", loc="final", equals=35, ref=35, units="ft", linear=True)
        climb.add_boundary_constraint("gam", loc="final", equals=5, ref=5, units="deg", linear=True)
        climb.add_path_constraint("gam", lower=0, upper=5, ref=5, units="deg")
        climb.add_path_constraint("h", lower=0, upper=35, ref=1.0, units="ft")
        climb.add_boundary_constraint("v_over_v_stall", loc="final", lower=1.25, ref=1.25)

        rto.add_objective("r", loc="final", ref=1.0)

        for phase_name, phase in traj._phases.items():
            if "T_nominal" in phase.parameter_options:
                phase.add_timeseries_output("T_nominal", output_name="T")
            if "T_engine_out" in phase.parameter_options:
                phase.add_timeseries_output("T_engine_out", output_name="T")
            if "T_shutdown" in phase.parameter_options:
                phase.add_timeseries_output("T_shutdown", output_name="T")
            phase.add_timeseries_output("alpha")

        #
        # Setup the problem and set the initial guess
        #
        p.setup()

        br_to_v1.set_time_val(initial=0.0, duration=35.0)
        br_to_v1.set_state_val("r", [0, 2500.0])
        br_to_v1.set_state_val("v", [0.0, 100.0])
        br_to_v1.set_parameter_val("alpha", 0.0, units="deg")

        v1_to_vr.set_time_val(initial=35.0, duration=35.0)
        v1_to_vr.set_state_val("r", [2500, 300.0])
        v1_to_vr.set_state_val("v", [100, 110.0])
        v1_to_vr.set_parameter_val("alpha", 0.0, units="deg")

        rto.set_time_val(initial=35.0, duration=35.0)
        rto.set_state_val("r", [2500, 5000.0])
        rto.set_state_val("v", [110, 0.0])
        rto.set_parameter_val("alpha", 0.0, units="deg")

        rotate.set_time_val(initial=70.0, duration=5.0)
        rotate.set_state_val("r", [1750, 1800.0])
        rotate.set_state_val("v", [80, 85.0])
        rotate.set_control_val("alpha", 0.0, units="deg")

        climb.set_time_val(initial=75.0, duration=15.0)
        climb.set_state_val("r", [5000, 5500.0], units="ft")
        climb.set_state_val("v", [160, 170.0], units="kn")
        climb.set_state_val("h", [0.0, 35.0], units="ft")
        climb.set_state_val("gam", [0.0, 5.0], units="deg")
        climb.set_control_val("alpha", 5.0, units="deg")

        return p

    @require_pyoptsparse(optimizer="IPOPT")
    def test_balanced_field_length_for_docs(self):
        for tx in (dm.Radau, dm.GaussLobatto):
            p = self._make_problem(tx, optimizer='IPOPT')

            traj = p.model.traj

            result = dm.run_problem(p, run_driver=True, simulate=True)

            sol_db = p.get_outputs_dir() / "dymos_solution.db"
            sim_db = traj.sim_prob.get_outputs_dir() / "dymos_simulation.db"

            sol = om.CaseReader(sol_db).get_case("final")
            sim = om.CaseReader(sim_db).get_case("final")

            sol_r_f_climb = sol.get_val("traj.climb.timeseries.r")[-1, ...]
            sol_r_f_rto = sol.get_val("traj.rto.timeseries.r")[-1, ...]
            sim_r_f_climb = sim.get_val("traj.climb.timeseries.r")[-1, ...]
            sim_r_f_rto = sim.get_val("traj.rto.timeseries.r")[-1, ...]

            self.assertTrue(result["success"])
            assert_near_equal(2114.387, sol_r_f_climb, tolerance=0.01)
            assert_near_equal(2114.387, sol_r_f_rto, tolerance=0.01)
            assert_near_equal(2114.387, sim_r_f_climb, tolerance=0.01)
            assert_near_equal(2114.387, sim_r_f_rto, tolerance=0.01)

    def test_no_regression(self):
        """Test that there are no regressions in the initial values of the driver vars."""
        import json

        # For now we only do this with GaussLobatto until we remove the legacy radau method.
        for tx in (dm.GaussLobatto,):
            p = self._make_problem(tx, optimizer=None)
            dm.run_problem(p, run_driver=False, simulate=False)
            driver_vars = p.list_driver_vars(out_stream=None)

            desvars = {}
            constraints = {}
            objs = {}

            for name, meta in driver_vars["constraints"]:
                constraints[name] = meta["val"].tolist()

            for name, meta in driver_vars["design_vars"]:
                desvars[name] = meta["val"].tolist()

            for name, meta in driver_vars["objectives"]:
                objs[name] = meta["val"].tolist()

            vars = {"constraints": constraints, "design_vars": desvars, "objectives": objs}

            # If we change dymos we might have to regenerate the regression
            # data. Dump the JSON and save it in the regression variables
            # defined above.
            # print(json.dumps(vars, indent='    '))

            reg_data = regression_data[tx]

            errors = {}

            for var, val in vars["constraints"].items():
                try:
                    assert_near_equal(
                        np.asarray(val), reg_data["constraints"][var], tolerance=1.0e-9
                    )
                except ValueError as e:
                    errors[var] = {"actual": val, "expected": reg_data["constraints"][var]}

            for var, val in vars["design_vars"].items():
                try:
                    assert_near_equal(
                        np.asarray(val), reg_data["design_vars"][var], tolerance=1.0e-9
                    )
                except ValueError as e:
                    errors[var] = {"actual": val, "expected": reg_data["design_vars"][var]}

            for var, val in vars["objectives"].items():
                try:
                    assert_near_equal(
                        np.asarray(val), reg_data["objectives"][var], tolerance=1.0e-9
                    )
                except ValueError as e:
                    errors[var] = {"actual": val, "expected": reg_data["objectives"][var]}

            if errors:
                msg = f"Outputs for transcription {str(tx.__name__)} showed a regression.\n"
                msg += json.dumps(errors, indent="  ")
                self.fail(msg)


if __name__ == "__main__":
    unittest.main()
