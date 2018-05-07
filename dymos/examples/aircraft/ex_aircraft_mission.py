from __future__ import division, print_function, absolute_import

__author__ = "rfalck"

from openmdao.api import Problem, Group, ScipyOptimizer
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    import pyoptsparse.pySNOPT.snopt as snopt
except ImportError:
    snopt = None

try:
    from openmdao.core.petsc_impl import PetscImpl
except ImportError as e:
    PetscImpl = None

from openmdao.core.basic_impl import BasicImpl

from pointer.core import Trajectory, RHS, CollocationPhase

from .aerodynamics import setup_surrogates_all
from .atmosphere import StandardAtmosphereGroup
from .velocity_comp import TrueAirspeedComp
from .flight_path_angle_comp_deprecated import FlightPathAngleComp
from .dynamic_pressure_comp import DynamicPressureComp
from .flight_equilibrium import FlightEquilibriumAnalysisGroup
from .propulsion import MaxThrustComp, ThrottleSettingComp, SFCComp
from .aircraft_eom import MissionEOM
from .aircraft_models import CRM



def mission(aircraft="CRM", num_seg=10, seg_ncn=2, rel_lengths="lgl",
            solver='SLSQP', aggregate_tau_constraint=False,
            variable_mach=False, distributed=True):

        if distributed and PetscImpl:
            impl = PetscImpl
        else:
            impl = BasicImpl

        prob = Problem(root=Group(), impl=impl)
        prob.root.add("traj0", Trajectory())

        # dynamic_controls = [ {"name":"M","units":"unitless"},  # Mach Number
        #                      {"name":"h","units":"km"} ]       # Altitude
        #
        # static_controls = [ {"name":"S", "units":"m**2"},
        #                     {"name":"W_p", "units":"N"},
        #                     {"name":"W_e", "units":"N"},
        #                     {"name":"SFC_SL", "units":"1/s"},
        #                     {"name":"max_thrust_sl", "units":"N"} ]



        mbi_CL, mbi_CD, mbi_CM, mbi_num = setup_surrogates_all(CRM['name'])
        mbi_CL.seterr(bounds='warn')
        mbi_CD.seterr(bounds='warn')
        mbi_CM.seterr(bounds='warn')

        phase0 = CollocationPhase(name='phase0',rhs_class=MissionRHS,num_seg=num_seg,seg_ncn=seg_ncn,rel_lengths=rel_lengths,
                                  rhs_init_args={'mbi_CL':mbi_CL,'mbi_CD':mbi_CD,'mbi_CM':mbi_CM,'mbi_num':mbi_num},
                                  distributed=distributed)

        prob.find_subsystem('traj0').add_phase(phase0)

        phase0.set_state_options('r', val=phase0.cardinal_node_space(0,1296.4), lower=0,upper=2000.4,ic_val=0,ic_fix=True,fc_val=1296.4,fc_fix=True,scaler=0.01,defect_scaler=0.1)
        phase0.set_state_options('W_f', val=phase0.cardinal_node_space(1.2E5,0), lower=0.0,upper=1.0E6,ic_fix=False,fc_fix=True,scaler=0.0001,defect_scaler=0.001)

        phase0.add_dynamic_control('M', val=phase0.node_space(0.80,0.80), opt=variable_mach, lower=0.2, upper=0.8,scaler=1.0,seg_rate_continuity=True)
        phase0.add_dynamic_control('alt', units='km', val=phase0.node_space(0.0,0.0),opt=True,ic_fix=True, fc_fix=True,lower=0.0,upper=20.0,scaler=1.0,seg_rate_continuity=True)


        pax_wt = 84.02869
        num_pax = 400
        g = 9.80665
        W_p = pax_wt * num_pax * g

        phase0.add_static_control(name="S", units='m**2', val=CRM["S"],opt=False)
        phase0.add_static_control(name="W_e", units='N', val=CRM["mass"]*g,opt=False)
        phase0.add_static_control(name="W_p", units='N', val=W_p,opt=False)
        phase0.add_static_control(name="SFC_SL", units='1/s', val=CRM["sfc_sl"],opt=False)  # CRM["sfc_sl"]
        phase0.add_static_control(name="max_thrust_sl", units='N', val=CRM["tsl_max"],opt=False)

        phase0.set_time_options(t0_val=0,t0_lower=0,t0_upper=0,tp_val=1.426355*3600,tp_lower=0.1*3600,tp_upper=3.0*3600,tp_scaler=1.0/3600.0)

        # phase0.add_constraint(name="gam",initial={"equals":0.0,"scaler":1.0},
        #                       final={"equals":0.0,"scaler":1.0},
        #                       path={"lower":np.radians(-15),"upper":np.radians(15),"scaler":1.0})

        phase0.add_path_value_constraint('tau', lower=0.01, upper=0.99, agg=aggregate_tau_constraint)


        if snopt is not None and solver == "SNOPT":
            prob.driver = pyOptSparseDriver()
            try:
                prob.driver.options["optimizer"] = "SNOPT"
                prob.driver.opt_settings["Major iterations limit"] = 100
                prob.driver.opt_settings["iSumm"] = 6
                prob.driver.opt_settings["Major step limit"] = 0.5
                prob.driver.opt_settings["Major feasibility tolerance"] = 1.0E-5
                prob.driver.opt_settings["Major optimality tolerance"] = 5.0E-5
                prob.driver.opt_settings["Minor feasibility tolerance"] = 1.0E-5
                prob.driver.opt_settings["Linesearch tolerance"] = 0.10
                prob.driver.opt_settings["Verify level"] = 3
            except:
                pass
        else:
            prob.driver = ScipyOptimizer()
            prob.driver.options['tol'] = 1.0E-4

        prob.driver.add_objective("traj0.phase0.rhs_c.W_f",
                                  indices=[0],
                                  scaler=1.0E-4)

        return prob


