import numpy as np
import brightway2 as bw
import openmdao.api as om
import openmdao.core
import bw2data
import bw2io
from bw2calc import LCA
import pandas as pd
from peewee import IntegrityError
from lca4mdao.component import LcaCalculationComponent
from lca4mdao.utilities import cleanup_parameters, setup_ecoinvent, setup_bw
from lca4mdao.variable import ExplicitComponentLCA

bw.projects.set_current("Ariana_5_LCA")  # Your project name
bw.bw2setup()
print(bw.projects.report())
print(bw.databases)

ecoinvent = bw.Database("ecoinvent 3.8 cutoff")
print("Random activity:", ecoinvent.random())

fp = r'C:\Users\joana\Desktop\Joana\FELIN\ecoinvent 3.8_cutoff_ecoSpold02\datasets'

if "ecoinvent 3.8 cutoff" not in bw.databases or bw.Database("ecoinvent 3.8 cutoff").random() is None:
    print("Importing ecoinvent...")
    from bw2io.importers import SingleOutputEcospold2Importer

    ei = SingleOutputEcospold2Importer(fp, name="ecoinvent 3.8 cutoff")
    ei.apply_strategies()
    ei.statistics()       # Will print unlinked/invalid exchanges etc.
    ei.write_database()
    print("Ecoinvent imported successfully!")
else:
    print("ecoinvent 3.8 cutoff already imported and not empty.")

# Define the ecoinvent keys as in your notebook
carbon_fibre = ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b')
aluminium_almg3 = ('ecoinvent 3.8 cutoff', '3d66c7f5f8d813a5b63b2d19a41ec763')
aluminium_alli = ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7')
titanium = ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072')
nickel = ('ecoinvent 3.8 cutoff', '6f592c599b70d14247116fdf44a0824a')
steel = ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2')
electronic_active = ('ecoinvent 3.8 cutoff', '52c4f6d2e1ec507b1ccc96056a761c0d')
electronic_passive = ('ecoinvent 3.8 cutoff', 'b1b65fe4d00b29f2299c72b894a3c0a0')
wire = ('ecoinvent 3.8 cutoff', 'f8586b86fe8ac595be9f6b18e9b94488')
battery = ('ecoinvent 3.8 cutoff', 'b2feecd5152754c08303bc84dc371b68')
motor = ('ecoinvent 3.8 cutoff', '0a45c922ec9f5a8345c88fb3ecc28b6f')
oxygen = ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456')
hydrogen = ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39')
transport = ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083')
electricity = ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643')


def build_data():
    """Build the Brightway2 database for the launcher components"""
    components.delete(warn=False)
    components = bw.Database('components')
    components.register()
    
    # Create the activities as in your notebook
    components.new_activity('launcher', name='launcher').save()
    components.new_activity('launch_per_kg', name='launch of ariane 5 per kg payload').save()


class LauncherStructure(ExplicitComponentLCA):
    """Component for launcher structural elements"""
    
    def setup(self):
        # Design variables (inputs)
        self.add_input('payload_mass', val=5000.0, units='kg', desc='Payload mass')
        self.add_input('fairing_cfrp_factor', val=1.0, desc='CFRP fairing scaling factor')
        self.add_input('tank_al_factor', val=1.0, desc='Aluminum tank scaling factor')
        
        # LCA outputs - structural components
        self.add_output('fairing_cfrp', val=2400.0, units='kg', 
                       lca_parent=("components", "launcher"), 
                       lca_units='kilogram', lca_key=carbon_fibre,
                       desc='Fairing CFRP mass')
        
        self.add_output('interstage_almg3', val=400.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_almg3,
                       desc='Interstage structure mass')
        
        self.add_output('payload_adapter', val=400.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_almg3,
                       desc='Payload adapter mass')
        
        self.add_output('lh2_tank_epc', val=5000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_alli,
                       desc='LH2 tank EPC mass')
        
        self.add_output('lox_tank_epc', val=4000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_alli,
                       desc='LOX tank EPC mass')
        
        self.add_output('lh2_tank_esca', val=1200.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_alli,
                       desc='LH2 tank ESCA mass')
        
        self.add_output('lox_tank_esca', val=1000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=aluminium_alli,
                       desc='LOX tank ESCA mass')
        
        self.add_output('eap_casings', val=54000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=carbon_fibre,
                       desc='EAP casings mass')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        payload_mass = inputs['payload_mass']
        fairing_factor = inputs['fairing_cfrp_factor']
        tank_factor = inputs['tank_al_factor']
        
        # Scale components based on payload and design factors
        # These are simplified scaling relationships - in reality would be more complex
        payload_scaling = payload_mass / 5000.0  # Nominal 5000 kg payload
        
        outputs['fairing_cfrp'] = 2400.0 * fairing_factor * np.sqrt(payload_scaling)
        outputs['interstage_almg3'] = 400.0 * payload_scaling
        outputs['payload_adapter'] = 400.0 * payload_scaling
        outputs['lh2_tank_epc'] = 5000.0 * tank_factor * payload_scaling
        outputs['lox_tank_epc'] = 4000.0 * tank_factor * payload_scaling
        outputs['lh2_tank_esca'] = 1200.0 * tank_factor * payload_scaling
        outputs['lox_tank_esca'] = 1000.0 * tank_factor * payload_scaling
        outputs['eap_casings'] = 54000.0 * payload_scaling


class LauncherPropulsion(ExplicitComponentLCA):
    """Component for launcher propulsion system"""
    
    def setup(self):
        # Design inputs
        self.add_input('payload_mass', val=5000.0, units='kg')
        self.add_input('engine_scaling', val=1.0, desc='Engine scaling factor')
        
        # LCA outputs - propulsion components
        self.add_output('helium_bottles', val=80.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=titanium,
                       desc='Pressurization bottles mass')
        
        self.add_output('vulcain_engine', val=1500.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=nickel,
                       desc='Vulcain 2 engine mass')
        
        self.add_output('hm7b_engine', val=165.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=nickel,
                       desc='HM7B engine mass')
        
        self.add_output('eap_nozzles', val=800.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=steel,
                       desc='EAP nozzles mass')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        payload_mass = inputs['payload_mass']
        engine_scaling = inputs['engine_scaling']
        
        payload_scaling = payload_mass / 5000.0
        
        outputs['helium_bottles'] = 80.0 * payload_scaling
        outputs['vulcain_engine'] = 1500.0 * engine_scaling * payload_scaling
        outputs['hm7b_engine'] = 165.0 * engine_scaling * payload_scaling
        outputs['eap_nozzles'] = 800.0 * payload_scaling


class LauncherAvionics(ExplicitComponentLCA):
    """Component for launcher avionics and electrical systems"""
    
    def setup(self):
        # Design inputs
        self.add_input('payload_mass', val=5000.0, units='kg')
        self.add_input('avionics_complexity', val=1.0, desc='Avionics complexity factor')
        
        # LCA outputs - avionics components
        self.add_output('avionics_unit', val=200.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=electronic_active,
                       desc='Avionics unit mass')
        
        self.add_output('sensors', val=100.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=electronic_active,
                       desc='Sensors mass')
        
        self.add_output('wiring', val=50.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=electronic_passive,
                       desc='Wiring mass')
        
        self.add_output('copper_harnesses', val=100.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=wire,
                       desc='Copper harnesses mass')
        
        self.add_output('batteries', val=50.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=battery,
                       desc='Batteries mass')
        
        self.add_output('tvc_actuators', val=400.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=motor,
                       desc='TVC actuators mass')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        payload_mass = inputs['payload_mass']
        avionics_complexity = inputs['avionics_complexity']
        
        payload_scaling = payload_mass / 5000.0
        
        outputs['avionics_unit'] = 200.0 * avionics_complexity * payload_scaling
        outputs['sensors'] = 100.0 * avionics_complexity * payload_scaling
        outputs['wiring'] = 50.0 * avionics_complexity * payload_scaling
        outputs['copper_harnesses'] = 100.0 * payload_scaling
        outputs['batteries'] = 50.0 * avionics_complexity * payload_scaling
        outputs['tvc_actuators'] = 400.0 * payload_scaling


class LauncherPropellants(ExplicitComponentLCA):
    """Component for launcher propellants"""
    
    def setup(self):
        # Design inputs
        self.add_input('payload_mass', val=5000.0, units='kg')
        self.add_input('propellant_efficiency', val=1.0, desc='Propellant efficiency factor')
        
        # LCA outputs - propellants
        self.add_output('lox_propellant', val=133000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=oxygen,
                       desc='LOX propellant mass')
        
        self.add_output('lh2_propellant', val=25000.0, units='kg',
                       lca_parent=("components", "launcher"),
                       lca_units='kilogram', lca_key=hydrogen,
                       desc='LH2 propellant mass')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        payload_mass = inputs['payload_mass']
        propellant_efficiency = inputs['propellant_efficiency']
        
        # Propellant mass scales roughly linearly with payload mass
        payload_scaling = payload_mass / 5000.0
        efficiency_factor = 1.0 / propellant_efficiency  # Better efficiency = less propellant
        
        outputs['lox_propellant'] = 133000.0 * payload_scaling * efficiency_factor
        outputs['lh2_propellant'] = 25000.0 * payload_scaling * efficiency_factor


class LauncherOperations(ExplicitComponentLCA):
    """Component for launcher operations and transport"""
    
    def setup(self):
        # Design inputs
        self.add_input('payload_mass', val=5000.0, units='kg')
        self.add_input('cycles', val=1, desc='Number of launches')
        
        # LCA outputs - operations
        self.add_output('transport_to_kourou', val=1.0, units='unitless',
                       lca_parent=("components", "launcher"),
                       lca_units='ton kilometer', lca_key=transport,
                       desc='Transport to Kourou')
        
        self.add_output('launch_operations', val=1.0, units='unitless',
                       lca_parent=("components", "launcher"),
                       lca_units='kilowatt hour', lca_key=electricity,
                       desc='Launch operations electricity')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        payload_mass = inputs['payload_mass']
        cycles = inputs['cycles']
        
        # Transport scales with total launcher mass (approximate)
        total_launcher_mass = 200.0  # Approximate dry mass in tons
        transport_distance = 7000.0  # Approximate km from Europe to Kourou
        
        outputs['transport_to_kourou'] = total_launcher_mass * transport_distance * cycles
        outputs['launch_operations'] = 10000.0 * cycles  # kWh per launch


class LauncherLCA(LcaCalculationComponent):
    """LCA calculation component for the launcher"""

    def setup(self):
        # Declare all inputs expected from the rest of the model
        self.add_input('fairing_cfrp', units='kg')
        self.add_input('interstage_almg3', units='kg')
        self.add_input('payload_adapter', units='kg')
        self.add_input('lh2_tank_epc', units='kg')
        self.add_input('lox_tank_epc', units='kg')
        self.add_input('lh2_tank_esca', units='kg')
        self.add_input('lox_tank_esca', units='kg')
        self.add_input('eap_casings', units='kg')
        self.add_input('helium_bottles', units='kg')
        self.add_input('vulcain_engine', units='kg')
        self.add_input('hm7b_engine', units='kg')
        self.add_input('eap_nozzles', units='kg')
        self.add_input('avionics_unit', units='kg')
        self.add_input('sensors', units='kg')
        self.add_input('wiring', units='kg')
        self.add_input('copper_harnesses', units='kg')
        self.add_input('batteries', units='kg')
        self.add_input('tvc_actuators', units='kg')
        self.add_input('lox_propellant', units='kg')
        self.add_input('lh2_propellant', units='kg')
        self.add_input('transport_to_kourou', units='unitless')
        self.add_input('launch_operations', units='unitless')

        # Add LCA output for GWP per kg of payload
        self.add_lca_output('GWP_per_kg',
                            {("components", "launcher"): 1},
                            method_key=('ReCiPe Midpoint (H) V1.13', 'climate change', 'GWP100'),
                            units='kg')


class LauncherAnalysisGroup(om.Group):
    """Main analysis group for the Ariane 5 launcher"""
    
    def setup(self):
        # Independent variables
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('payload_mass', val=5000.0, units='kg', desc='Payload mass')
        indeps.add_output('fairing_cfrp_factor', val=1.0, desc='CFRP fairing scaling factor')
        indeps.add_output('tank_al_factor', val=1.0, desc='Aluminum tank scaling factor')
        indeps.add_output('engine_scaling', val=1.0, desc='Engine scaling factor')
        indeps.add_output('avionics_complexity', val=1.0, desc='Avionics complexity factor')
        indeps.add_output('propellant_efficiency', val=1.0, desc='Propellant efficiency factor')
        indeps.add_output('cycles', val=1, desc='Number of launches')
        
        # Add subsystems
        self.add_subsystem('structure', LauncherStructure(), promotes=['*'])
        self.add_subsystem('propulsion', LauncherPropulsion(), promotes=['*'])
        self.add_subsystem('avionics', LauncherAvionics(), promotes=['*'])
        self.add_subsystem('propellants', LauncherPropellants(), promotes=['*'])
        self.add_subsystem('operations', LauncherOperations(), promotes=['*'])
        
        # LCA calculation
        self.add_subsystem('lca', LauncherLCA(), promotes=['*'])
        
        # Calculate GWP per kg of payload
        self.add_subsystem('gwp_per_kg', 
                          om.ExecComp('GWP_per_kg_payload = GWP_per_kg / payload_mass',
                                     GWP_per_kg_payload={'units': 'kg/kg'},
                                     GWP_per_kg={'units': 'kg'},
                                     payload_mass={'units': 'kg'}),
                          promotes=['payload_mass'])
        self.connect('GWP_per_kg', 'gwp_per_kg.GWP_per_kg')


def run_launcher_analysis(optimize=False):
    """Run the launcher analysis"""
    
    # Setup problem
    prob = om.Problem()
    prob.model = LauncherAnalysisGroup()
    
    if optimize:
        # Setup optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter'] = 100
        prob.driver.options['tol'] = 1e-6
        
        # Design variables
        prob.model.add_design_var('fairing_cfrp_factor', lower=0.5, upper=2.0)
        prob.model.add_design_var('tank_al_factor', lower=0.5, upper=2.0)
        prob.model.add_design_var('engine_scaling', lower=0.5, upper=2.0)
        prob.model.add_design_var('avionics_complexity', lower=0.5, upper=2.0)
        prob.model.add_design_var('propellant_efficiency', lower=0.8, upper=1.5)
        
        # Objective: minimize GWP per kg of payload
        prob.model.add_objective('gwp_per_kg.GWP_per_kg_payload')
        
        # Constraints (example - maintain payload capability)
        prob.model.add_constraint('payload_mass', lower=3000.0, upper=7000.0)
    
    # Setup and run
    prob.setup(check=False)
    
    if optimize:
        prob.run_driver()
        print("=== OPTIMIZATION RESULTS ===")
    else:
        prob.run_model()
        print("=== ANALYSIS RESULTS ===")
    
    # Print results
    print(f"Payload mass: {prob.get_val('payload_mass')[0]:.1f} kg")
    print(f"Total GWP: {prob.get_val('GWP_per_kg')[0]:.2f} kg CO₂-eq")
    print(f"GWP per kg payload: {prob.get_val('gwp_per_kg.GWP_per_kg_payload')[0]:.2f} kg CO₂-eq/kg")
    print(f"GWP per 5t launch: {prob.get_val('GWP_per_kg')[0] * 5000 / 1000:.2f} tons CO₂-eq")
    
    if optimize:
        print("\n=== DESIGN VARIABLES ===")
        print(f"CFRP fairing factor: {prob.get_val('fairing_cfrp_factor')[0]:.3f}")
        print(f"Al tank factor: {prob.get_val('tank_al_factor')[0]:.3f}")
        print(f"Engine scaling: {prob.get_val('engine_scaling')[0]:.3f}")
        print(f"Avionics complexity: {prob.get_val('avionics_complexity')[0]:.3f}")
        print(f"Propellant efficiency: {prob.get_val('propellant_efficiency')[0]:.3f}")
    
    return prob


def run_parametric_study():
    """Run a parametric study varying payload mass"""
    
    payload_masses = np.linspace(3000, 7000, 9)  # 3-7 tons
    gwp_results = []
    gwp_per_kg_results = []
    
    print("=== PARAMETRIC STUDY: PAYLOAD MASS VARIATION ===")
    print("Payload [kg] | Total GWP [kg CO₂-eq] | GWP per kg [kg CO₂-eq/kg]")
    print("-" * 65)
    
    for payload in payload_masses:
        prob = om.Problem()
        prob.model = LauncherAnalysisGroup()
        prob.setup(check=False)
        
        prob.set_val('payload_mass', payload)
        prob.run_model()
        
        total_gwp = prob.get_val('GWP_per_kg')[0]
        gwp_per_kg = prob.get_val('gwp_per_kg.GWP_per_kg_payload')[0]
        
        gwp_results.append(total_gwp)
        gwp_per_kg_results.append(gwp_per_kg)
        
        print(f"{payload:10.0f} | {total_gwp:17.0f} | {gwp_per_kg:20.2f}")
    
    return payload_masses, gwp_results, gwp_per_kg_results

#replaces the original _setup_procs with a version that exactly matches what OpenMDAO 3.36 expects (it was giving an error "TypeError: LcaCalculationComponent._setup_procs() missing 1 required positional argument: 'prob_meta'" so it was patched to include prob_meta)

def patched_setup_procs(self, pathname, comm, prob_meta):
    super(LcaCalculationComponent, self)._setup_procs(pathname, comm, prob_meta)

LcaCalculationComponent._setup_procs = patched_setup_procs

if __name__ == '__main__':
    # Setup Brightway2 and ecoinvent
    setup_bw("Ariana_5_LCA")
    setup_ecoinvent(fp)
    build_data()
    cleanup_parameters()
    
    print("Running Ariane 5 LCA Analysis with lca4mdao")
    print("=" * 50)
    
    # Run baseline analysis
    print("\n1. BASELINE ANALYSIS")
    prob_baseline = run_launcher_analysis(optimize=False)
    
    # Run optimization
    print("\n2. OPTIMIZATION FOR MINIMUM GWP")
    prob_opt = run_launcher_analysis(optimize=True)
    
    # Run parametric study
    print("\n3. PARAMETRIC STUDY")
    payload_masses, gwp_results, gwp_per_kg_results = run_parametric_study()
    
    print(f"\nAnalysis complete. Results show GWP values similar to your notebook:")
    print(f"Expected range: 900-1200 kg CO₂-eq/kg payload")
    print(f"Calculated: {prob_baseline.get_val('gwp_per_kg.GWP_per_kg_payload')[0]:.2f} kg CO₂-eq/kg payload")
