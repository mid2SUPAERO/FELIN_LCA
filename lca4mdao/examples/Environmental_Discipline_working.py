import numpy as np
import openmdao.api as om
import os
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"  # base folder of BW data
import brightway2 as bw
bw.projects.set_current("LCA_FELIN")

#requires ecoinvent to exist and be populated
if "ecoinvent 3.8 cutoff" not in bw.databases or len(bw.Database("ecoinvent 3.8 cutoff")) < 10000:
    raise RuntimeError("Ecoinvent not found or empty in project 'LCA_FELIN'. Fix the project copy before running.")

from lca4mdao.component import LcaCalculationComponent
from lca4mdao.variable import ExplicitComponentLCA
from lca4mdao.utilities import cleanup_parameters

# ESA single-score helper
def _find_method(*needles):
    for m in bw.methods:
        s = str(m).lower()
        if all(n.lower() in s for n in needles):
            return m
    raise RuntimeError(f"Couldn't find method with keywords: {needles}")

# Map ESA codes -> method search patterns (uses IPCC 2013 for GWP)
ESA_METHOD_PATTERNS = {
    'GWP':    ('ipcc 2013', 'climate change', 'gwp 100a'),
    'ODEPL':  ('ef v3.0', 'ozone depletion'),
    'IORAD':  ('ef v3.0', 'ionising radiation'), 
    'PCHEM':  ('ef v3.0', 'photochemical ozone'),
    'PMAT':   ('ef v3.0', 'particulate matter'),
    'HTOXnc': ('ef v3.0', 'human toxicity', 'non-cancer'),
    'HTOXc':  ('ef v3.0', 'human toxicity', 'cancer'),
    'ACIDef': ('ef v3.0', 'acidification'),
    'FWEUT':  ('ef v3.0', 'freshwater eutrophication'),
    'MWEUT':  ('ef v3.0', 'marine eutrophication'),
    'TEUT':   ('ef v3.0', 'terrestrial eutrophication'),
    'FWTOX':  ('ef v3.0', 'freshwater ecotoxicity'),
    'LUP':    ('ef v3.0', 'land use'),
    'WDEPL':  ('ef v3.0', 'water use'),
    'ADEPLf': ('ef v3.0', 'resource use', 'fossils'),
    'ADEPLmu':('ef v3.0', 'resource use', 'minerals'),
}

#ESA normalisation & weights (from table of Simapro)
ESA_NORMALISATION = {
    'GWP': 1.235e-4,
    'ODEPL': 18.64,
    'IORAD': 2.37e-4,
    'PCHEM': 0.02463,
    'PMAT': 1680.0,
    'HTOXnc': 4354.0,
    'HTOXc': 59173.0,
    'ACIDef': 0.01800,
    'FWEUT': 0.6223,
    'MWEUT': 0.05116,
    'TEUT': 0.005658,
    'FWTOX': 2.343e-5,
    'LUP': 1.22e-6,
    'WDEPL': 8.719e-6,
    'ADEPLf': 1.538e-5,
    'ADEPLmu': 15.71,
}

ESA_WEIGHTS = {
    'GWP': 0.2106,
    'ODEPL': 0.0631,
    'IORAD': 0.0501,
    'PCHEM': 0.0478,
    'PMAT': 0.0896,
    'HTOXnc': 0.0184,
    'HTOXc': 0.0213,
    'ACIDef': 0.0620,
    'FWEUT': 0.0280,
    'MWEUT': 0.0296,
    'TEUT': 0.0371,
    'FWTOX': 0.0192,
    'LUP': 0.0794,
    'WDEPL': 0.0851,
    'ADEPLf': 0.0832,
    'ADEPLmu': 0.0755,
}

def _find_first_matching(options):
    #Return the first bw.method whose string contains *all* tokens in an option.
    ms = list(bw.methods)
    for opt in options:
        for m in ms:
            s = str(m).lower()
            if all(tok in s for tok in opt):
                return m
    raise RuntimeError(f"Couldn't find any of: {options}")

# For each ESA code, give a list of fallback token-tuples (first that matches wins)
ESA_METHOD_OPTIONS = {
    'GWP':    [
        ('ipcc 2013', 'climate change', 'gwp 100a'),
        ('ef v3.0', 'climate change', 'gwp100'),
        ('cml', 'climate change', 'gwp 100a'),
    ],
    'ODEPL':  [
        ('ef v3.0', 'ozone depletion', 'odp'),
        ('cml', 'ozone depletion'),
    ],
    'IORAD':  [
        ('ef v3.0', 'ionising radiation: human health'),
        ('cml', 'ionising radiation'),
    ],
    'PCHEM':  [
        ('ef v3.0', 'photochemical ozone'),
        ('cml', 'photochemical oxid'),
    ],
    'PMAT':   [
        ('ef v3.0', 'particulate matter formation', 'impact on human health'),
        ('cml', 'particulate matter'),
    ],
    'HTOXnc': [
        ('ef v3.0', 'human toxicity: non-carcinogenic'),
        ('ef v2.0', 'human toxicity: non-carcinogenic'),
        ('usetox', 'human toxicity', 'non-carcinogenic'),
        ('cml', 'human toxicity'),
    ],
    'HTOXc':  [
        ('ef v3.0', 'human toxicity: carcinogenic'),
        ('ef v2.0', 'human toxicity: carcinogenic'),
        ('usetox', 'human toxicity', 'carcinogenic'),
        ('cml', 'human toxicity'),
    ],
    'ACIDef': [
        ('ef v3.0', 'acidification', 'accumulated exceedance'),
        ('cml', 'acidification'),
    ],
    'FWEUT':  [
        ('ef v3.0', 'eutrophication: freshwater'),
        ('cml', 'freshwater eutrophication'),
    ],
    'MWEUT':  [
        ('ef v3.0', 'eutrophication: marine'),
        ('cml', 'marine eutrophication'),
    ],
    'TEUT':   [
        ('ef v3.0', 'eutrophication: terrestrial'),
        ('cml', 'terrestrial eutrophication'),
    ],
    'FWTOX':  [
        ('recipe midpoint (h) v1.13', 'freshwater ecotoxicity', 'fetp100'),
        ('recipe midpoint (i)', 'freshwater ecotoxicity', 'fetp100'),
        ('recipe midpoint', 'freshwater ecotoxicity'),
    ],
    'LUP':    [
        ('ef v3.0', 'land use', 'soil quality index'),
        ('ef v2.0', 'land use', 'soil quality index'),
    ],
    'WDEPL':  [
        ('ef v3.0', 'water use', 'user deprivation'),
        ('ef v2.0', 'water use', 'user deprivation'),
    ],
    'ADEPLf': [
        ('cml v4.8 2016', 'abiotic depletion potential', 'fossil fuels'),
        ('cml', 'abiotic depletion', 'fossil'),
    ],
    'ADEPLmu':[
        ('cml v4.8 2016', 'abiotic depletion potential', 'elements'),
        ('ef v3.0', 'material resources: metals/minerals', 'abiotic depletion potential'),
        ('cml', 'abiotic depletion', 'elements'),
    ],
}

#for debugging (source: chatgpt)
def reset_lca4mdao_parameter_group():
    """Remove stale 'lca4mdao' parameter group in this BW project."""
    try:
        from bw2data.parameters import ActivityParameter, Group
        q = Group.select().where(Group.name == "lca4mdao")
        if q.exists():
            ids = [g.id for g in q]
            ActivityParameter.delete().where(ActivityParameter.group.in_(ids)).execute()
            Group.delete().where(Group.id.in_(ids)).execute()
            print("✓ Reset parameter group 'lca4mdao'")
        else:
            print("✓ Parameter group 'lca4mdao' not present (clean)")
    except Exception as e:
        print(f"⚠ Couldn't reset parameter group (non-fatal): {e}")

# Ensure the environment is ready for LCA calculations
def ensure_environment_ready():
    try:
        if not create_launcher_database():
            return False
        reset_lca4mdao_parameter_group()
        print("✓ Environment ready (using existing project & ecoinvent).")
        return True
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        return False

# Create the launcher database blocks
def create_launcher_database(overwrite=False):
    """Create or refresh the tiny helper DB used by lca_parent."""
    try:
        data = {
            ('launcher_components', 'launcher_system'): {
                'name': 'Complete launcher system',
                'unit': 'unit',
                'exchanges': [],
            },
            ('launcher_components', 'stage1_structure'): {
                'name': 'Stage 1 structural components',
                'unit': 'kilogram',
                'exchanges': [],
            },
            ('launcher_components', 'stage2_structure'): {
                'name': 'Stage 2 structural components',
                'unit': 'kilogram',
                'exchanges': [],
            },
            ('launcher_components', 'propulsion_system'): {
                'name': 'Propulsion system components',
                'unit': 'kilogram',
                'exchanges': [],
            },
            ('launcher_components', 'propellants'): {
                'name': 'Rocket propellants',
                'unit': 'kilogram',
                'exchanges': [],
            },
            ('launcher_components', 'operations'): {
                'name': 'Launch operations and transport',
                'unit': 'kilogram',
                'exchanges': [],
            },
        }

        if 'launcher_components' in bw.databases:
            db = bw.Database('launcher_components')
            try:
                n = len(db)
            except Exception:
                del bw.databases['launcher_components']
            else:
                if overwrite or n == 0:
                    del bw.databases['launcher_components']

        bw.Database('launcher_components').write(data)
        print("✓ Launcher components database ready")
        return True

    except Exception as e:
        print(f"✗ Error creating launcher database: {e}")
        return False

# MATERIALS FROM ECOINVENT - still being improved

def get_material_keys():
    available_dbs = list(bw.databases)
    print(f"Available databases: {available_dbs}")

    if "ecoinvent 3.8 cutoff" in available_dbs and len(bw.Database("ecoinvent 3.8 cutoff")) > 10000:
        print(f"Using ecoinvent database with {len(bw.Database('ecoinvent 3.8 cutoff'))} activities")
        return {
            'carbon_fibre': ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b'),
            'aluminium_alloy': ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7'),
            'steel_stainless': ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2'),
            'nickel_alloy': ('ecoinvent 3.8 cutoff', '6f592c599b70d14247116fdf44a0824a'),
            'titanium_alloy': ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072'),
            'liquid_oxygen': ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456'),
            'liquid_hydrogen': ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39'),
            'transport_freight': ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083'),
            'electricity_fr': ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643'),
        }

    raise RuntimeError("Ecoinvent not available; refusing to fall back to test materials.")


# ENVIRONMENTAL DISCIPLINE 
# - using components from lca4mdao example: explicitcomponentlca 
# - some units were adapted example: transport was ton kg but openmdao doesn't recognise it, so changed to unitless

class FELINEnvironmentalDiscipline(ExplicitComponentLCA):
    
    def setup(self):
        # Get material keys
        self.MATERIAL_KEYS = get_material_keys()
        
        # INPUTS FROM FELIN 
        # Mass inputs from Structure disciplines
        self.add_input('Dry_mass_stage_1', val=45000.0, units='kg')
        self.add_input('Dry_mass_stage_2', val=6000.0, units='kg')
        
        # Mass inputs from Trajectory discipline
        self.add_input('Prop_mass_stage_1', val=320000.0, units='kg')
        self.add_input('Prop_mass_stage_2', val=75000.0, units='kg')
        self.add_input('GLOW', val=450000.0, units='kg')
        
        # Configuration inputs
        self.add_input('N_eng_stage_1', val=8.0)
        self.add_input('N_eng_stage_2', val=1.0)
        self.add_input('OF_stage_1', val=5.0)
        self.add_input('OF_stage_2', val=5.0)
        
        # Performance inputs
        self.add_input('Isp_stage_1', val=430.0, units='s')
        self.add_input('Isp_stage_2', val=465.0, units='s')
        
        # Geometry and performance
        self.add_input('Diameter_stage_1', val=5.0, units='m')
        self.add_input('Diameter_stage_2', val=5.0, units='m')
        self.add_input('Mass_flow_rate_stage_1', val=250.0, units='kg/s')
        self.add_input('Mass_flow_rate_stage_2', val=250.0, units='kg/s')
        self.add_input('Thrust_stage_1', val=1000.0, units='kN')
        self.add_input('Thrust_stage_2', val=800.0, units='kN')
        
        # MATERIAL COMPOSITION DESIGN VARIABLES
        self.add_input('cfrp_fraction_stage1', val=0.25)
        self.add_input('aluminum_fraction_stage1', val=0.65)
        self.add_input('steel_fraction_stage1', val=0.10)
        
        self.add_input('cfrp_fraction_stage2', val=0.35)
        self.add_input('aluminum_fraction_stage2', val=0.55)
        self.add_input('steel_fraction_stage2', val=0.10)
        
        self.add_input('engine_nickel_fraction', val=0.60)
        self.add_input('engine_steel_fraction', val=0.30)
        self.add_input('engine_titanium_fraction', val=0.10)
        
        self.add_input('payload_mass', val=15000.0, units='kg')
        
        # LCA OUTPUTS
        # Stage 1 structure materials
        self.add_output('stage1_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=self.MATERIAL_KEYS['carbon_fibre'], 
                       lca_units='kilogram')
        
        self.add_output('stage1_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=self.MATERIAL_KEYS['aluminium_alloy'],
                       lca_units='kilogram')
        
        self.add_output('stage1_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=self.MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram')
        
        # Stage 2 structure materials  
        self.add_output('stage2_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=self.MATERIAL_KEYS['carbon_fibre'],
                       lca_units='kilogram')
        
        self.add_output('stage2_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=self.MATERIAL_KEYS['aluminium_alloy'],
                       lca_units='kilogram')
        
        self.add_output('stage2_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=self.MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram')
        
        # Engine materials
        self.add_output('engines_nickel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=self.MATERIAL_KEYS['nickel_alloy'],
                       lca_units='kilogram')
        
        self.add_output('engines_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=self.MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram')
        
        self.add_output('engines_titanium_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=self.MATERIAL_KEYS['titanium_alloy'],
                       lca_units='kilogram')
        
        # Propellants
        self.add_output('total_lox_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propellants"),
                       lca_key=self.MATERIAL_KEYS['liquid_oxygen'],
                       lca_units='kilogram')
        
        self.add_output('total_fuel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propellants"),
                       lca_key=self.MATERIAL_KEYS['liquid_hydrogen'],
                       lca_units='kilogram')
        
        # Transport and operations
        self.add_output('transport_operations', val=0.0, units='unitless',
                        lca_parent=("launcher_components", "operations"),
                        lca_key=self.MATERIAL_KEYS['transport_freight'],
                        lca_units='ton kilometer')

        self.add_output('launch_operations_electricity', val=0.0, units='unitless',
                        lca_parent=("launcher_components", "operations"),
                        lca_key=self.MATERIAL_KEYS['electricity_fr'],
                        lca_units='kilowatt hour')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Extract inputs
        dry_mass_s1 = inputs['Dry_mass_stage_1']
        dry_mass_s2 = inputs['Dry_mass_stage_2']
        prop_mass_s1 = inputs['Prop_mass_stage_1']
        prop_mass_s2 = inputs['Prop_mass_stage_2']
        glow = inputs['GLOW']
        
        n_eng_s1 = inputs['N_eng_stage_1']
        n_eng_s2 = inputs['N_eng_stage_2']
        of_s1 = inputs['OF_stage_1']
        of_s2 = inputs['OF_stage_2']
        
        thrust_s1 = inputs['Thrust_stage_1']
        thrust_s2 = inputs['Thrust_stage_2']
        
        # Material fractions
        cfrp_frac_s1 = inputs['cfrp_fraction_stage1']
        al_frac_s1 = inputs['aluminum_fraction_stage1']
        steel_frac_s1 = inputs['steel_fraction_stage1']
        
        cfrp_frac_s2 = inputs['cfrp_fraction_stage2']
        al_frac_s2 = inputs['aluminum_fraction_stage2']
        steel_frac_s2 = inputs['steel_fraction_stage2']
        
        ni_frac = inputs['engine_nickel_fraction']
        steel_eng_frac = inputs['engine_steel_fraction']
        ti_frac = inputs['engine_titanium_fraction']
        
        # CALCULATE STRUCTURE MATERIALS
        outputs['stage1_cfrp_mass'] = dry_mass_s1 * cfrp_frac_s1
        outputs['stage1_aluminum_mass'] = dry_mass_s1 * al_frac_s1
        outputs['stage1_steel_mass'] = dry_mass_s1 * steel_frac_s1
        
        outputs['stage2_cfrp_mass'] = dry_mass_s2 * cfrp_frac_s2
        outputs['stage2_aluminum_mass'] = dry_mass_s2 * al_frac_s2
        outputs['stage2_steel_mass'] = dry_mass_s2 * steel_frac_s2
        
        # CALCULATE ENGINE MATERIALS
        # Realistic engine mass based on thrust
        thrust_per_engine_s1 = thrust_s1 / max(n_eng_s1, 1)
        thrust_per_engine_s2 = thrust_s2 / max(n_eng_s2, 1)
        
        # Engine mass scaling: ~1.4 kg per kN thrust
        engine_mass_s1_per_unit = thrust_per_engine_s1 * 1.4
        engine_mass_s2_per_unit = thrust_per_engine_s2 * 1.4
        
        total_engine_mass = (n_eng_s1 * engine_mass_s1_per_unit + 
                            n_eng_s2 * engine_mass_s2_per_unit)
        
        outputs['engines_nickel_mass'] = total_engine_mass * ni_frac
        outputs['engines_steel_mass'] = total_engine_mass * steel_eng_frac
        outputs['engines_titanium_mass'] = total_engine_mass * ti_frac
        
        # CALCULATE PROPELLANTS
        # LOX/LH2 breakdown based on O/F ratio
        lox_mass_s1 = (of_s1 / (1 + of_s1)) * prop_mass_s1 if of_s1 > 0 else prop_mass_s1 * 0.83
        fuel_mass_s1 = (1 / (1 + of_s1)) * prop_mass_s1 if of_s1 > 0 else prop_mass_s1 * 0.17
        
        lox_mass_s2 = (of_s2 / (1 + of_s2)) * prop_mass_s2 if of_s2 > 0 else prop_mass_s2 * 0.83
        fuel_mass_s2 = (1 / (1 + of_s2)) * prop_mass_s2 if of_s2 > 0 else prop_mass_s2 * 0.17
        
        outputs['total_lox_mass'] = lox_mass_s1 + lox_mass_s2
        outputs['total_fuel_mass'] = fuel_mass_s1 + fuel_mass_s2
        
        # Transport: *ton-km*
        transport_distance_km = 5000.0
        launcher_dry_mass_tons = (dry_mass_s1 + dry_mass_s2 + total_engine_mass) / 1000.0
        outputs['transport_operations'] = launcher_dry_mass_tons * transport_distance_km  # tkm

        # Electricity: *kWh*
        electricity_kwh = glow * 0.15
        outputs['launch_operations_electricity'] = electricity_kwh  # kWh

# LCA CALCULATION with robust error handling

class FELINLCACalculation(LcaCalculationComponent):
    
    def setup(self):
        # mass-type inputs (kg)
        kg_inputs = [
            'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
            'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass',
            'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass',
            'total_lox_mass', 'total_fuel_mass',
        ]
        for name in kg_inputs:
            self.add_input(name, units='kg')

        # ops inputs
        self.add_input('transport_operations', units='unitless')
        self.add_input('launch_operations_electricity', units='unitless')


        # Mission parameters
        self.add_input('payload_mass', val=15000.0, units='kg')
        self.add_input('GLOW', val=450000.0, units='kg')

        # LCA output with a GWP 100a method
        try:
            methods = list(bw.methods)
            if not methods:
                raise Exception("No LCIA methods available")

            def s(m): return str(m).lower()

            # candidates: any GWP / climate method
            cands = [m for m in methods if ('gwp' in s(m) or 'climate change' in s(m))]

            # prefer IPCC 100a first, then any 100a, else first climate method
            pref = [m for m in cands if ('ipcc' in s(m) and '100a' in s(m))]
            if not pref:
                pref = [m for m in cands if '100a' in s(m)]
            if not pref:
                pref = cands

            method = pref[0]
            print(f"✓ Using LCIA method: {method}")

            self.add_lca_output(
                'GWP_total',
                {
                    ("launcher_components", "stage1_structure"): 1,
                    ("launcher_components", "stage2_structure"): 1,
                    ("launcher_components", "propulsion_system"): 1,
                    ("launcher_components", "propellants"): 1,
                    ("launcher_components", "operations"): 1,
                },
                method_key=method,
                units='kg',
            )

        except Exception as e:
            print(f"⚠ LCA method setup failed: {e}")
            print("Using fallback calculation...")
            self.add_output('GWP_total', val=0.0, units='kg')
            self._use_fallback = True

                #ESA category LCIA outputs
        group_dict = {
            ("launcher_components", "stage1_structure"): 1,
            ("launcher_components", "stage2_structure"): 1,
            ("launcher_components", "propulsion_system"): 1,
            ("launcher_components", "propellants"): 1,
            ("launcher_components", "operations"): 1,
        }

        self.ESA_methods = {}
        for code, options in ESA_METHOD_OPTIONS.items():
            m = _find_first_matching(options)
            self.ESA_methods[code] = m
            self.add_lca_output(f'ESA_{code}', group_dict, method_key=m, units='unitless')



    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if hasattr(self, '_use_fallback'):
            # GWP calculation
            gwp_factors = {
                'carbon_fibre': 14.1,
                'aluminium': 8.9,
                'steel': 2.3,
                'nickel': 16.0,
                'titanium': 12.0,
                'lox': 0.8,
                'lh2': 12.0,
                'transport': 1.0,
                'electricity': 1.0
            }
            
            gwp = (inputs['stage1_cfrp_mass'] * gwp_factors['carbon_fibre'] +
                   inputs['stage2_cfrp_mass'] * gwp_factors['carbon_fibre'] +
                   inputs['stage1_aluminum_mass'] * gwp_factors['aluminium'] +
                   inputs['stage2_aluminum_mass'] * gwp_factors['aluminium'] +
                   inputs['stage1_steel_mass'] * gwp_factors['steel'] +
                   inputs['stage2_steel_mass'] * gwp_factors['steel'] +
                   inputs['engines_nickel_mass'] * gwp_factors['nickel'] +
                   inputs['engines_steel_mass'] * gwp_factors['steel'] +
                   inputs['engines_titanium_mass'] * gwp_factors['titanium'] +
                   inputs['total_lox_mass'] * gwp_factors['lox'] +
                   inputs['total_fuel_mass'] * gwp_factors['lh2'] +
                   inputs['transport_operations'] * gwp_factors['transport'] +
                   inputs['launch_operations_electricity'] * gwp_factors['electricity'])
            
            outputs['GWP_total'] = gwp
            print(f"Fallback GWP calculation: {float(gwp):.0f} kg CO2-eq")

class ESASingleScore(om.ExplicitComponent):
    def setup(self):
        for code in ESA_WEIGHTS:
            self.add_input(f'ESA_{code}', units='unitless')
        self.add_output('ESA_single_score', units='unitless')

    def compute(self, inputs, outputs):
        score = 0.0
        for code in ESA_WEIGHTS:
            val = float(np.asarray(inputs[f'ESA_{code}']).ravel()[0])
            norm = ESA_NORMALISATION[code]
            w = ESA_WEIGHTS[code]
            if norm > 0:
                score += w * (val / norm)
        outputs['ESA_single_score'] = score

# ENVIRONMENTAL METRICS

class FELINEnvironmentalMetrics(om.ExplicitComponent):
    """Environmental metrics calculation"""
    
    def setup(self):
        self.add_input('GWP_total', val=0.0, units='kg')
        self.add_input('payload_mass', val=15000.0, units='kg')
        self.add_input('GLOW', val=450000.0, units='kg')
        self.add_input('Isp_stage_1', val=430.0, units='s')
        self.add_input('Isp_stage_2', val=465.0, units='s')
        
        self.add_output('GWP_per_kg_payload', val=0.0, units='kg/kg')
        self.add_output('environmental_score', val=0.0)
        self.add_output('environmental_efficiency', val=0.0, units='kg/kg')
        self.add_output('performance_env_ratio', val=0.0, units='s*kg/kg')
        self.add_output('mass_efficiency', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        gwp = max(inputs['GWP_total'], 1.0)
        payload = max(inputs['payload_mass'], 1.0)
        glow = max(inputs['GLOW'], 1.0)
        isp1 = inputs['Isp_stage_1']
        isp2 = inputs['Isp_stage_2']
        
        outputs['GWP_per_kg_payload'] = gwp / payload
        outputs['environmental_efficiency'] = payload / (gwp / 1000.0)
        
        # Normalized environmental score
        gwp_ref = 1200.0
        outputs['environmental_score'] = (gwp / payload) / gwp_ref
        
        # Performance metrics
        avg_isp = (isp1 + isp2) / 2.0
        outputs['performance_env_ratio'] = avg_isp / (gwp / payload)
        outputs['mass_efficiency'] = payload / glow


class FELINEnvironmentalGroup(om.Group):
    """Main environmental group"""
    
    def setup(self):
        # Environmental discipline
        self.add_subsystem('env_discipline', FELINEnvironmentalDiscipline(),
                          promotes_inputs=[
                              'Dry_mass_stage_1', 'Dry_mass_stage_2',
                              'Prop_mass_stage_1', 'Prop_mass_stage_2', 'GLOW',
                              'N_eng_stage_1', 'N_eng_stage_2', 'OF_stage_1', 'OF_stage_2',
                              'Isp_stage_1', 'Isp_stage_2',
                              'Diameter_stage_1', 'Diameter_stage_2',
                              'Mass_flow_rate_stage_1', 'Mass_flow_rate_stage_2',
                              'Thrust_stage_1', 'Thrust_stage_2',
                              'cfrp_fraction_stage1', 'aluminum_fraction_stage1', 'steel_fraction_stage1',
                              'cfrp_fraction_stage2', 'aluminum_fraction_stage2', 'steel_fraction_stage2',
                              'engine_nickel_fraction', 'engine_steel_fraction', 'engine_titanium_fraction',
                              'payload_mass'
                          ],
                          promotes_outputs=[
                              'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
                              'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass',
                              'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass',
                              'total_lox_mass', 'total_fuel_mass',
                              'transport_operations', 'launch_operations_electricity'
                          ])
        
        # LCA calculation
        self.add_subsystem('lca_calc', FELINLCACalculation(),
                          promotes_inputs=[
                              'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
                              'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass',
                              'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass',
                              'total_lox_mass', 'total_fuel_mass',
                              'transport_operations', 'launch_operations_electricity',
                              'payload_mass', 'GLOW'
                          ],
                          promotes_outputs=['GWP_total'])
        
        # after lca_calc
        self.add_subsystem('esa_score', ESASingleScore(),
                            promotes_inputs=[f'ESA_{c}' for c in ESA_WEIGHTS.keys()],
                            promotes_outputs=['ESA_single_score'])
        
        # Environmental metrics
        self.add_subsystem('env_metrics', FELINEnvironmentalMetrics(),
                          promotes_inputs=[
                              'GWP_total', 'payload_mass', 'GLOW', 'Isp_stage_1', 'Isp_stage_2'
                          ],
                          promotes_outputs=[
                              'GWP_per_kg_payload', 'environmental_score', 'environmental_efficiency',
                              'performance_env_ratio', 'mass_efficiency'
                          ])


# OPENMDAO COMPATIBILITY PATCH

def apply_openmdao_patch():
    """Apply compatibility patch for OpenMDAO 3.36+"""
    def patched_setup_procs(self, pathname, comm, prob_meta):
        super(LcaCalculationComponent, self)._setup_procs(pathname, comm, prob_meta)
    
    LcaCalculationComponent._setup_procs = patched_setup_procs
    print("✓ OpenMDAO compatibility patch applied")

# TESTING AND UTILITY FUNCTIONS

print("Project:", bw.projects.current)
print("DBs:", list(bw.databases))
if "ecoinvent 3.8 cutoff" not in bw.databases:
    raise RuntimeError("Ecoinvent missing in LCA_FELIN")


def test_lca_environment():
    """Test the LCA environment setup"""
    
    print("=" * 60)
    print("TESTING FELIN-LCA ENVIRONMENT")
    print("=" * 60)
    
    try:
        # Test database access
        print(f"Current project: {bw.projects.current}")
        print(f"Available databases: {list(bw.databases)}")
        
        # Test available methods
        if bw.methods:
            print(f"Available LCIA methods: {len(list(bw.methods))}")
        else:
            print("⚠ No LCIA methods available")
        
        # Test environmental discipline
        from openmdao.api import Problem
        
        prob = Problem()
        prob.model = FELINEnvironmentalGroup()
        prob.setup(check=False)
        
        # Set test values based on your FELIN baseline
        prob.set_val('Dry_mass_stage_1', 45000)
        prob.set_val('Dry_mass_stage_2', 6000) 
        prob.set_val('Prop_mass_stage_1', 320000)
        prob.set_val('Prop_mass_stage_2', 75000)
        prob.set_val('GLOW', 450000)
        prob.set_val('payload_mass', 15000)
        prob.set_val('N_eng_stage_1', 8)
        prob.set_val('N_eng_stage_2', 1)
        prob.set_val('OF_stage_1', 5.0)
        prob.set_val('OF_stage_2', 5.0)
        prob.set_val('Isp_stage_1', 430.0)
        prob.set_val('Isp_stage_2', 465.0)
        prob.set_val('Thrust_stage_1', 1000.0)
        prob.set_val('Thrust_stage_2', 800.0)
        
        prob.run_model()
        
        # Check results
        gwp_total = prob.get_val('GWP_total')[0]
        gwp_per_kg = prob.get_val('GWP_per_kg_payload')[0]
        lox_mass = prob.get_val('total_lox_mass')[0]
        fuel_mass = prob.get_val('total_fuel_mass')[0]
        env_efficiency = prob.get_val('environmental_efficiency')[0]
        
        print(f"\n✓ Environmental discipline test successful!")
        print(f"  GWP total: {gwp_total:.0f} kg CO₂-eq")
        print(f"  GWP per kg payload: {gwp_per_kg:.1f} kg CO₂-eq/kg")
        print(f"  Environmental efficiency: {env_efficiency:.2f} kg payload/tonne CO₂-eq")
        print(f"  LOX mass: {lox_mass:.0f} kg")
        print(f"  LH2 mass: {fuel_mass:.0f} kg")
        
        # Environmental rating
        if gwp_per_kg < 800:
            rating = "EXCELLENT"
        elif gwp_per_kg < 1200:
            rating = "GOOD"
        elif gwp_per_kg < 1800:
            rating = "ACCEPTABLE"
        else:
            rating = "HIGH IMPACT"
            
        print(f"  Environmental rating: {rating}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_environmental_results(prob):
    """Print environmental results from a FELIN problem"""
    
    try:
        print("\n" + "="*60)
        print("ENVIRONMENTAL IMPACT RESULTS")
        print("="*60)
        
        # Basic metrics
        gwp_total = prob.get_val('GWP_total')[0]
        gwp_per_kg = prob.get_val('GWP_per_kg_payload')[0]
        payload = prob.get_val('payload_mass')[0]
        env_efficiency = prob.get_val('environmental_efficiency')[0]
        env_score = prob.get_val('environmental_score')[0]
        
        print(f"Payload mass:           {payload:8.0f} kg")
        print(f"Total GWP:              {gwp_total:8.0f} kg CO₂-eq")
        print(f"GWP per kg payload:     {gwp_per_kg:8.1f} kg CO₂-eq/kg")
        print(f"Environmental efficiency: {env_efficiency:6.2f} kg payload/tonne CO₂-eq")
        print(f"Environmental score:    {env_score:8.3f}")
        
        # Material breakdown
        print(f"\n--- MATERIAL BREAKDOWN ---")
        lox_mass = prob.get_val('total_lox_mass')[0]
        fuel_mass = prob.get_val('total_fuel_mass')[0]
        cfrp_total = (prob.get_val('stage1_cfrp_mass')[0] + 
                     prob.get_val('stage2_cfrp_mass')[0])
        al_total = (prob.get_val('stage1_aluminum_mass')[0] + 
                   prob.get_val('stage2_aluminum_mass')[0])
        
        print(f"Total LOX:              {lox_mass:8.0f} kg ({lox_mass/(lox_mass+fuel_mass):.1%})")
        print(f"Total LH2:              {fuel_mass:8.0f} kg ({fuel_mass/(lox_mass+fuel_mass):.1%})")
        print(f"Total CFRP:             {cfrp_total:8.0f} kg")
        print(f"Total Aluminum:         {al_total:8.0f} kg")
        
        # Environmental context
        print(f"\n--- ENVIRONMENTAL CONTEXT ---")
        co2_tonnes = gwp_total / 1000
        car_km = gwp_total / 120  # Average car: 120g CO2/km
        
        print(f"CO₂-eq per mission:     {co2_tonnes:8.1f} tonnes")
        print(f"Equivalent car travel:  {car_km:8.0f} km")
        
        # Environmental rating
        if gwp_per_kg < 800:
            rating = "EXCELLENT"
        elif gwp_per_kg < 1200:
            rating = "GOOD"
        elif gwp_per_kg < 1800:
            rating = "ACCEPTABLE"
        else:
            rating = "HIGH IMPACT"
            
        print(f"Environmental rating:   {rating}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error displaying environmental results: {e}")


# MAIN EXECUTION

def main():
    print("FELIN-LCA Fixed Environmental Discipline")
    print("=" * 50)

    apply_openmdao_patch()

    try:
        success = ensure_environment_ready()  # no path, no imports
        if success:
            if test_lca_environment():
                print("\n🎉 FELIN-LCA INTEGRATION SUCCESSFUL!")
                print("✓ Using existing Brightway project: LCA_FELIN")
                print("✓ Ecoinvent is present")
            else:
                print("\n⚠ Environment ok but testing failed (see logs)")
        else:
            print("\n❌ Environment not ready (see logs)")
    except Exception as e:
        print(f"\n❌ Main execution failed: {e}")
        import traceback; traceback.print_exc()



if __name__ == '__main__':
    main()