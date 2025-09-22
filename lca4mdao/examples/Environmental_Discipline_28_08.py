"""
FELIN-LCA Environmental Discipline Module
Corrected for actual FELIN trajectory outputs
Compatible with OpenMDAO 3.36.0 and Brightway2 2.4.7
"""

import numpy as np
import openmdao.api as om
import os
from pathlib import Path

# Brightway2 setup
try:
    import brightway2 as bw
    bw.projects.set_current("LCA_FELIN")
    
    # Validate ecoinvent database
    if "ecoinvent 3.8 cutoff" not in bw.databases:
        print("Warning: Ecoinvent 3.8 cutoff not found. Using fallback calculations.")
        USE_FALLBACK = True
    else:
        db_size = len(bw.Database("ecoinvent 3.8 cutoff"))
        if db_size < 10000:
            print(f"Warning: Ecoinvent has only {db_size} activities. Using fallback.")
            USE_FALLBACK = True
        else:
            USE_FALLBACK = False
            
    from lca4mdao.component import LcaCalculationComponent
    from lca4mdao.variable import ExplicitComponentLCA
    LCA4MDAO_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Brightway2/lca4mdao not available: {e}")
    print("Using simplified environmental calculations")
    USE_FALLBACK = True
    LCA4MDAO_AVAILABLE = False
    
    # Dummy classes for compatibility
    class LcaCalculationComponent(om.ExplicitComponent):
        pass
    class ExplicitComponentLCA(om.ExplicitComponent):
        pass

# ============================================================================
# MATERIAL MAPPING TO ECOINVENT (only if available)
# ============================================================================

if not USE_FALLBACK:
    FELIN_MATERIALS = {
        'carbon_fibre': ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b'),
        'aluminium_alloy': ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7'),
        'steel_stainless': ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2'),
        'titanium': ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072'),
        'nickel': ('ecoinvent 3.8 cutoff', '6f592c599b70d14247116fdf44a0824a'),
        'liquid_oxygen': ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456'),
        'liquid_hydrogen': ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39'),
        'transport_ship': ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083'),
        'electricity_fr': ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643'),
    }
else:
    FELIN_MATERIALS = {}

# ============================================================================
# ESA SINGLE SCORE METHODOLOGY
# ============================================================================

ESA_METHOD_OPTIONS = {
    'GWP': [('ipcc 2013', 'climate change', 'gwp 100a')],
    'ODEPL': [('ef v3.0', 'ozone depletion')],
    'IORAD': [('ef v3.0', 'ionising radiation')],
    'PCHEM': [('ef v3.0', 'photochemical ozone')],
    'PMAT': [('ef v3.0', 'particulate matter')],
    'HTOXnc': [('ef v3.0', 'human toxicity', 'non-carcinogenic')],
    'HTOXc': [('ef v3.0', 'human toxicity', 'carcinogenic')],
    'ACIDef': [('ef v3.0', 'acidification')],
    'FWEUT': [('ef v3.0', 'eutrophication', 'freshwater')],
    'MWEUT': [('ef v3.0', 'eutrophication', 'marine')],
    'TEUT': [('ef v3.0', 'eutrophication', 'terrestrial')],
    'FWTOX': [('recipe midpoint', 'freshwater ecotoxicity')],
    'LUP': [('ef v3.0', 'land use')],
    'WDEPL': [('ef v3.0', 'water use')],
    'ADEPLf': [('cml v4.8', 'abiotic depletion', 'fossil')],
    'ADEPLmu': [('cml v4.8', 'abiotic depletion', 'elements')],
}

# From your table
ESA_NORMALISATION = {
    'GWP': 0.0001235, 'ODEPL': 18.64, 'IORAD': 0.000237, 'PCHEM': 0.02463,
    'PMAT': 1680.0, 'HTOXnc': 4354.0, 'HTOXc': 59173.0, 'ACIDef': 0.018,
    'FWEUT': 0.6223, 'MWEUT': 0.05116, 'TEUT': 0.005658, 'FWTOX': 0.00002343,
    'LUP': 0.00000122, 'WDEPL': 0.00008719, 'ADEPLf': 0.00001538, 'ADEPLmu': 15.71,
}

ESA_WEIGHTS = {
    'GWP': 0.2106, 'ODEPL': 0.0631, 'IORAD': 0.0501, 'PCHEM': 0.0478,
    'PMAT': 0.0896, 'HTOXnc': 0.0184, 'HTOXc': 0.0213, 'ACIDef': 0.062,
    'FWEUT': 0.028, 'MWEUT': 0.0296, 'TEUT': 0.0371, 'FWTOX': 0.0192,
    'LUP': 0.0794, 'WDEPL': 0.0851, 'ADEPLf': 0.0832, 'ADEPLmu': 0.0755,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_first_matching(options):
    """Find first matching LCIA method from options list"""
    if USE_FALLBACK:
        return None
    try:
        methods = list(bw.methods)
        for opt in options:
            for m in methods:
                s = str(m).lower()
                if all(tok.lower() in s for tok in opt):
                    return m
    except:
        pass
    return None

def create_launcher_database(overwrite=False):
    """Create FELIN launcher component database structure"""
    if USE_FALLBACK:
        return True
        
    data = {
        ('felin_launcher', 'complete_launcher'): {
            'name': 'FELIN complete launcher', 'unit': 'unit', 'exchanges': [],
        },
        ('felin_launcher', 'stage1_structure'): {
            'name': 'Stage 1 structural components', 'unit': 'kg', 'exchanges': [],
        },
        ('felin_launcher', 'stage2_structure'): {
            'name': 'Stage 2 structural components', 'unit': 'kg', 'exchanges': [],
        },
        ('felin_launcher', 'propulsion_system'): {
            'name': 'Propulsion system components', 'unit': 'kg', 'exchanges': [],
        },
        ('felin_launcher', 'propellants'): {
            'name': 'LOX/LH2 propellants', 'unit': 'kg', 'exchanges': [],
        },
        ('felin_launcher', 'operations'): {
            'name': 'Launch operations and transport', 'unit': 'unit', 'exchanges': [],
        },
    }
    
    try:
        if 'felin_launcher' in bw.databases:
            if not overwrite:
                return True
            del bw.databases['felin_launcher']
        
        bw.Database('felin_launcher').write(data)
        print("✓ FELIN launcher database created")
    except:
        print("Warning: Could not create launcher database")
    return True

# Initialize database if Brightway is available
if not USE_FALLBACK:
    create_launcher_database()

# ============================================================================
# SIMPLIFIED ENVIRONMENTAL DISCIPLINE (Always used for material calculation)
# ============================================================================

class FELINEnvironmentalDiscipline(om.ExplicitComponent):
    """
    Environmental discipline for FELIN launcher
    Works with actual FELIN trajectory outputs
    """
    
    def setup(self):
        # === INPUTS FROM FELIN DISCIPLINES ===
        
        # From Structure
        self.add_input('Dry_mass_stage_1', val=45000.0, units='kg')
        self.add_input('Dry_mass_stage_2', val=6000.0, units='kg')
        
        # From Trajectory
        self.add_input('Prop_mass_stage_1', val=350000.0, units='kg')
        self.add_input('Prop_mass_stage_2', val=75000.0, units='kg')
        self.add_input('GLOW', val=450000.0, units='kg')
        
        # Trajectory arrays (actual FELIN outputs)
        self.add_input('V_ascent', shape=4000, units='m/s')
        self.add_input('nx_ascent', shape=4000)  # Load factor array
        self.add_input('alt_ascent', shape=4000, units='m')
        self.add_input('Nb_pt_ascent', val=100.)  # Number of valid points
        self.add_input('max_pdyn_load_ascent_stage_1', val=40e3, units='Pa')
        
        # From indeps
        self.add_input('N_eng_stage_1', val=7.0)
        self.add_input('N_eng_stage_2', val=1.0)
        self.add_input('OF_stage_1', val=5.0)
        self.add_input('OF_stage_2', val=5.5)
        self.add_input('Thrust_stage_1', val=1000.0, units='kN')
        self.add_input('Thrust_stage_2', val=1150.0, units='kN')
        self.add_input('Isp_stage_1', val=330.0, units='s')
        self.add_input('Isp_stage_2', val=330.0, units='s')
        self.add_input('payload_mass', val=5000.0, units='kg')
        
        # === MATERIAL DESIGN VARIABLES ===
        # Stage 1: 10-40% CFRP, 40-70% Al, 5-20% Steel
        self.add_input('cfrp_fraction_stage1', val=0.25)
        self.add_input('aluminum_fraction_stage1', val=0.65)
        self.add_input('steel_fraction_stage1', val=0.10)
        
        # Stage 2: 20-50% CFRP, 30-60% Al, 5-20% Steel
        self.add_input('cfrp_fraction_stage2', val=0.35)
        self.add_input('aluminum_fraction_stage2', val=0.55)
        self.add_input('steel_fraction_stage2', val=0.10)
        
        # Engines: 40-70% Ni, 20-40% Steel, 5-20% Ti
        self.add_input('engine_nickel_fraction', val=0.60)
        self.add_input('engine_steel_fraction', val=0.30)
        self.add_input('engine_titanium_fraction', val=0.10)
        
        # === OUTPUTS ===
        
        # Material masses for LCA
        self.add_output('stage1_cfrp_mass', val=0.0, units='kg')
        self.add_output('stage1_aluminum_mass', val=0.0, units='kg')
        self.add_output('stage1_steel_mass', val=0.0, units='kg')
        self.add_output('stage2_cfrp_mass', val=0.0, units='kg')
        self.add_output('stage2_aluminum_mass', val=0.0, units='kg')
        self.add_output('stage2_steel_mass', val=0.0, units='kg')
        self.add_output('engines_nickel_mass', val=0.0, units='kg')
        self.add_output('engines_steel_mass', val=0.0, units='kg')
        self.add_output('engines_titanium_mass', val=0.0, units='kg')
        self.add_output('total_lox_mass', val=0.0, units='kg')
        self.add_output('total_lh2_mass', val=0.0, units='kg')
        self.add_output('transport_tkm', val=0.0, units='t*km')
        self.add_output('electricity_kwh', val=0.0, units='kW*h')
        
        # Environmental scores
        self.add_output('GWP_total', val=0.0, units='kg')
        self.add_output('ESA_single_score', val=0.0)
        
        # Metrics
        self.add_output('delta_v_achieved', val=0.0, units='m/s')
        self.add_output('max_acceleration_g', val=0.0)
        
        # Individual ESA categories
        for code in ESA_WEIGHTS.keys():
            self.add_output(f'ESA_{code}', val=0.0)
            self.add_output(f'ESA_{code}_normalized', val=0.0)

    def compute(self, inputs, outputs):
        """
        Compute material masses and environmental scores
        """
        
        # Extract trajectory performance
        nb_pts = int(inputs['Nb_pt_ascent'])
        if nb_pts > 0:
            # Calculate delta-v from velocity profile
            v_initial = inputs['V_ascent'][0]
            v_final = inputs['V_ascent'][nb_pts-1]
            outputs['delta_v_achieved'] = v_final - v_initial
            
            # Get max acceleration from nx array
            max_nx = np.max(np.abs(inputs['nx_ascent'][:nb_pts]))
            outputs['max_acceleration_g'] = max_nx
        else:
            outputs['delta_v_achieved'] = 0.0
            outputs['max_acceleration_g'] = 3.0
        
        # === CALCULATE STRUCTURE MATERIALS ===
        
        # Stage 1 materials
        dry_s1 = inputs['Dry_mass_stage_1']
        cfrp_f1 = inputs['cfrp_fraction_stage1']
        al_f1 = inputs['aluminum_fraction_stage1']
        steel_f1 = inputs['steel_fraction_stage1']
        
        total_f1 = cfrp_f1 + al_f1 + steel_f1
        if total_f1 > 0:
            outputs['stage1_cfrp_mass'] = dry_s1 * (cfrp_f1 / total_f1)
            outputs['stage1_aluminum_mass'] = dry_s1 * (al_f1 / total_f1)
            outputs['stage1_steel_mass'] = dry_s1 * (steel_f1 / total_f1)
        
        # Stage 2 materials
        dry_s2 = inputs['Dry_mass_stage_2']
        cfrp_f2 = inputs['cfrp_fraction_stage2']
        al_f2 = inputs['aluminum_fraction_stage2']
        steel_f2 = inputs['steel_fraction_stage2']
        
        total_f2 = cfrp_f2 + al_f2 + steel_f2
        if total_f2 > 0:
            outputs['stage2_cfrp_mass'] = dry_s2 * (cfrp_f2 / total_f2)
            outputs['stage2_aluminum_mass'] = dry_s2 * (al_f2 / total_f2)
            outputs['stage2_steel_mass'] = dry_s2 * (steel_f2 / total_f2)
        
        # === ENGINE MATERIALS ===
        thrust_s1 = inputs['Thrust_stage_1']
        thrust_s2 = inputs['Thrust_stage_2']
        n_eng_s1 = inputs['N_eng_stage_1']
        n_eng_s2 = inputs['N_eng_stage_2']
        
        # Engine mass estimation (1.5 kg/kN for LOX/LH2)
        engine_mass_total = (thrust_s1 * n_eng_s1 + thrust_s2 * n_eng_s2) * 1.5
        
        ni_f = inputs['engine_nickel_fraction']
        steel_eng_f = inputs['engine_steel_fraction']
        ti_f = inputs['engine_titanium_fraction']
        
        total_eng_f = ni_f + steel_eng_f + ti_f
        if total_eng_f > 0:
            outputs['engines_nickel_mass'] = engine_mass_total * (ni_f / total_eng_f)
            outputs['engines_steel_mass'] = engine_mass_total * (steel_eng_f / total_eng_f)
            outputs['engines_titanium_mass'] = engine_mass_total * (ti_f / total_eng_f)
        
        # === PROPELLANTS ===
        prop_s1 = inputs['Prop_mass_stage_1']
        prop_s2 = inputs['Prop_mass_stage_2']
        of_s1 = inputs['OF_stage_1']
        of_s2 = inputs['OF_stage_2']
        
        # Stage 1 propellants
        lox_s1 = prop_s1 * (of_s1 / (1 + of_s1))
        lh2_s1 = prop_s1 * (1 / (1 + of_s1))
        
        # Stage 2 propellants
        lox_s2 = prop_s2 * (of_s2 / (1 + of_s2))
        lh2_s2 = prop_s2 * (1 / (1 + of_s2))
        
        outputs['total_lox_mass'] = lox_s1 + lox_s2
        outputs['total_lh2_mass'] = lh2_s1 + lh2_s2
        
        # === OPERATIONS ===
        total_dry_mass = dry_s1 + dry_s2 + engine_mass_total
        outputs['transport_tkm'] = (total_dry_mass / 1000.0) * 7000  # 7000 km to Kourou
        outputs['electricity_kwh'] = inputs['GLOW'] * 0.2  # 0.2 kWh/kg
        
        # === SIMPLIFIED LCA CALCULATION ===
        # More realistic GWP factors (kg CO2eq/kg)
        gwp_factors = {
            'cfrp': 20.0,      # Was 14.1 - too low
            'aluminum': 10.3,   # Was 8.9
            'steel': 1.9,       # Was 2.3
            'nickel': 6.5,      # Was 16.0 - too high
            'titanium': 35.0,   # Was 12.0 - too low
            'lox': 0.15,        # Was 0.8 - too high
            'lh2': 9.0,         # Was 12.0
            'transport': 0.022, # per t*km
            'electricity': 0.057 # French grid per kWh
        }

        gwp = (
            outputs['stage1_cfrp_mass'] * gwp_factors['cfrp'] +
            outputs['stage2_cfrp_mass'] * gwp_factors['cfrp'] +
            outputs['stage1_aluminum_mass'] * gwp_factors['aluminum'] +
            outputs['stage2_aluminum_mass'] * gwp_factors['aluminum'] +
            outputs['stage1_steel_mass'] * gwp_factors['steel'] +
            outputs['stage2_steel_mass'] * gwp_factors['steel'] +
            outputs['engines_nickel_mass'] * gwp_factors['nickel'] +
            outputs['engines_steel_mass'] * gwp_factors['steel'] +
            outputs['engines_titanium_mass'] * gwp_factors['titanium'] +
            outputs['total_lox_mass'] * gwp_factors['lox'] +
            outputs['total_lh2_mass'] * gwp_factors['lh2'] +
            outputs['transport_tkm'] * gwp_factors['transport'] +
            outputs['electricity_kwh'] * gwp_factors['electricity']
        )

        outputs['GWP_total'] = gwp

        # Better correlations for other impacts based on GWP
        # These are rough approximations - real LCA would be better
        impact_ratios = {
            'GWP': 1.0,
            'ODEPL': 2.09e-7,    # kg CFC-11 eq / kg CO2 eq
            'IORAD': 1.43e-5,    # kBq U235 eq / kg CO2 eq  
            'PCHEM': 1.82e-4,    # kg NMVOC eq / kg CO2 eq
            'PMAT': 4.31e-5,     # kg PM2.5 eq / kg CO2 eq
            'HTOXnc': 3.71e-7,   # CTUh / kg CO2 eq
            'HTOXc': 1.84e-8,    # CTUh / kg CO2 eq
            'ACIDef': 3.24e-3,   # mol H+ eq / kg CO2 eq
            'FWEUT': 1.48e-5,    # kg P eq / kg CO2 eq
            'MWEUT': 2.18e-4,    # kg N eq / kg CO2 eq
            'TEUT': 1.93e-3,     # mol N eq / kg CO2 eq
            'FWTOX': 1.01,       # CTUe / kg CO2 eq
            'LUP': 129.4,        # Pt / kg CO2 eq
            'WDEPL': 0.0819,     # m3 water eq / kg CO2 eq
            'ADEPLf': 11.9,      # MJ / kg CO2 eq
            'ADEPLmu': 7.48e-6,  # kg Sb eq / kg CO2 eq
        }

        # Calculate ESA single score with proper normalization
        esa_score = 0.0
        for code, weight in ESA_WEIGHTS.items():
            # Calculate raw impact
            raw_value = gwp * impact_ratios[code]
            outputs[f'ESA_{code}'] = raw_value
            
            # Normalize (multiply by normalization factor from your table)
            normalized = raw_value * ESA_NORMALISATION[code]
            outputs[f'ESA_{code}_normalized'] = normalized
            
            # Weight and sum
            esa_score += weight * normalized

        outputs['ESA_single_score'] = esa_score

# ============================================================================
# COMPATIBILITY PATCHES
# ============================================================================

def apply_openmdao_patch():
    """Patch for lca4mdao compatibility with OpenMDAO 3.36"""
    if not LCA4MDAO_AVAILABLE:
        return
        
    try:
        import inspect
        from lca4mdao.component import LcaCalculationComponent as _LCC
        
        sig = inspect.signature(_LCC._setup_procs)
        if 'prob_meta' not in sig.parameters:
            _orig = _LCC._setup_procs
            
            def _setup_procs_compat(self, pathname, comm, prob_meta):
                return _orig(self, pathname, comm)
            
            _LCC._setup_procs = _setup_procs_compat
            print("Applied OpenMDAO compatibility patch")
    except Exception as e:
        print(f"Patch not needed or failed: {e}")

# Apply patch if needed
apply_openmdao_patch()