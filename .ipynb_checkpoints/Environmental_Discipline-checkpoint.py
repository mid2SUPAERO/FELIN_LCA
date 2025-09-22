"""
FELIN-LCA Environmental Discipline Module
Master Thesis: Integration of LCA into FELIN MDO Framework
Cleaned version without unnecessary manufacturing processes
Compatible with FELIN's LOX/LH2 propulsion system
"""

import numpy as np
import openmdao.api as om
import os
from pathlib import Path

# Only set if not provided by the OS/user
os.environ.setdefault("BW2DIR", r"C:\Users\joana\AppData\Local\pylca\Brightway3")

#os.environ["BW2DIR"] = os.environ.get("BW2DIR", str(Path.home() / ".brightway2"))
import brightway2 as bw
bw.projects.set_current("LCA_FELIN")

# Validate ecoinvent database
if "ecoinvent 3.8 cutoff" not in bw.databases or len(bw.Database("ecoinvent 3.8 cutoff")) < 10000:
    raise RuntimeError("Ecoinvent not found or empty in project 'LCA_FELIN'.")

from lca4mdao.component import LcaCalculationComponent
from lca4mdao.variable import ExplicitComponentLCA

# ============================================================================
# MATERIAL MAPPING TO ECOINVENT (ONLY EXISTING ACTIVITIES)
# ============================================================================

FELIN_MATERIALS = {
    # Structural materials
    'carbon_fibre': ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b'),
    'aluminium_almg3': ('ecoinvent 3.8 cutoff', '3d66c7f5f8d813a5b63b2d19a41ec763'),
    'aluminium_alloy': ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7'),
    'titanium': ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072'),
    'nickel': ('ecoinvent 3.8 cutoff', '6f592c599b70d14247116fdf44a0824a'),
    'steel_stainless': ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2'),
    
    # Electronics and systems
    'electronic_active': ('ecoinvent 3.8 cutoff', '52c4f6d2e1ec507b1ccc96056a761c0d'),
    'electronic_passive': ('ecoinvent 3.8 cutoff', 'b1b65fe4d00b29f2299c72b894a3c0a0'),
    'wire_copper': ('ecoinvent 3.8 cutoff', 'f8586b86fe8ac595be9f6b18e9b94488'),
    'battery_lib': ('ecoinvent 3.8 cutoff', 'b2feecd5152754c08303bc84dc371b68'),
    'motor_electric': ('ecoinvent 3.8 cutoff', '0a45c922ec9f5a8345c88fb3ecc28b6f'),
    
    # Propellants (LOX/LH2 for FELIN)
    'liquid_oxygen': ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456'),
    'liquid_hydrogen': ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39'),
    
    # Transport and operations
    'transport_ship': ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083'),
    'electricity_fr': ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643'),
}

# ============================================================================
# ESA SINGLE SCORE METHODOLOGY
# ============================================================================

ESA_METHOD_OPTIONS = {
    'GWP': [('ipcc 2013', 'climate change', 'gwp 100a'), ('ef v3.0', 'climate change')],
    'ODEPL': [('ef v3.0', 'ozone depletion'), ('cml', 'ozone depletion')],
    'IORAD': [('ef v3.0', 'ionising radiation'), ('cml', 'ionising radiation')],
    'PCHEM': [('ef v3.0', 'photochemical ozone'), ('cml', 'photochemical')],
    'PMAT': [('ef v3.0', 'particulate matter'), ('cml', 'particulate')],
    'HTOXnc': [('ef v3.0', 'human toxicity', 'non-carcinogenic'), ('usetox', 'non-carcinogenic')],
    'HTOXc': [('ef v3.0', 'human toxicity', 'carcinogenic'), ('usetox', 'carcinogenic')],
    'ACIDef': [('ef v3.0', 'acidification'), ('cml', 'acidification')],
    'FWEUT': [('ef v3.0', 'eutrophication', 'freshwater'), ('cml', 'freshwater')],
    'MWEUT': [('ef v3.0', 'eutrophication', 'marine'), ('cml', 'marine')],
    'TEUT': [('ef v3.0', 'eutrophication', 'terrestrial'), ('cml', 'terrestrial')],
    'FWTOX': [('recipe midpoint', 'freshwater ecotoxicity'), ('cml', 'ecotoxicity')],
    'LUP': [('ef v3.0', 'land use'), ('ef v2.0', 'land use')],
    'WDEPL': [('ef v3.0', 'water use'), ('ef v2.0', 'water use')],
    'ADEPLf': [('cml v4.8', 'abiotic depletion', 'fossil'), ('cml', 'fossil')],
    'ADEPLmu': [('cml v4.8', 'abiotic depletion', 'elements'), ('cml', 'elements')],
}

ESA_NORMALISATION = {
    'GWP': 1.235e-4, 'ODEPL': 18.64, 'IORAD': 2.37e-4, 'PCHEM': 0.02463,
    'PMAT': 1680.0, 'HTOXnc': 4354.0, 'HTOXc': 59173.0, 'ACIDef': 0.01800,
    'FWEUT': 0.6223, 'MWEUT': 0.05116, 'TEUT': 0.005658, 'FWTOX': 2.343e-5,
    'LUP': 1.22e-6, 'WDEPL': 8.719e-6, 'ADEPLf': 1.538e-5, 'ADEPLmu': 15.71,
}

ESA_WEIGHTS = {
    'GWP': 0.2106, 'ODEPL': 0.0631, 'IORAD': 0.0501, 'PCHEM': 0.0478,
    'PMAT': 0.0896, 'HTOXnc': 0.0184, 'HTOXc': 0.0213, 'ACIDef': 0.0620,
    'FWEUT': 0.0280, 'MWEUT': 0.0296, 'TEUT': 0.0371, 'FWTOX': 0.0192,
    'LUP': 0.0794, 'WDEPL': 0.0851, 'ADEPLf': 0.0832, 'ADEPLmu': 0.0755,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_first_matching(options):
    """Find first matching LCIA method from options list"""
    methods = list(bw.methods)
    for opt in options:
        for m in methods:
            s = str(m).lower()
            if all(tok.lower() in s for tok in opt):
                return m
    return methods[0] if methods else None

def create_launcher_database(overwrite=False):
    """Create FELIN launcher component database structure"""
    data = {
        ('felin_launcher', 'complete_launcher'): {
            'name': 'FELIN complete launcher',
            'unit': 'unit',
            'exchanges': [],
        },
        ('felin_launcher', 'stage1_structure'): {
            'name': 'Stage 1 structural components',
            'unit': 'kg',
            'exchanges': [],
        },
        ('felin_launcher', 'stage2_structure'): {
            'name': 'Stage 2 structural components',
            'unit': 'kg',
            'exchanges': [],
        },
        ('felin_launcher', 'propulsion_system'): {
            'name': 'Propulsion system components',
            'unit': 'kg',
            'exchanges': [],
        },
        ('felin_launcher', 'propellants'): {
            'name': 'LOX/LH2 propellants',
            'unit': 'kg',
            'exchanges': [],
        },
        ('felin_launcher', 'operations'): {
            'name': 'Launch operations and transport',
            'unit': 'unit',
            'exchanges': [],
        },
    }
    
    if 'felin_launcher' in bw.databases:
        if overwrite:
            del bw.databases['felin_launcher']
        else:
            return True
    
    bw.Database('felin_launcher').write(data)
    print("✓ FELIN launcher database created")
    return True

# ============================================================================
# FELIN ENVIRONMENTAL DISCIPLINE
# ============================================================================

class FELINEnvironmentalDiscipline(ExplicitComponentLCA):
    """
    Environmental discipline for FELIN launcher
    Compatible with FELIN's existing structure
    """
    
    def setup(self):
        # === INPUTS FROM FELIN DISCIPLINES (matching launch_vehicle_group.py) ===
        
        # From Structure disciplines
        self.add_input('Dry_mass_stage_1', val=45000.0, units='kg')
        self.add_input('Dry_mass_stage_2', val=6000.0, units='kg')
        
        # From Trajectory discipline
        self.add_input('Prop_mass_stage_1', val=350000.0, units='kg')
        self.add_input('Prop_mass_stage_2', val=75000.0, units='kg')
        
        # From indeps (configuration)
        self.add_input('N_eng_stage_1', val=7.0)
        self.add_input('N_eng_stage_2', val=1.0)
        self.add_input('OF_stage_1', val=2.7)
        self.add_input('OF_stage_2', val=2.7)
        self.add_input('Thrust_stage_1', val=1000.0, units='kN')
        self.add_input('Thrust_stage_2', val=1150.0, units='kN')
        
        # From Propulsion discipline
        self.add_input('Isp_stage_1', val=330.0, units='s')
        self.add_input('Isp_stage_2', val=330.0, units='s')
        
        # Additional needed inputs
        self.add_input('Diameter_stage_1', val=4.6, units='m')
        self.add_input('Diameter_stage_2', val=4.6, units='m')
        self.add_input('Mass_flow_rate_stage_1', val=219.0, units='kg/s')
        self.add_input('Mass_flow_rate_stage_2', val=219.0, units='kg/s')
        
        # Mission parameters (set default, can be overridden)
        self.add_input('payload_mass', val=15000.0, units='kg')
        
        # Calculate GLOW internally
        self.add_input('GLOW', val=450000.0, units='kg', 
                      desc='Gross Lift-Off Weight (calculated from masses)')
        
        # === DESIGN VARIABLES FOR OPTIMIZATION ===
        
        # Material composition fractions (these can be optimization variables)
        self.add_input('cfrp_fraction_stage1', val=0.25)
        self.add_input('aluminum_fraction_stage1', val=0.65)
        self.add_input('steel_fraction_stage1', val=0.10)
        
        self.add_input('cfrp_fraction_stage2', val=0.35)
        self.add_input('aluminum_fraction_stage2', val=0.55)
        self.add_input('steel_fraction_stage2', val=0.10)
        
        self.add_input('engine_nickel_fraction', val=0.60)
        self.add_input('engine_steel_fraction', val=0.30)
        self.add_input('engine_titanium_fraction', val=0.10)
        
        # === LCA OUTPUTS (materials for LCA calculation) ===
        
        # Stage 1 materials
        self.add_output('stage1_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage1_structure"),
                       lca_key=FELIN_MATERIALS['carbon_fibre'],
                       lca_units='kilogram')
        
        self.add_output('stage1_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage1_structure"),
                       lca_key=FELIN_MATERIALS['aluminium_alloy'],
                       lca_units='kilogram')
        
        self.add_output('stage1_steel_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage1_structure"),
                       lca_key=FELIN_MATERIALS['steel_stainless'],
                       lca_units='kilogram')
        
        # Stage 2 materials
        self.add_output('stage2_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage2_structure"),
                       lca_key=FELIN_MATERIALS['carbon_fibre'],
                       lca_units='kilogram')
        
        self.add_output('stage2_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage2_structure"),
                       lca_key=FELIN_MATERIALS['aluminium_alloy'],
                       lca_units='kilogram')
        
        self.add_output('stage2_steel_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "stage2_structure"),
                       lca_key=FELIN_MATERIALS['steel_stainless'],
                       lca_units='kilogram')
        
        # Engine materials
        self.add_output('engines_nickel_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "propulsion_system"),
                       lca_key=FELIN_MATERIALS['nickel'],
                       lca_units='kilogram')
        
        self.add_output('engines_steel_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "propulsion_system"),
                       lca_key=FELIN_MATERIALS['steel_stainless'],
                       lca_units='kilogram')
        
        self.add_output('engines_titanium_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "propulsion_system"),
                       lca_key=FELIN_MATERIALS['titanium'],
                       lca_units='kilogram')
        
        # Propellants (LOX/LH2)
        self.add_output('total_lox_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "propellants"),
                       lca_key=FELIN_MATERIALS['liquid_oxygen'],
                       lca_units='kilogram')
        
        self.add_output('total_fuel_mass', val=0.0, units='kg',
                       lca_parent=("felin_launcher", "propellants"),
                       lca_key=FELIN_MATERIALS['liquid_hydrogen'],
                       lca_units='kilogram')
        
        # Operations
        self.add_output('transport_operations', val=0.0, units='unitless',
                       lca_parent=("felin_launcher", "operations"),
                       lca_key=FELIN_MATERIALS['transport_ship'],
                       lca_units='ton kilometer')
        
        self.add_output('launch_operations_electricity', val=0.0, units='unitless',
                       lca_parent=("felin_launcher", "operations"),
                       lca_key=FELIN_MATERIALS['electricity_fr'],
                       lca_units='kilowatt hour')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Compute material masses and operations for LCA
        """
        
        # Extract inputs
        dry_mass_s1 = inputs['Dry_mass_stage_1']
        dry_mass_s2 = inputs['Dry_mass_stage_2']
        prop_mass_s1 = inputs['Prop_mass_stage_1']
        prop_mass_s2 = inputs['Prop_mass_stage_2']
        
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
        
        # Calculate GLOW (total mass at liftoff)
        glow = dry_mass_s1 + dry_mass_s2 + prop_mass_s1 + prop_mass_s2 + inputs['payload_mass']
        
        # === CALCULATE STRUCTURE MATERIALS ===
        # Normalize fractions to ensure they sum to 1.0
        total_frac_s1 = cfrp_frac_s1 + al_frac_s1 + steel_frac_s1
        if total_frac_s1 > 0:
            outputs['stage1_cfrp_mass'] = dry_mass_s1 * (cfrp_frac_s1 / total_frac_s1)
            outputs['stage1_aluminum_mass'] = dry_mass_s1 * (al_frac_s1 / total_frac_s1)
            outputs['stage1_steel_mass'] = dry_mass_s1 * (steel_frac_s1 / total_frac_s1)
        else:
            outputs['stage1_cfrp_mass'] = dry_mass_s1 * 0.25
            outputs['stage1_aluminum_mass'] = dry_mass_s1 * 0.65
            outputs['stage1_steel_mass'] = dry_mass_s1 * 0.10
        
        total_frac_s2 = cfrp_frac_s2 + al_frac_s2 + steel_frac_s2
        if total_frac_s2 > 0:
            outputs['stage2_cfrp_mass'] = dry_mass_s2 * (cfrp_frac_s2 / total_frac_s2)
            outputs['stage2_aluminum_mass'] = dry_mass_s2 * (al_frac_s2 / total_frac_s2)
            outputs['stage2_steel_mass'] = dry_mass_s2 * (steel_frac_s2 / total_frac_s2)
        else:
            outputs['stage2_cfrp_mass'] = dry_mass_s2 * 0.35
            outputs['stage2_aluminum_mass'] = dry_mass_s2 * 0.55
            outputs['stage2_steel_mass'] = dry_mass_s2 * 0.10
        
        # === CALCULATE ENGINE MATERIALS ===
        # Engine mass estimation based on thrust and number of engines
        # Using typical mass/thrust ratio for LOX/LH2 engines
        thrust_per_engine_s1 = thrust_s1 / max(n_eng_s1, 1)
        thrust_per_engine_s2 = thrust_s2 / max(n_eng_s2, 1)
        
        # Engine mass scaling: ~1.5 kg per kN thrust for LOX/LH2 engines
        engine_mass_s1_per_unit = thrust_per_engine_s1 * 1.5
        engine_mass_s2_per_unit = thrust_per_engine_s2 * 1.5
        
        total_engine_mass = (n_eng_s1 * engine_mass_s1_per_unit + 
                            n_eng_s2 * engine_mass_s2_per_unit)
        
        # Normalize engine material fractions
        total_eng_frac = ni_frac + steel_eng_frac + ti_frac
        if total_eng_frac > 0:
            outputs['engines_nickel_mass'] = total_engine_mass * (ni_frac / total_eng_frac)
            outputs['engines_steel_mass'] = total_engine_mass * (steel_eng_frac / total_eng_frac)
            outputs['engines_titanium_mass'] = total_engine_mass * (ti_frac / total_eng_frac)
        else:
            outputs['engines_nickel_mass'] = total_engine_mass * 0.60
            outputs['engines_steel_mass'] = total_engine_mass * 0.30
            outputs['engines_titanium_mass'] = total_engine_mass * 0.10
        
        # === CALCULATE PROPELLANTS (LOX/LH2) ===
        # Based on O/F ratio from FELIN's propulsion model
        # O/F = mass_oxidizer / mass_fuel
        
        # Stage 1 propellants
        if of_s1 > 0:
            lox_mass_s1 = (of_s1 / (1 + of_s1)) * prop_mass_s1
            lh2_mass_s1 = (1 / (1 + of_s1)) * prop_mass_s1
        else:
            # Default for LOX/LH2 if O/F not specified
            lox_mass_s1 = prop_mass_s1 * 0.73  # ~2.7 O/F ratio
            lh2_mass_s1 = prop_mass_s1 * 0.27
        
        # Stage 2 propellants
        if of_s2 > 0:
            lox_mass_s2 = (of_s2 / (1 + of_s2)) * prop_mass_s2
            lh2_mass_s2 = (1 / (1 + of_s2)) * prop_mass_s2
        else:
            lox_mass_s2 = prop_mass_s2 * 0.73
            lh2_mass_s2 = prop_mass_s2 * 0.27
        
        outputs['total_lox_mass'] = lox_mass_s1 + lox_mass_s2
        outputs['total_fuel_mass'] = lh2_mass_s1 + lh2_mass_s2
        
        # === CALCULATE OPERATIONS ===
        
        # Transport: simplified model (ton-km)
        # Assume 5000 km transport distance for launcher components
        transport_distance_km = 5000.0
        launcher_dry_mass_tons = (dry_mass_s1 + dry_mass_s2 + total_engine_mass) / 1000.0
        outputs['transport_operations'] = launcher_dry_mass_tons * transport_distance_km
        
        # Launch operations electricity (kWh)
        # Estimate based on GLOW
        electricity_per_kg = 0.15  # kWh per kg GLOW (typical for launch operations)
        outputs['launch_operations_electricity'] = glow * electricity_per_kg


# ============================================================================
# LCA CALCULATION COMPONENT
# ============================================================================

class FELINLCACalculation(LcaCalculationComponent):
    """
    LCA calculation component with ESA scoring
    """
    def setup(self):
        # -------- Inputs --------
        material_inputs = [
            'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
            'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass',
            'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass',
            'total_lox_mass', 'total_fuel_mass'
        ]
        for name in material_inputs:
            self.add_input(name, units='kg')

        # Operations (values already expressed in the correct LCA units by the upstream component)
        self.add_input('transport_operations', units='unitless')
        self.add_input('launch_operations_electricity', units='unitless')

        # Mission parameters (used only for the simple fallback calc)
        self.add_input('payload_mass', val=15000.0, units='kg')
        self.add_input('Dry_mass_stage_1', val=45000.0, units='kg')
        self.add_input('Dry_mass_stage_2', val=6000.0, units='kg')
        self.add_input('Prop_mass_stage_1', val=350000.0, units='kg')
        self.add_input('Prop_mass_stage_2', val=75000.0, units='kg')

        # Grouping of foreground activities (must match your ExplicitComponentLCA lca_parent)
        group_dict = {
            ("felin_launcher", "stage1_structure"): 1,
            ("felin_launcher", "stage2_structure"): 1,
            ("felin_launcher", "propulsion_system"): 1,
            ("felin_launcher", "propellants"): 1,
            ("felin_launcher", "operations"): 1,
        }

        # --- Fallback flag: set True if any method is missing/fails ---
        self._use_fallback = False

        # -------- ESA category outputs --------
        self.ESA_methods = {}
        for code, options in ESA_METHOD_OPTIONS.items():
            method = _find_first_matching(options)
            if method is not None:
                self.ESA_methods[code] = method
                try:
                    self.add_lca_output(f'ESA_{code}', group_dict, method_key=method, units='unitless')
                    continue  # success for this code
                except Exception:
                    # Couldn't create the LCA-backed output; fall back to a plain output
                    self._use_fallback = True
            else:
                # No suitable method found → fallback
                self._use_fallback = True

            # Ensure the variable exists so ESASingleScore can read it
            self.add_output(f'ESA_{code}', val=0.0, units='unitless')

        # -------- Primary GWP output --------
        gwp_method = _find_first_matching([('ipcc 2013', 'climate change', 'gwp 100a')])
        if gwp_method is not None:
            try:
                self.add_lca_output('GWP_total', group_dict, method_key=gwp_method, units='kg')
            except Exception:
                self.add_output('GWP_total', val=0.0, units='kg')
                self._use_fallback = True
        else:
            self.add_output('GWP_total', val=0.0, units='kg')
            self._use_fallback = True

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        If any LCIA method was unavailable or failed during setup, compute a simple
        fallback estimate for GWP_total and ESA_* so downstream ESASingleScore can run.
        Otherwise, do nothing here (LcaCalculationComponent handles it internally).
        """
        if not getattr(self, '_use_fallback', False):
            return  # normal LCA path already computed by base class

        # --- Simple fallback (documented approximations) ---
        gwp_factors = {
            'carbon_fibre': 14.1,
            'aluminium':    8.9,
            'steel':        2.3,
            'nickel':       16.0,
            'titanium':     12.0,
            'lox':           0.8,
            'lh2':          12.0,
            'transport_tkm': 0.02,
            'electricity_kwh': 0.4  # French grid
        }

        gwp = (
            inputs['stage1_cfrp_mass']        * gwp_factors['carbon_fibre'] +
            inputs['stage2_cfrp_mass']        * gwp_factors['carbon_fibre'] +
            inputs['stage1_aluminum_mass']    * gwp_factors['aluminium'] +
            inputs['stage2_aluminum_mass']    * gwp_factors['aluminium'] +
            inputs['stage1_steel_mass']       * gwp_factors['steel'] +
            inputs['stage2_steel_mass']       * gwp_factors['steel'] +
            inputs['engines_nickel_mass']     * gwp_factors['nickel'] +
            inputs['engines_steel_mass']      * gwp_factors['steel'] +
            inputs['engines_titanium_mass']   * gwp_factors['titanium'] +
            inputs['total_lox_mass']          * gwp_factors['lox'] +
            inputs['total_fuel_mass']         * gwp_factors['lh2'] +
            inputs['transport_operations']    * gwp_factors['transport_tkm'] +
            inputs['launch_operations_electricity'] * gwp_factors['electricity_kwh']
        )

        outputs['GWP_total'] = gwp

        # Fill every ESA_* so ESASingleScore always has inputs
        for code in ESA_METHOD_OPTIONS.keys():
            name = f'ESA_{code}'
            if name in outputs:  # only if it was created in setup
                if code == 'GWP':
                    outputs[name] = gwp
                else:
                    outputs[name] = 0.001 * gwp  # placeholder correlation

# ===== ESA aggregation and env metrics =====
class ESASingleScore(om.ExplicitComponent):
    def setup(self):
        # One input per ESA code present in ESA_WEIGHTS
        for code in ESA_WEIGHTS.keys():
            self.add_input(f'ESA_{code}', val=0.0, units='unitless')
        self.add_output('ESA_single_score', val=0.0, units='unitless')

    def compute(self, inputs, outputs):
        # Normalise and weight (ESA logic)
        score = 0.0
        for code, w in ESA_WEIGHTS.items():
            x = inputs[f'ESA_{code}']
            n = ESA_NORMALISATION.get(code, 1.0)
            score += w * (x / n)
        outputs['ESA_single_score'] = score


class EnvironmentalMetrics(om.ExplicitComponent):
    def setup(self):
        self.add_input('GWP_total', val=0.0, units='kg')
        self.add_input('payload_mass', val=15000.0, units='kg')
        self.add_input('GLOW', val=450000.0, units='kg')
        self.add_input('Isp_stage_1', val=330.0, units='s')
        self.add_input('Isp_stage_2', val=330.0, units='s')
        self.add_input('ESA_single_score', val=0.0)  # <-- add this

        self.add_output('GWP_per_kg_payload', val=0.0, units='kg/kg')
        self.add_output('environmental_score', val=0.0)           # now = ESA single score
        self.add_output('environmental_efficiency', val=0.0)      # payload per GWP
        self.add_output('performance_env_ratio', val=0.0)         # Isp2 / GWP (toy metric)
        self.add_output('mass_efficiency', val=0.0)               # payload / GLOW

    def compute(self, inputs, outputs):
        payload = max(float(inputs['payload_mass']), 1e-9)
        gwp     = float(inputs['GWP_total'])
        glow    = max(float(inputs['GLOW']), 1e-9)
        isp2    = float(inputs['Isp_stage_2'])

        outputs['GWP_per_kg_payload']   = gwp / payload
        outputs['environmental_score']  = float(inputs['ESA_single_score'])  # <-- main env metric
        outputs['environmental_efficiency'] = payload / (gwp + 1e-9)
        outputs['performance_env_ratio']    = isp2 / (gwp + 1e-9)
        outputs['mass_efficiency']          = payload / glow

# ===== Group that connects the discipline and calculator and exposes final signals =====
class FELINEnvironmentalGroup(om.Group):
    def setup(self):
        # 1) Materials/operations builder
        # 2) LCA calculation (lca4mdao)
        self.add_subsystem(
            'lca_calc', FELINLCACalculation(),
            promotes_inputs=[
                'stage1_cfrp_mass','stage1_aluminum_mass','stage1_steel_mass',
                'stage2_cfrp_mass','stage2_aluminum_mass','stage2_steel_mass',
                'engines_nickel_mass','engines_steel_mass','engines_titanium_mass',
                'total_lox_mass','total_fuel_mass',
                'transport_operations','launch_operations_electricity',
                'payload_mass','Dry_mass_stage_1','Dry_mass_stage_2',
                'Prop_mass_stage_1','Prop_mass_stage_2'
            ],
            promotes_outputs=['GWP_total'] + [f'ESA_{c}' for c in ESA_WEIGHTS.keys()]
        )


        # 2) LCA calculation (lca4mdao)
        self.add_subsystem('lca_calc', FELINLCACalculation(),
            promotes_inputs=[
                'stage1_cfrp_mass','stage1_aluminum_mass','stage1_steel_mass',
                'stage2_cfrp_mass','stage2_aluminum_mass','stage2_steel_mass',
                'engines_nickel_mass','engines_steel_mass','engines_titanium_mass',
                'total_lox_mass','total_fuel_mass',
                'transport_operations','launch_operations_electricity',
                'payload_mass','Dry_mass_stage_1','Dry_mass_stage_2',
                'Prop_mass_stage_1','Prop_mass_stage_2'
            ],
            promotes_outputs=[
                'GWP_total',  # + all ESA_* that FELINLCACalculation defines
            ])

        # 3) ESA single score aggregator
        self.add_subsystem('esa_score', ESASingleScore(),
            promotes_inputs=[f'ESA_{c}' for c in ESA_WEIGHTS.keys()],
            promotes_outputs=['ESA_single_score'])

        # 4) Extra derived metrics
        # 4) Extra derived metrics
        self.add_subsystem(
            'env_metrics', EnvironmentalMetrics(),
            promotes_inputs=['GWP_total','payload_mass','GLOW','Isp_stage_1','Isp_stage_2','ESA_single_score'],
            promotes_outputs=['GWP_per_kg_payload','environmental_score',
                            'environmental_efficiency','performance_env_ratio','mass_efficiency']
        )



# No-op patch so the import in Launch_vehicle_Group works
def apply_openmdao_patch():
    return None
