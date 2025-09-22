"""
FELIN-LCA Environmental Discipline with LCA4MDAO Integration
Properly uses ExplicitComponentLCA for dynamic parameter mapping
"""

import numpy as np
import openmdao.api as om
from lca4mdao.variable import ExplicitComponentLCA
from lca4mdao.component import LcaCalculationComponent
from lca4mdao.utilities import setup_bw, cleanup_parameters
import brightway2 as bw

# ========================================
# SETUP FUNCTIONS
# ========================================

def validate_material_design(cfrp_s1, al_s1, steel_s1, cfrp_s2, al_s2, steel_s2):
    """Validate material fractions for structural integrity"""
    
    # First stage constraints
    if al_s1 < 0.4:  # Minimum aluminum
        return False
    if cfrp_s1 > 0.3:  # Maximum CFRP
        return False
    if steel_s1 < 0.1:  # Minimum steel
        return False

    # Second stage constraints
    if al_s2 < 0.3:  # Minimum aluminum
        return False
    if cfrp_s2 > 0.5:  # Maximum CFRP
        return False
    if steel_s2 < 0.1:  # Minimum steel
        return False
    
    # Sum to 1.0
    if abs(cfrp_s1 + al_s1 + steel_s1 - 1.0) > 0.01:
        return False
    if abs(cfrp_s2 + al_s2 + steel_s2 - 1.0) > 0.01:
        return False
    
    return True

def setup_launcher_lca_environment():
    
    # Create custom launcher database
    if "launcher" in bw.databases:
        bw.Database("launcher").delete() # Remove existing to avoid conflicts
    
    launcher_db = bw.Database("launcher")
    launcher_db.register()
    
    eco = bw.Database("ecoinvent 3.8 cutoff")
    
    # Create launcher system activity
    launcher_system = launcher_db.new_activity(
        code="launcher_system",
        name="FELIN Launcher System",
        unit="unit",
        type="process"
    )
    
    # All material exchanges will be defined with formulas
    # These formulas will be controlled by LCA4MDAO parameters
    
    # Manufacturing materials
    launcher_system.new_exchange(
        input=eco.get('5f83b772ba1476f12d0b3ef634d4409b'),  # CFRP
        amount=0, formula="cfrp_total", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('8392648c098b86d088a9821ce11ed9dd'),  # Al structure
        amount=0, formula="aluminum_structure", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('03f6b6ba551e8541bf47842791abd3f7'),  # AlLi tank
        amount=0, formula="aluminum_tank", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('9b20aabdab5590c519bb3d717c77acf2'),  # Steel
        amount=0, formula="steel_total", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('223d2ca85f5c350a6a043725a2b71226'),  # Polyurethane for insulation
        amount=0, formula="insulation_total", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('52c4f6d2e1ec507b1ccc96056a761c0d'),  # Electronics
        amount=0, formula="electronics_total", type="technosphere"
    ).save()
    
    # Operation materials
    launcher_system.new_exchange(
        input=eco.get('53b5def592497847e2d0b4d62f2c4456'),  # LOX
        amount=0, formula="lox_total", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('a834063e527dafabe7d179a804a13f39'),  # LH2
        amount=0, formula="lh2_total", type="technosphere"
    ).save()
    
    # Transport and energy
    launcher_system.new_exchange(
        input=eco.get('41205d7711c0fad4403e4c2f9284b083'),  # Ship transport
        amount=0, formula="transport_tkm", type="technosphere"
    ).save()
    
    launcher_system.new_exchange(
        input=eco.get('3855bf674145307cd56a3fac8c83b643'),  # Electricity
        amount=0, formula="electricity_kwh", type="technosphere"
    ).save()
    
    # CO2 emissions (biosphere)
    co2 = bw.Database("biosphere3").search("carbon dioxide, fossil")[0]
    launcher_system.new_exchange(
        input=co2, amount=0, formula="co2_combustion", type="biosphere"
    ).save()
    
    # Production exchange
    launcher_system.new_exchange(
        input=launcher_system, amount=1, type="production"
    ).save()
    
    launcher_system.save()
    
    print("Environment setup complete")
    return True


# ========================================
# MASS BREAKDOWN CONSTANTS
# ========================================

MASS_BREAKDOWN = {
    'stage_1': {
        'tanks': 0.30,
        'interstage': 0.15,
        'thrust_frame': 0.10,
        'skirts': 0.10,
        'insulation': 0.05,
        'engines': 0.20,
        'avionics': 0.05,
        'other': 0.05
    },
    'stage_2': {
        'tanks': 0.35,
        'interstage': 0.10,
        'thrust_frame': 0.10,
        'skirts': 0.10,
        'insulation': 0.05,
        'engines': 0.15,
        'avionics': 0.10,
        'other': 0.05
    }
}

# ESA Impact Assessment Methods
ESA_METHODS = {
    'GWP': ('EF v3.0', 'climate change', 'global warming potential (GWP100)'),
    'ODEPL': ('EF v3.0', 'ozone depletion', 'ozone depletion potential (ODP) '),
    'IORAD': ('EF v3.0', 'ionising radiation: human health', 'human exposure efficiency relative to u235'),
    'PCHEM': ('EF v3.0', 'photochemical ozone formation: human health', 'tropospheric ozone concentration increase'),
    'PMAT': ('EF v3.0', 'particulate matter formation', 'impact on human health'),
    'HTOXnc': ('EF v3.0', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh) '),
    'HTOXc': ('EF v3.0', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh) '),
    'ACIDef': ('EF v3.0', 'acidification', 'accumulated exceedance (ae)'),
    'FWEUT': ('EF v3.0', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)'),
    'MWEUT': ('EF v3.0', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)'),
    'TEUT': ('EF v3.0', 'eutrophication: terrestrial', 'accumulated exceedance (AE) '),
    'FWTOX': ('EF v3.0', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe) '),
    'LUP': ('EF v3.0', 'land use', 'soil quality index'),
    'WDEPL': ('EF v3.0', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'),
    'ADEPLf': ('EF v3.0', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels'),
    'ADEPLmu': ('EF v3.0', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)'),
}

# ESA Normalization factors
ESA_NORMALIZATION = {
    'GWP': 0.0001235,
    'ODEPL': 18.64,
    'IORAD': 0.0002370,
    'PCHEM': 0.02463,
    'PMAT': 1680,
    'HTOXnc': 4354,
    'HTOXc': 59173,
    'ACIDef': 0.01800,
    'FWEUT': 0.6223,
    'MWEUT': 0.05116,
    'TEUT': 0.005658,
    'FWTOX': 0.00002343,
    'LUP': 0.000001220,
    'WDEPL': 0.00008719,
    'ADEPLf': 0.00001538,
    'ADEPLmu': 15.71,
}

# ESA Weighting factors
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

# ========================================
# MAIN COMPONENTS
# ========================================

class EnvironmentalMaterialMapping(ExplicitComponentLCA):
    """
    Maps launcher design variables to LCA parameters using LCA4MDAO
    This component properly uses ExplicitComponentLCA for automatic parameter updates
    """
    
    def setup(self):
        # INPUTS FROM OTHER DISCIPLINES
        self.add_input('Dry_mass_stage_1', val=30000.0) #units='kg'
        self.add_input('Dry_mass_stage_2', val=7000.0) #units='kg'
        self.add_input('Prop_mass_stage_1', val=250000.0) #units='kg'
        self.add_input('Prop_mass_stage_2', val=50000.0) #units='kg'
        self.add_input('GLOW', val=342000.0) #units='kg'
        self.add_input('OF_stage_1', val=5.5)
        self.add_input('OF_stage_2', val=5.5)
        self.add_input('N_eng_stage_1', val=8.0)
        self.add_input('N_eng_stage_2', val=1.0)
        self.add_input('payload_mass', val=5000.0) #units='kg'
        
        # Normalized material fractions
        self.add_input('cfrp_fraction_stage1', val=0.30)
        self.add_input('aluminum_fraction_stage1', val=0.60)
        self.add_input('steel_fraction_stage1', val=0.10)
        self.add_input('cfrp_fraction_stage2', val=0.40)
        self.add_input('aluminum_fraction_stage2', val=0.50)
        self.add_input('steel_fraction_stage2', val=0.10)
        
        # Trajectory data for performance metrics
        self.add_input('V_ascent', shape=4000) #units='m/s'
        self.add_input('alt_ascent', shape=4000) #units='m'
        self.add_input('nx_ascent', shape=4000) #units=None
        self.add_input('Nb_pt_ascent', val=195.)
        self.add_input('gamma_ascent', shape=4000) # rad
        
        # Get database references
        launcher_db = bw.Database("launcher")
        eco = bw.Database("ecoinvent 3.8 cutoff")
        launcher_system = launcher_db.get("launcher_system")
        
        # ========================================
        # LCA-MAPPED OUTPUTS (using ExplicitComponentLCA)
        # These automatically update Brightway2 parameters
        # ========================================
        
        # CFRP total
        self.add_output(
            'cfrp_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('5f83b772ba1476f12d0b3ef634d4409b').key,
            lca_name='cfrp_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Aluminum structure
        self.add_output(
            'aluminum_structure', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('8392648c098b86d088a9821ce11ed9dd').key,
            lca_name='aluminum_structure',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Aluminum tanks
        self.add_output(
            'aluminum_tank', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('03f6b6ba551e8541bf47842791abd3f7').key,
            lca_name='aluminum_tank',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Steel total
        self.add_output(
            'steel_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('9b20aabdab5590c519bb3d717c77acf2').key,
            lca_name='steel_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Insulation
        self.add_output(
            'insulation_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('223d2ca85f5c350a6a043725a2b71226').key,
            lca_name='insulation_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Electronics
        self.add_output(
            'electronics_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('52c4f6d2e1ec507b1ccc96056a761c0d').key,
            lca_name='electronics_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # LOX
        self.add_output(
            'lox_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('53b5def592497847e2d0b4d62f2c4456').key,
            lca_name='lox_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # LH2
        self.add_output(
            'lh2_total', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=eco.get('a834063e527dafabe7d179a804a13f39').key,
            lca_name='lh2_total',
            lca_units='kilogram',
            exchange_type='technosphere'
        )
        
        # Transport
        self.add_output(
            'transport_tkm', val=0.0, units='t*km', #tkm
            lca_parent=launcher_system.key,
            lca_key=eco.get('41205d7711c0fad4403e4c2f9284b083').key,
            lca_name='transport_tkm',
            lca_units='ton kilometer',
            exchange_type='technosphere'
        )
        
        # Electricity
        self.add_output(
            'electricity_kwh', val=0.0, units='kW*h', #kwh
            lca_parent=launcher_system.key,
            lca_key=eco.get('3855bf674145307cd56a3fac8c83b643').key,
            lca_name='electricity_kwh',
            lca_units='kilowatt hour',
            exchange_type='technosphere'
        )
        
        # CO2 emissions
        co2 = bw.Database("biosphere3").search("carbon dioxide, fossil")[0]
        self.add_output(
            'co2_combustion', val=0.0, units='kg',
            lca_parent=launcher_system.key,
            lca_key=co2.key,
            lca_name='co2_combustion',
            lca_units='kilogram',
            exchange_type='biosphere'
        )
        
        # ========================================
        # REGULAR OUTPUTS (not LCA-mapped)
        # ========================================
        
        # Performance metrics
        self.add_output('delta_v_achieved', val=7800.0, units='m/s')
        self.add_output('max_acceleration_g', val=3.5, units=None)
        
        # Material mass tracking (for monitoring)
        self.add_output('stage1_structure_cfrp', val=0.0, units='kg')
        self.add_output('stage1_structure_aluminum', val=0.0, units='kg')
        self.add_output('stage1_structure_steel', val=0.0, units='kg')
        self.add_output('stage2_structure_cfrp', val=0.0, units='kg')
        self.add_output('stage2_structure_aluminum', val=0.0, units='kg')
        self.add_output('stage2_structure_steel', val=0.0, units='kg')
        
    def compute(self, inputs, outputs):
        """
        Compute material masses and map to LCA parameters
        LCA4MDAO will automatically update Brightway2 parameters with these values
        """
        
        # Extract trajectory performance
        nb_pts = int(inputs['Nb_pt_ascent'][0])

        if nb_pts > 0:
            # Get trajectory data
            v_array = inputs['V_ascent'][:nb_pts]
            alt_array = inputs['alt_ascent'][:nb_pts]
            gamma_array = inputs['gamma_ascent'][:nb_pts]
            
            # Basic delta-V
            v_initial = v_array[0]
            v_final = v_array[-1]
            
            # Check orbital injection conditions
            final_alt = alt_array[-1]
            final_gamma = gamma_array[-1]
            
            # Orbital velocity at final altitude
            r_final = 6371000 + final_alt  # Earth radius + altitude
            v_circular = np.sqrt(398600.4418e9 / r_final)
            
            # Calculate actual achieved delta-V considering orbital mechanics
            # Account for gravity losses during ascent
            if final_alt > 100000:  # Above 100km
                # In orbit - use vis-viva equation
                outputs['delta_v_achieved'] = v_final - v_initial
            else:
                # Suborbital - add penalty
                outputs['delta_v_achieved'] = (v_final - v_initial) * 0.9
            
            # Maximum acceleration
            nx_array = inputs['nx_ascent'][:nb_pts]
            outputs['max_acceleration_g'] = np.max(np.abs(nx_array))
        else:
            outputs['delta_v_achieved'] = 0.0
            outputs['max_acceleration_g'] = 0.0
        
        # Get dry masses
        dry_s1 = inputs['Dry_mass_stage_1'][0]
        dry_s2 = inputs['Dry_mass_stage_2'][0]
        
        # Calculate structural component masses
        structural_mass_s1 = dry_s1 * (
            MASS_BREAKDOWN['stage_1']['interstage'] +
            MASS_BREAKDOWN['stage_1']['thrust_frame'] +
            MASS_BREAKDOWN['stage_1']['skirts']
        )
        structural_mass_s2 = dry_s2 * (
            MASS_BREAKDOWN['stage_2']['interstage'] +
            MASS_BREAKDOWN['stage_2']['thrust_frame'] +
            MASS_BREAKDOWN['stage_2']['skirts']
        )
        
        # Stage 1 structural materials
        outputs['stage1_structure_cfrp'] = structural_mass_s1 * inputs['cfrp_fraction_stage1'][0]
        outputs['stage1_structure_aluminum'] = structural_mass_s1 * inputs['aluminum_fraction_stage1'][0]
        outputs['stage1_structure_steel'] = structural_mass_s1 * inputs['steel_fraction_stage1'][0]
        
        # Stage 2 structural materials
        outputs['stage2_structure_cfrp'] = structural_mass_s2 * inputs['cfrp_fraction_stage2'][0]
        outputs['stage2_structure_aluminum'] = structural_mass_s2 * inputs['aluminum_fraction_stage2'][0]
        outputs['stage2_structure_steel'] = structural_mass_s2 * inputs['steel_fraction_stage2'][0]
        
        # Total CFRP (LCA-mapped output)
        outputs['cfrp_total'] = (
            outputs['stage1_structure_cfrp'] + 
            outputs['stage2_structure_cfrp']
        )
        
        # Total aluminum structure (LCA-mapped output)
        outputs['aluminum_structure'] = (
            outputs['stage1_structure_aluminum'] +
            outputs['stage2_structure_aluminum']
        )
        
        # Aluminum tanks (fixed material, LCA-mapped output)
        outputs['aluminum_tank'] = (
            dry_s1 * MASS_BREAKDOWN['stage_1']['tanks'] +
            dry_s2 * MASS_BREAKDOWN['stage_2']['tanks']
        )
        
        # Add aluminum from engines (70% of engine mass)
        engine_mass_total = (
            dry_s1 * MASS_BREAKDOWN['stage_1']['engines'] +
            dry_s2 * MASS_BREAKDOWN['stage_2']['engines']
        )
        outputs['aluminum_structure'] += engine_mass_total * 0.7
        
        # Total steel (LCA-mapped output)
        outputs['steel_total'] = (
            outputs['stage1_structure_steel'] +
            outputs['stage2_structure_steel'] +
            engine_mass_total * 0.3  # 30% of engines
        )
        
        # Insulation (LCA-mapped output)
        outputs['insulation_total'] = (
            dry_s1 * MASS_BREAKDOWN['stage_1']['insulation'] +
            dry_s2 * MASS_BREAKDOWN['stage_2']['insulation']
        )
        
        # Electronics/Avionics (LCA-mapped output)
        outputs['electronics_total'] = (
            dry_s1 * MASS_BREAKDOWN['stage_1']['avionics'] +
            dry_s2 * MASS_BREAKDOWN['stage_2']['avionics']
        )
        
        # Propellants (LCA-mapped outputs)
        prop_s1 = inputs['Prop_mass_stage_1'][0]
        prop_s2 = inputs['Prop_mass_stage_2'][0]
        of_s1 = inputs['OF_stage_1'][0]
        of_s2 = inputs['OF_stage_2'][0]
        
        outputs['lox_total'] = (
            prop_s1 * (of_s1 / (1 + of_s1)) +
            prop_s2 * (of_s2 / (1 + of_s2))
        )
        
        outputs['lh2_total'] = (
            prop_s1 * (1 / (1 + of_s1)) +
            prop_s2 * (1 / (1 + of_s2))
        )
        
        # Transport (LCA-mapped output)
        total_dry_mass = dry_s1 + dry_s2
        outputs['transport_tkm'] = (total_dry_mass / 1000.0) * 7000  # 7000 km to Kourou
        
        # Assembly electricity (LCA-mapped output)
        outputs['electricity_kwh'] = inputs['GLOW'][0] * 0.2  # 0.2 kWh per kg
        
        # CO2 from combustion (LCA-mapped output) not yet used
        outputs['co2_combustion'] = 0.0


class LauncherLCACalculation(LcaCalculationComponent):
    """
    LCA Calculation Component that performs the actual impact assessment
    Uses the parameters set by EnvironmentalMaterialMapping
    """
    
    def setup(self):

        # IMPORTANT: Declare inputs for all LCA-mapped variables
        # These come from EnvironmentalMaterialMapping outputs
        self.add_input('cfrp_total', val=0.0, units='kg')
        self.add_input('aluminum_structure', val=0.0, units='kg')
        self.add_input('aluminum_tank', val=0.0, units='kg')
        self.add_input('steel_total', val=0.0, units='kg')
        self.add_input('insulation_total', val=0.0, units='kg')
        self.add_input('electronics_total', val=0.0, units='kg')
        self.add_input('lox_total', val=0.0, units='kg')
        self.add_input('lh2_total', val=0.0, units='kg')
        self.add_input('transport_tkm', val=0.0, units='t*km')
        self.add_input('electricity_kwh', val=0.0, units='kW*h')
        self.add_input('co2_combustion', val=0.0, units='kg')

        # Define functional unit (1 launcher system)
        launcher_db = bw.Database("launcher")
        launcher_system = launcher_db.get("launcher_system")
        functional_unit = {launcher_system.key: 1}
        
                # Add LCA outputs for all 16 ESA categories
        for code, method_key in ESA_METHODS.items():
            self.add_lca_output(
                name=f'{code}_total',
                functional_unit=functional_unit,
                method_key=method_key
            )


class ESASingleScoreCalculator(om.ExplicitComponent):

    def setup(self):
        # Inputs: raw impact values from LCA
        for code in ESA_METHODS.keys():
            self.add_input(f'{code}_total', val=0.0, units=None)
        
        # Outputs
        self.add_output('ESA_single_score', val=100.0, ref=100.0)
        
        # Individual normalized and weighted scores for analysis
        for code in ESA_METHODS.keys():
            self.add_output(f'{code}_normalized', val=0.0)
            self.add_output(f'{code}_weighted', val=0.0)

    def compute(self, inputs, outputs):
        """
        Calculate ESA single score using normalization and weighting
        """
        
        esa_score = 0.0
        
        for code in ESA_METHODS.keys():
            # Get raw value
            raw_value = inputs[f'{code}_total'][0]
            
            # Normalize
            normalized = raw_value * ESA_NORMALIZATION[code]
            outputs[f'{code}_normalized'] = normalized
            
            # Weight
            weighted = normalized * ESA_WEIGHTS[code]
            outputs[f'{code}_weighted'] = weighted
            
            # Add to total score
            esa_score += weighted
        
        outputs['ESA_single_score'] = esa_score

class MaterialNormalizer(om.ExplicitComponent):
    """
    Ensures material fractions sum to 1.0 for each stage
    """
    
    def setup(self):
        # Raw (unnormalized) fractions as inputs
        self.add_input('raw_cfrp_s1', val=0.30)
        self.add_input('raw_aluminum_s1', val=0.60)
        self.add_input('raw_steel_s1', val=0.10)
        
        self.add_input('raw_cfrp_s2', val=0.40)
        self.add_input('raw_aluminum_s2', val=0.50)
        self.add_input('raw_steel_s2', val=0.10)
        
        # Normalized fractions as outputs
        self.add_output('cfrp_fraction_stage1', val=0.30)
        self.add_output('aluminum_fraction_stage1', val=0.60)
        self.add_output('steel_fraction_stage1', val=0.10)
        
        self.add_output('cfrp_fraction_stage2', val=0.40)
        self.add_output('aluminum_fraction_stage2', val=0.50)
        self.add_output('steel_fraction_stage2', val=0.10)
        
    def compute(self, inputs, outputs):
        """
        Normalize material fractions to sum to 1.0
        """
        
        # Stage 1
        sum_s1 = inputs['raw_cfrp_s1'][0] + inputs['raw_aluminum_s1'][0] + inputs['raw_steel_s1'][0]
        if sum_s1 > 0:
            outputs['cfrp_fraction_stage1'] = inputs['raw_cfrp_s1'][0] / sum_s1
            outputs['aluminum_fraction_stage1'] = inputs['raw_aluminum_s1'][0] / sum_s1
            outputs['steel_fraction_stage1'] = inputs['raw_steel_s1'][0] / sum_s1
        else:
            # Default to aluminum if all zeros
            outputs['cfrp_fraction_stage1'] = 0.0
            outputs['aluminum_fraction_stage1'] = 1.0
            outputs['steel_fraction_stage1'] = 0.0
        
        # Stage 2
        sum_s2 = inputs['raw_cfrp_s2'][0] + inputs['raw_aluminum_s2'][0] + inputs['raw_steel_s2'][0]
        if sum_s2 > 0:
            outputs['cfrp_fraction_stage2'] = inputs['raw_cfrp_s2'][0] / sum_s2
            outputs['aluminum_fraction_stage2'] = inputs['raw_aluminum_s2'][0] / sum_s2
            outputs['steel_fraction_stage2'] = inputs['raw_steel_s2'][0] / sum_s2
        else:
            outputs['cfrp_fraction_stage2'] = 0.0
            outputs['aluminum_fraction_stage2'] = 1.0
            outputs['steel_fraction_stage2'] = 0.0