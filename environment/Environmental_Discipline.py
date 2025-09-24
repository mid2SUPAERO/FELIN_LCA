# -*- coding: utf-8 -*-
"""
Environmental Discipline for FELIN Launcher LCA
Simplified version using direct Brightway2 integration without LCA4MDAO
Integrates with k_SM material system from structural discipline
"""

import numpy as np
from openmdao.api import ExplicitComponent
import brightway2 as bw

try:
    import brightway2 as bw
    bw.projects.set_current("LCA_FELIN")
    
    if "ecoinvent 3.8 cutoff" not in bw.databases:
        raise ImportError("ERROR: Ecoinvent 3.8 cutoff database not found.")
    
    print("✓ Brightway2 and ecoinvent successfully loaded")
    LCA_ENABLED = True
    
except ImportError as e:
    print(f"WARNING: {e}")
    print("Environmental discipline running in degraded mode")
    LCA_ENABLED = False

from environment.material_helpers import MaterialConverter

eco = bw.Database("ecoinvent 3.8 cutoff")

# ========================================
# ESA IMPACT ASSESSMENT METHODS
# ========================================

ESA_METHODS = {
    # Climate & radiation
    'GWP'   : ('EF v3.0', 'climate change', 'global warming potential (GWP100)'),
    'IORAD' : ('EF v3.0', 'ionising radiation: human health', 'human exposure efficiency relative to u235'),

    # Ozone & acidification
    'ODEPL' : ('EF v3.0', 'ozone depletion', 'ozone depletion potential (ODP) '),   # NOTE trailing space
    'ACIDef': ('EF v3.0', 'acidification', 'accumulated exceedance (ae)'),

    # Photochemical ozone & particulate matter
    'PCHEM' : ('EF v3.0', 'photochemical ozone formation: human health', 'tropospheric ozone concentration increase'),
    'PMAT'  : ('EF v3.0', 'particulate matter formation', 'impact on human health'),

    # Toxicity
    'HTOXnc': ('EF v3.0', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh) '),  # trailing space
    'HTOXc' : ('EF v3.0', 'human toxicity: carcinogenic',    'comparative toxic unit for human (CTUh) '),  # trailing space

    # Eutrophication
    'FWEUT' : ('EF v3.0', 'eutrophication: freshwater',  'fraction of nutrients reaching freshwater end compartment (P)'),
    'MWEUT' : ('EF v3.0', 'eutrophication: marine',      'fraction of nutrients reaching marine end compartment (N)'),
    'TEUT'  : ('EF v3.0', 'eutrophication: terrestrial', 'accumulated exceedance (AE) '),                  # trailing space

    # Ecotox, land, water, resources
    'FWTOX' : ('EF v3.0', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe) '),     # trailing space
    'LUP'   : ('EF v3.0', 'land use', 'soil quality index'),
    'WDEPL' : ('EF v3.0', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'),
    'ADEPLf': ('EF v3.0', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels'),
    'ADEPLmu':('EF v3.0', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)'),
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
# ECOINVENT MATERIAL CODES
# ========================================

ECOINVENT_CODES = {
    # Structural materials
    'aluminum_7075': ('ecoinvent 3.8 cutoff','8392648c098b86d088a9821ce11ed9dd'),  # Aluminum for structural parts
    'aluminum_lithium': ('ecoinvent 3.8 cutoff','03f6b6ba551e8541bf47842791abd3f7'),  # Aluminum-lithium alloy for tanks
    'cfrp': ('ecoinvent 3.8 cutoff','5f83b772ba1476f12d0b3ef634d4409b'),  # Carbon fiber reinforced plastic
    'steel': ('ecoinvent 3.8 cutoff','9b20aabdab5590c519bb3d717c77acf2'),  # High-strength steel
    'titanium': ('ecoinvent 3.8 cutoff','3412f692460ecd5ce8dcfcd5adb1c072'),  # Titanium alloy

    # Insulation and other materials
    'polyurethane_foam': ('ecoinvent 3.8 cutoff','223d2ca85f5c350a6a043725a2b71226'),  # Thermal insulation
    'electronics': ('ecoinvent 3.8 cutoff','52c4f6d2e1ec507b1ccc96056a761c0d'),  # Electronics/avionics

    # Propellants
    'lox': ('ecoinvent 3.8 cutoff','53b5def592497847e2d0b4d62f2c4456'),  # Liquid oxygen
    'lh2': ('ecoinvent 3.8 cutoff','a834063e527dafabe7d179a804a13f39'),  # Liquid hydrogen

    # Transport and energy
    'transport_ship': ('ecoinvent 3.8 cutoff','41205d7711c0fad4403e4c2f9284b083'),  # Ship transport
    'transport_truck': ('ecoinvent 3.8 cutoff','7e3b4d9c1a6f2805b9d7c3e1f4a62938'),  # Truck transport
    'electricity': ('ecoinvent 3.8 cutoff','3855bf674145307cd56a3fac8c83b643'),  # Electricity, medium voltage
}

# ========================================
# MATERIAL MAPPING FOR COMPONENTS
# ========================================

COMPONENT_MATERIALS = {
    # Variable material components (affected by k_SM)
    'thrust_frame': 'variable',  # Al or Composite based on k_SM
    'interstage': 'variable',    # Al or Composite based on k_SM
    'intertank': 'variable',     # Al or Composite based on k_SM
    
    # Fixed material components
    'tanks': 'aluminum_tank',    # Always aluminum-lithium alloy
    'engines': 'mixed_engine',   # 70% aluminum, 20% steel, 10% titanium
    'tps': 'polyurethane_foam',  # Thermal protection
    'tvc': 'mixed_tvc',          # 60% aluminum, 40% steel
    'avionics': 'electronics',   # Electronics
    'eps': 'electronics',        # Electrical power system
}

class Environmental_Discipline_Comp(ExplicitComponent):
    """
    Environmental/LCA discipline for launcher optimization
    Uses real ecoinvent data with Brightway2, calculates ESA impact categories
    """
    
    def setup(self):
        # ========================================
        # INPUTS FROM STRUCTURAL DISCIPLINE
        # ========================================
        
        # Individual component masses from Stage 1
        self.add_input('M_eng_stage_1', val=100.)
        self.add_input('M_thrust_frame_stage_1', val=100.)
        self.add_input('M_FT_stage_1', val=100.)
        self.add_input('M_OxT_stage_1', val=100.)
        self.add_input('M_TPS_OxT_stage_1', val=100.)
        self.add_input('M_TPS_FT_stage_1', val=100.)
        self.add_input('M_TVC_stage_1', val=100.)
        self.add_input('M_avio_stage_1', val=100.)
        self.add_input('M_EPS_stage_1', val=100.)
        self.add_input('M_intertank_stage_1', val=100.)
        self.add_input('M_interstage_stage_1', val=100.)
        
        # Material fractions for variable components
        self.add_input('Al_fraction_thrust_frame_stage_1', val=1.0)
        self.add_input('Composite_fraction_thrust_frame_stage_1', val=0.0)
        self.add_input('Al_fraction_interstage_stage_1', val=1.0)
        self.add_input('Composite_fraction_interstage_stage_1', val=0.0)
        self.add_input('Al_fraction_intertank_stage_1', val=1.0)
        self.add_input('Composite_fraction_intertank_stage_1', val=0.0)
        
        # Stage 2 mass (simplified)
        self.add_input('Dry_mass_stage_2', val=3000.)
        
        # Propellant masses
        self.add_input('Prop_mass_stage_1', val=250000.)
        self.add_input('Prop_mass_stage_2', val=50000.)
        self.add_input('OF_stage_1', val=5.5)
        self.add_input('OF_stage_2', val=5.5)
        
        # ========================================
        # OUTPUTS - ESA IMPACT CATEGORIES
        # ========================================
        
        # Individual impact category scores
        for code in ESA_METHODS.keys():
            self.add_output(f'{code}_impact', val=0.0)
            self.add_output(f'{code}_normalized', val=0.0)
            self.add_output(f'{code}_weighted', val=0.0)
        
        # Single scores
        self.add_output('ESA_single_score', val=100.0)
        self.add_output('LCA_score', val=1000.0)  # For compatibility
        
        # Material mass tracking (for analysis)
        self.add_output('total_aluminum_7075_kg', val=0.0)
        self.add_output('total_aluminum_lithium_kg', val=0.0)
        self.add_output('total_composite_kg', val=0.0)
        self.add_output('total_steel_kg', val=0.0)
        self.add_output('total_other_kg', val=0.0)
        
        # CO2 and energy for simplified reporting
        self.add_output('CO2_eq', val=0.0)
        self.add_output('Energy_consumption', val=0.0)
        
        # Stage-wise impacts for analysis
        self.add_output('LCA_stage_1', val=500.)
        self.add_output('LCA_stage_2', val=500.)
        self.add_output('LCA_propellants', val=100.)
    
    def compute(self, inputs, outputs):
        """
        Calculate environmental impacts based on component masses and materials
        """
        
        # ========================================
        # CALCULATE MATERIAL MASSES
        # ========================================
        
        # Initialize material accumulators
        aluminum_7075 = 0.0
        aluminum_lithium = 0.0
        composite_total = 0.0
        steel_total = 0.0
        titanium_total = 0.0
        polyurethane_total = 0.0
        electronics_total = 0.0
        
        # Stage 1 components with variable materials
        # Thrust frame (variable Al/Composite)
        aluminum_7075 += inputs['M_thrust_frame_stage_1'][0] * inputs['Al_fraction_thrust_frame_stage_1'][0]
        composite_total += inputs['M_thrust_frame_stage_1'][0] * inputs['Composite_fraction_thrust_frame_stage_1'][0]
        
        # Interstage (variable Al/Composite)
        aluminum_7075 += inputs['M_interstage_stage_1'][0] * inputs['Al_fraction_interstage_stage_1'][0]
        composite_total += inputs['M_interstage_stage_1'][0] * inputs['Composite_fraction_interstage_stage_1'][0]

        # Intertank (variable Al/Composite)
        aluminum_7075 += inputs['M_intertank_stage_1'][0] * inputs['Al_fraction_intertank_stage_1'][0]
        composite_total += inputs['M_intertank_stage_1'][0] * inputs['Composite_fraction_intertank_stage_1'][0]
        
        # Fixed material components
        # Tanks (aluminum-lithium alloy)
        aluminum_lithium += inputs['M_FT_stage_1'][0] + inputs['M_OxT_stage_1'][0]
        
        # Engines (70% Al, 20% steel, 10% titanium)
        engine_mass = inputs['M_eng_stage_1'][0]
        aluminum_lithium += engine_mass * 0.7
        steel_total += engine_mass * 0.2
        titanium_total += engine_mass * 0.1
        
        # TVC (60% Al, 40% steel)
        tvc_mass = inputs['M_TVC_stage_1'][0]
        aluminum_7075 += tvc_mass * 0.6
        steel_total += tvc_mass * 0.4
        
        # Thermal protection (polyurethane foam)
        polyurethane_total += inputs['M_TPS_OxT_stage_1'][0] + inputs['M_TPS_FT_stage_1'][0]
        
        # Electronics (avionics + EPS)
        electronics_total += inputs['M_avio_stage_1'][0] + inputs['M_EPS_stage_1'][0]
        
        # Stage 2 (simplified breakdown)
        # Assume: 50% Al, 20% Composite, 15% steel, 5% titanium, 10% other
        dry_mass_s2 = inputs['Dry_mass_stage_2'][0]
        aluminum_7075 += dry_mass_s2 * 0.3
        aluminum_lithium += dry_mass_s2 * 0.2
        composite_total += dry_mass_s2 * 0.2
        steel_total += dry_mass_s2 * 0.15
        titanium_total += dry_mass_s2 * 0.05
        electronics_total += dry_mass_s2 * 0.05
        polyurethane_total += dry_mass_s2 * 0.05
        
        # Store material totals
        outputs['total_aluminum_7075_kg'] = aluminum_7075
        outputs['total_aluminum_lithium_kg'] = aluminum_lithium
        outputs['total_composite_kg'] = composite_total
        outputs['total_steel_kg'] = steel_total
        outputs['total_other_kg'] = titanium_total + polyurethane_total + electronics_total

        total_dry_mass = (aluminum_7075 + aluminum_lithium + composite_total + steel_total +
                  titanium_total + polyurethane_total + electronics_total)

        
        # ========================================
        # CALCULATE PROPELLANT MASSES
        # ========================================
        
        prop_s1 = inputs['Prop_mass_stage_1'][0]
        prop_s2 = inputs['Prop_mass_stage_2'][0]
        of_s1 = inputs['OF_stage_1'][0]
        of_s2 = inputs['OF_stage_2'][0]
        
        lox_total = (prop_s1 * (of_s1 / (1 + of_s1)) + 
                     prop_s2 * (of_s2 / (1 + of_s2)))
        lh2_total = (prop_s1 * (1 / (1 + of_s1)) + 
                     prop_s2 * (1 / (1 + of_s2)))
        
        print(f"LOX mass: {lox_total/1000:.1f} tonnes")
        print(f"LH2 mass: {lh2_total/1000:.1f} tonnes")

        # In Environmental_Discipline.py compute() method:
        print("\n=== LCA DEBUG ===")
        print(f"Structural mass: {total_dry_mass:.1f} kg")
        print(f"LOX mass: {lox_total:.1f} kg") 
        print(f"LH2 mass: {lh2_total:.1f} kg")
        
        # ========================================
        # CREATE INVENTORY FOR LCA
        # ========================================
        
        # ---- Build inventory (activity -> amount) ----
        if not LCA_ENABLED:
            self._use_placeholder_impacts(outputs); return

        def _act(ptr):
            # ptr is ('db','code') from ECOINVENT_CODES
            return bw.get_activity(ptr)

        amounts = {
            'aluminum_7075':      float(aluminum_7075),
            'aluminum_lithium':   float(aluminum_lithium),
            'cfrp':               float(composite_total),
            'steel':              float(steel_total),
            'titanium':           float(titanium_total),
            'polyurethane_foam':  float(polyurethane_total),
            'electronics':        float(electronics_total),
            'lox':                float(lox_total),
            'lh2':                float(lh2_total),
            'transport_ship':     float((total_dry_mass / 1000.0) * 7000.0),
            'electricity':        float(total_dry_mass * 0.2),
        }

        inventory = {}
        for key, amt in amounts.items():
            if amt > 0:
                try:
                    inventory[_act(ECOINVENT_CODES[key])] = amt
                except Exception as e:
                    # Skip unresolved items but keep the MDO loop running
                    print(f"[Environmental] WARN: could not resolve '{key}': {e}")
        
        # ========================================
        # CALCULATE IMPACTS FOR EACH ESA CATEGORY
        # ========================================

        esa_single_score = 0.0
        for code, method in ESA_METHODS.items():
            try:
                # Calculate impact for this category
                lca = bw.LCA(inventory, method)
                lca.lci()
                lca.lcia()
                impact_value = lca.score

                # Store raw impact
                outputs[f'{code}_impact'] = impact_value

                # Normalize
                normalized = impact_value * ESA_NORMALIZATION[code]
                outputs[f'{code}_normalized'] = normalized

                # Weight
                weighted = normalized * ESA_WEIGHTS[code]
                outputs[f'{code}_weighted'] = weighted

                # Add to single score
                esa_single_score += weighted

            except Exception as e:
                print(f"Warning: Could not calculate {code}: {e}")
                outputs[f'{code}_impact'] = 0.0
                outputs[f'{code}_normalized'] = 0.0
                outputs[f'{code}_weighted'] = 0.0

        # Set single scores
        outputs['ESA_single_score'] = esa_single_score
        outputs['LCA_score'] = esa_single_score   # Scale for compatibility

        # Extract key impacts for simple reporting
        outputs['CO2_eq'] = outputs['GWP_impact'][0] if 'GWP_impact' in outputs else 0.0
        outputs['Energy_consumption'] = outputs['ADEPLf_impact'][0] if 'ADEPLf_impact' in outputs else 0.0

        # Stage-wise breakdown (simplified)
        # Allocate impacts proportionally to mass
        total_mass = total_dry_mass + lox_total + lh2_total
        stage1_fraction = (aluminum_7075 * 0.7 + composite_total * 0.8) / total_mass
        stage2_fraction = dry_mass_s2 / total_mass
        propellant_fraction = (lox_total + lh2_total) / total_mass
        outputs['LCA_stage_1'] = outputs['LCA_score'][0] * stage1_fraction
        outputs['LCA_stage_2'] = outputs['LCA_score'][0] * stage2_fraction
        outputs['LCA_propellants'] = outputs['LCA_score'][0] * propellant_fraction

    def _use_placeholder_impacts(self, outputs):
        """ Use placeholder impact values when ecoinvent is not available
        Based on typical aerospace LCA values from literature
        """
        # Simplified impact factors (kg CO2-eq per kg material)
        impact_factors = {
            'aluminum': 8.5,
            'composite': 25.0,
            'steel': 2.3,
            'titanium': 35.0,
            'electronics': 50.0,
            'polyurethane': 3.5,
            'lh2': 9.0,
            'lox': 0.2,
        }

        # Calculate simplified GWP
        gwp = (
            outputs['total_aluminum_kg'][0] * impact_factors['aluminum'] +
            outputs['total_composite_kg'][0] * impact_factors['composite'] +
            outputs['total_steel_kg'][0] * impact_factors['steel'] +
            outputs['total_other_kg'][0] * 15.0  # Average for other materials
        )

        # Set placeholder values for all impact categories
        # Use GWP as proxy with scaling factors
        scaling = {
            'GWP': 1.0,
            'ODEPL': 0.0001,
            'IORAD': 0.01,
            'PCHEM': 0.1,
            'PMAT': 0.05,
            'HTOXnc': 0.02,
            'HTOXc': 0.01,
            'ACIDef': 0.08,
            'FWEUT': 0.03,
            'MWEUT': 0.03,
            'TEUT': 0.04,
            'FWTOX': 0.02,
            'LUP': 0.05,
            'WDEPL': 0.06,
            'ADEPLf': 0.15,
            'ADEPLmu': 0.10,
        }

        esa_single_score = 0.0
        for code in ESA_METHODS.keys():
            # Raw impact (scaled from GWP)
            impact = gwp * scaling[code]
            outputs[f'{code}_impact'] = impact

            # Normalize
            normalized = impact * ESA_NORMALIZATION[code]
            outputs[f'{code}_normalized'] = normalized

            # Weight
            weighted = normalized * ESA_WEIGHTS[code]
            outputs[f'{code}_weighted'] = weighted

            # Add to single score
            esa_single_score += weighted

        # Set output scores
        outputs['ESA_single_score'] = esa_single_score
        outputs['CO2_eq'] = gwp
        outputs['Energy_consumption'] = gwp * 20  # Rough estimate

        # Stage breakdown
        outputs['LCA_stage_1'] = outputs['LCA_score'][0] * 0.6
        outputs['LCA_stage_2'] = outputs['LCA_score'][0] * 0.2
        outputs['LCA_propellants'] = outputs['LCA_score'][0] * 0.2