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
    'electronics': ('ecoinvent 3.8 cutoff','b1b65fe4d00b29f2299c72b894a3c0a0'),  # Electronics/avionics

    # Propellants
    'lox': ('ecoinvent 3.8 cutoff','53b5def592497847e2d0b4d62f2c4456'),  # Liquid oxygen
    'lh2': ('ecoinvent 3.8 cutoff','a834063e527dafabe7d179a804a13f39'),  # Liquid hydrogen

    # Transport and energy
    'transport_ship': ('ecoinvent 3.8 cutoff','41205d7711c0fad4403e4c2f9284b083'),  # Ship transport
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

        def get_scalar(val):
            """Safely extract scalar from array or scalar input"""
            if hasattr(val, '__len__'):
                return float(val[0]) if len(val) > 0 else 0.0
            return float(val)
        
        # Initialize material accumulators
        aluminum_7075 = 0.0
        aluminum_lithium = 0.0
        composite_total = 0.0
        steel_total = 0.0
        titanium_total = 0.0
        polyurethane_total = 0.0
        electronics_total = 0.0

        # Stage 1 components with variable materials
        # Use get_scalar for all inputs to handle arrays properly
        m_thrust_frame = get_scalar(inputs['M_thrust_frame_stage_1'])
        al_frac_tf = get_scalar(inputs['Al_fraction_thrust_frame_stage_1'])
        comp_frac_tf = get_scalar(inputs['Composite_fraction_thrust_frame_stage_1'])
        
        aluminum_7075 += m_thrust_frame * al_frac_tf
        composite_total += m_thrust_frame * comp_frac_tf
        
        # Continue with other components...
        m_interstage = get_scalar(inputs['M_interstage_stage_1'])
        al_frac_is = get_scalar(inputs['Al_fraction_interstage_stage_1'])
        comp_frac_is = get_scalar(inputs['Composite_fraction_interstage_stage_1'])
        
        aluminum_7075 += m_interstage * al_frac_is
        composite_total += m_interstage * comp_frac_is

        m_intertank = get_scalar(inputs['M_intertank_stage_1'])
        al_frac_it = get_scalar(inputs['Al_fraction_intertank_stage_1'])
        comp_frac_it = get_scalar(inputs['Composite_fraction_intertank_stage_1'])

        aluminum_7075 += m_intertank * al_frac_it
        composite_total += m_intertank * comp_frac_it

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

        missing_codes = []
        for key, code_tuple in ECOINVENT_CODES.items():
            try:
                act = bw.get_activity(code_tuple)
            except:
                missing_codes.append(key)
                print(f"WARNING: Cannot find ecoinvent activity for {key}")
        
        if missing_codes:
            print(f"Missing {len(missing_codes)} ecoinvent activities, using placeholders")
            self._use_placeholder_impacts(outputs)
            return
        
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
    
        # ========================================
        # DETAILED IMPACT BREAKDOWN DEBUGGING
        # ========================================
        
        print("\n" + "="*80)
        print("DETAILED LCA IMPACT BREAKDOWN")
        print("="*80)
        
        # First, let's see the material quantities
        print("\n--- MATERIAL QUANTITIES ---")
        materials = {
            'aluminum_7075': aluminum_7075,
            'aluminum_lithium': aluminum_lithium,
            'cfrp': composite_total,
            'steel': steel_total,
            'titanium': titanium_total,
            'polyurethane_foam': polyurethane_total,
            'electronics': electronics_total,
            'lox': lox_total,
            'lh2': lh2_total,
        }
        
        for mat, qty in materials.items():
            if qty > 0:
                print(f"{mat:20s}: {qty:10.1f} kg ({qty/1000:7.2f} tonnes)")
        
        print(f"\nTransport (tkm):      {(total_dry_mass / 1000.0) * 7000.0:10.1f} tkm")
        print(f"Electricity:          {total_dry_mass * 0.2:10.1f} kWh")
        
        # ========================================
        # INDIVIDUAL MATERIAL IMPACT ANALYSIS
        # ========================================
        
        print("\n" + "-"*80)
        print("IMPACT PER MATERIAL (GWP and ADEPLmu)")
        print("-"*80)
        
        def _act(ptr):
            return bw.get_activity(ptr)
        
        # Test each material individually
        material_impacts = {}
        
        for mat_name, qty in materials.items():
            if qty > 0 and mat_name in ECOINVENT_CODES:
                try:
                    # Create inventory with just this material
                    single_mat_inventory = {_act(ECOINVENT_CODES[mat_name]): qty}
                    
                    # Calculate GWP
                    lca_gwp = bw.LCA(single_mat_inventory, ESA_METHODS['GWP'])
                    lca_gwp.lci()
                    lca_gwp.lcia()
                    gwp_impact = lca_gwp.score
                    
                    # Calculate ADEPLmu (mineral depletion)
                    lca_adeplmu = bw.LCA(single_mat_inventory, ESA_METHODS['ADEPLmu'])
                    lca_adeplmu.lci()
                    lca_adeplmu.lcia()
                    adeplmu_impact = lca_adeplmu.score
                    
                    # Calculate FWTOX (freshwater ecotoxicity)
                    lca_fwtox = bw.LCA(single_mat_inventory, ESA_METHODS['FWTOX'])
                    lca_fwtox.lci()
                    lca_fwtox.lcia()
                    fwtox_impact = lca_fwtox.score
                    
                    material_impacts[mat_name] = {
                        'quantity': qty,
                        'GWP': gwp_impact,
                        'ADEPLmu': adeplmu_impact,
                        'FWTOX': fwtox_impact,
                        'GWP_per_kg': gwp_impact / qty,
                        'ADEPLmu_per_kg': adeplmu_impact / qty,
                        'FWTOX_per_kg': fwtox_impact / qty,
                    }
                    
                    print(f"\n{mat_name}:")
                    print(f"  Quantity:        {qty:10.1f} kg")
                    print(f"  GWP total:       {gwp_impact:10.1f} kg CO2-eq")
                    print(f"  GWP per kg:      {gwp_impact/qty:10.4f} kg CO2-eq/kg")
                    print(f"  ADEPLmu total:   {adeplmu_impact:10.4f} kg Sb-eq")
                    print(f"  ADEPLmu per kg:  {adeplmu_impact/qty:10.6f} kg Sb-eq/kg")
                    print(f"  FWTOX total:     {fwtox_impact:10.2e} CTUe")
                    print(f"  FWTOX per kg:    {fwtox_impact/qty:10.2e} CTUe/kg")
                    
                except Exception as e:
                    print(f"\n{mat_name}: ERROR - {e}")
        
        # ========================================
        # RANKING BY CONTRIBUTION
        # ========================================
        
        print("\n" + "-"*80)
        print("MATERIALS RANKED BY GWP CONTRIBUTION")
        print("-"*80)
        
        # Sort by GWP contribution
        gwp_ranking = sorted(material_impacts.items(), 
                            key=lambda x: x[1]['GWP'], 
                            reverse=True)
        
        total_gwp = sum(m['GWP'] for m in material_impacts.values())
        
        for mat, impacts in gwp_ranking:
            pct = 100 * impacts['GWP'] / total_gwp if total_gwp > 0 else 0
            print(f"{mat:20s}: {impacts['GWP']:12.1f} kg CO2-eq ({pct:5.1f}%)")
        
        print("\n" + "-"*80)
        print("MATERIALS RANKED BY ADEPLmu CONTRIBUTION")
        print("-"*80)
        
        # Sort by ADEPLmu contribution
        adeplmu_ranking = sorted(material_impacts.items(), 
                                key=lambda x: x[1]['ADEPLmu'], 
                                reverse=True)
        
        total_adeplmu = sum(m['ADEPLmu'] for m in material_impacts.values())
        
        for mat, impacts in adeplmu_ranking:
            pct = 100 * impacts['ADEPLmu'] / total_adeplmu if total_adeplmu > 0 else 0
            print(f"{mat:20s}: {impacts['ADEPLmu']:12.4f} kg Sb-eq ({pct:5.1f}%)")
        
        # ========================================
        # INVESTIGATE SPECIFIC ACTIVITIES
        # ========================================
        
        print("\n" + "-"*80)
        print("ECOINVENT ACTIVITY DETAILS")
        print("-"*80)
        
        # Check what's actually in the hydrogen and oxygen activities
        for key in ['lox', 'lh2']:
            try:
                act = _act(ECOINVENT_CODES[key])
                print(f"\n{key.upper()} Activity: {act['name']}")
                print(f"  Location: {act.get('location', 'N/A')}")
                print(f"  Unit: {act.get('unit', 'N/A')}")
                
                # Look at the top exchanges (inputs to this activity)
                print(f"  Top exchanges (inputs):")
                exchanges = list(act.exchanges())[:5]  # First 5 exchanges
                for exc in exchanges:
                    if exc['type'] == 'technosphere':
                        print(f"    - {exc.input['name']}: {exc['amount']} {exc.get('unit', '')}")
                        
            except Exception as e:
                print(f"\n{key.upper()}: Could not analyze - {e}")
        
        # ========================================
        # BREAKDOWN BY LIFECYCLE PHASE
        # ========================================
        
        print("\n" + "-"*80)
        print("LIFECYCLE PHASE BREAKDOWN")
        print("-"*80)
        
        # Structural materials (manufacturing phase)
        structural_mats = ['aluminum_7075', 'aluminum_lithium', 'cfrp', 'steel', 
                        'titanium', 'polyurethane_foam', 'electronics']
        
        structural_gwp = sum(material_impacts.get(m, {}).get('GWP', 0) 
                            for m in structural_mats)
        structural_adeplmu = sum(material_impacts.get(m, {}).get('ADEPLmu', 0) 
                                for m in structural_mats)
        
        # Propellants (use phase)
        propellant_gwp = (material_impacts.get('lox', {}).get('GWP', 0) + 
                        material_impacts.get('lh2', {}).get('GWP', 0))
        propellant_adeplmu = (material_impacts.get('lox', {}).get('ADEPLmu', 0) + 
                            material_impacts.get('lh2', {}).get('ADEPLmu', 0))
        
        print(f"\nStructural Materials:")
        print(f"  GWP:     {structural_gwp:12.1f} kg CO2-eq ({100*structural_gwp/total_gwp:5.1f}%)")
        print(f"  ADEPLmu: {structural_adeplmu:12.4f} kg Sb-eq ({100*structural_adeplmu/total_adeplmu:5.1f}%)")
        
        print(f"\nPropellants:")
        print(f"  GWP:     {propellant_gwp:12.1f} kg CO2-eq ({100*propellant_gwp/total_gwp:5.1f}%)")
        print(f"  ADEPLmu: {propellant_adeplmu:12.4f} kg Sb-eq ({100*propellant_adeplmu/total_adeplmu:5.1f}%)")
        
        # ========================================
        # NORMALIZATION CHECK
        # ========================================
        
        print("\n" + "-"*80)
        print("NORMALIZATION AND WEIGHTING CHECK")
        print("-"*80)
        
        print("\nADEPLmu calculation chain:")
        print(f"  Raw impact:     {total_adeplmu:.4f} kg Sb-eq")
        print(f"  Normalization:  {total_adeplmu:.4f} × {ESA_NORMALIZATION['ADEPLmu']} = {total_adeplmu * ESA_NORMALIZATION['ADEPLmu']:.2f}")
        print(f"  Weighting:      {total_adeplmu * ESA_NORMALIZATION['ADEPLmu']:.2f} × {ESA_WEIGHTS['ADEPLmu']} = {total_adeplmu * ESA_NORMALIZATION['ADEPLmu'] * ESA_WEIGHTS['ADEPLmu']:.2f} Pt")
        
        print("\nGWP calculation chain:")
        print(f"  Raw impact:     {total_gwp:.1f} kg CO2-eq")
        print(f"  Normalization:  {total_gwp:.1f} × {ESA_NORMALIZATION['GWP']} = {total_gwp * ESA_NORMALIZATION['GWP']:.2f}")
        print(f"  Weighting:      {total_gwp * ESA_NORMALIZATION['GWP']:.2f} × {ESA_WEIGHTS['GWP']} = {total_gwp * ESA_NORMALIZATION['GWP'] * ESA_WEIGHTS['GWP']:.2f} Pt")
        
        # ========================================
        # SUSPICIOUS VALUES CHECK
        # ========================================
        
        print("\n" + "-"*80)
        print("SUSPICIOUS VALUES CHECK")
        print("-"*80)
        
        # Check for unusually high per-kg impacts
        print("\nMaterials with potentially incorrect impacts:")
        
        for mat, impacts in material_impacts.items():
            gwp_per_kg = impacts['GWP_per_kg']
            adeplmu_per_kg = impacts['ADEPLmu_per_kg']
            
            # Expected ranges (approximate)
            expected_ranges = {
                'aluminum_7075': (5, 15, 0.0001, 0.001),      # GWP_min, GWP_max, ADP_min, ADP_max
                'aluminum_lithium': (8, 20, 0.0001, 0.002),
                'cfrp': (15, 35, 0.001, 0.01),
                'steel': (1, 5, 0.00005, 0.001),
                'titanium': (20, 50, 0.001, 0.01),
                'lox': (0.1, 0.5, 0.00001, 0.0001),
                'lh2': (5, 20, 0.0001, 0.001),  # Green H2 would be 5-20, grey H2 could be 50+
            }
            
            if mat in expected_ranges:
                gwp_min, gwp_max, adp_min, adp_max = expected_ranges[mat]
                
                if gwp_per_kg < gwp_min or gwp_per_kg > gwp_max:
                    print(f"\n⚠ {mat}: GWP = {gwp_per_kg:.2f} kg CO2/kg (expected {gwp_min}-{gwp_max})")
                
                if adeplmu_per_kg < adp_min or adeplmu_per_kg > adp_max:
                    print(f"⚠ {mat}: ADEPLmu = {adeplmu_per_kg:.6f} kg Sb-eq/kg (expected {adp_min}-{adp_max})")
        
        print("\n" + "="*80)
        
        # Return the breakdown for further analysis
        return material_impacts

    # Additional function to search for alternative activities
    def search_alternative_activities():
        """Search ecoinvent for alternative oxygen and hydrogen activities"""
        import brightway2 as bw
        db = bw.Database("ecoinvent 3.8 cutoff")
        
        print("\n=== OXYGEN ACTIVITIES ===")
        for act in db.search("oxygen", limit=20):
            if "liquid" in act['name'].lower():
                print(f"{act['name'][:80]} - {act['code']}")
                # Test impact
                try:
                    inventory = {act: 1.0}  # 1 kg
                    lca = bw.LCA(inventory, ('EF v3.0', 'climate change', 'global warming potential (GWP100)'))
                    lca.lci()
                    lca.lcia()
                    print(f"  → GWP: {lca.score:.4f} kg CO2-eq/kg")
                except:
                    pass
        
        print("\n=== HYDROGEN ACTIVITIES ===")
        for act in db.search("hydrogen", limit=20):
            print(f"{act['name'][:80]} - {act['code']}")
            # Test impact
            try:
                inventory = {act: 1.0}  # 1 kg
                lca = bw.LCA(inventory, ('EF v3.0', 'climate change', 'global warming potential (GWP100)'))
                lca.lci()
                lca.lcia()
                print(f"  → GWP: {lca.score:.4f} kg CO2-eq/kg")
            except:
                pass

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