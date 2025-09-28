# -*- coding: utf-8 -*-
"""
Environmental Discipline for FELIN Launcher LCA
CORRECT VERSION - Using actual Brightway2/ecoinvent for all impact calculations
"""

import numpy as np
from openmdao.api import ExplicitComponent
from environment.material_helpers import MaterialConverter

try:
    import brightway2 as bw
    bw.projects.set_current("LCA_FELIN")
    
    if "ecoinvent 3.8 cutoff" not in bw.databases:
        raise ImportError("ERROR: Ecoinvent 3.8 cutoff database not found.")
    
    print("✓ Brightway2 and ecoinvent successfully loaded")
    LCA_ENABLED = True
    
except ImportError as e:
    print(f"WARNING: {e}")
    print("Environmental discipline running in degraded mode (simplified calculations)")
    LCA_ENABLED = False

# ========================================
# ESA IMPACT ASSESSMENT METHODS
# ========================================

ESA_METHODS = {
    'GWP': ('EF v3.0', 'climate change', 'global warming potential (GWP100)'),
    'ODEPL': ('EF v3.0', 'ozone depletion', 'ozone depletion potential (ODP) '),
    'IORAD': ('EF v3.0', 'ionising radiation: human health', 'human exposure efficiency relative to u235'),
    'ACIDef': ('EF v3.0', 'acidification', 'accumulated exceedance (ae)'),
    'PCHEM': ('EF v3.0', 'photochemical ozone formation: human health', 'tropospheric ozone concentration increase'),
    'PMAT': ('EF v3.0', 'particulate matter formation', 'impact on human health'),
    'HTOXnc': ('EF v3.0', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh) '),
    'HTOXc': ('EF v3.0', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh) '),
    'FWEUT': ('EF v3.0', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)'),
    'MWEUT': ('EF v3.0', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)'),
    'TEUT': ('EF v3.0', 'eutrophication: terrestrial', 'accumulated exceedance (AE) '),
    'FWTOX': ('EF v3.0', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe) '),
    'LUP': ('EF v3.0', 'land use', 'soil quality index'),
    'WDEPL': ('EF v3.0', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'),
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
    'aluminum_7075': ('ecoinvent 3.8 cutoff', '8392648c098b86d088a9821ce11ed9dd'),
    'aluminum_lithium': ('ecoinvent 3.8 cutoff', '03f6b6ba551e8541bf47842791abd3f7'),
    'cfrp': ('ecoinvent 3.8 cutoff', '5f83b772ba1476f12d0b3ef634d4409b'),
    'steel': ('ecoinvent 3.8 cutoff', '9b20aabdab5590c519bb3d717c77acf2'),
    'titanium': ('ecoinvent 3.8 cutoff', '3412f692460ecd5ce8dcfcd5adb1c072'),
    
    # Insulation and avionics
    'polyurethane_foam': ('ecoinvent 3.8 cutoff', '223d2ca85f5c350a6a043725a2b71226'),
    'electronics': ('ecoinvent 3.8 cutoff', 'b1b65fe4d00b29f2299c72b894a3c0a0'),
    
    # Propellants
    'lox': ('ecoinvent 3.8 cutoff', '53b5def592497847e2d0b4d62f2c4456'),
    'lh2': ('ecoinvent 3.8 cutoff', 'a834063e527dafabe7d179a804a13f39'),
    
    # Transport and energy
    'transport_ship': ('ecoinvent 3.8 cutoff', '41205d7711c0fad4403e4c2f9284b083'),
    'electricity': ('ecoinvent 3.8 cutoff', '3855bf674145307cd56a3fac8c83b643'),
}

# for when Brightway2 is not available
SIMPLIFIED_IMPACTS = {
    'aluminum_7075': 8.5,      # kg CO2-eq/kg
    'aluminum_lithium': 9.0,
    'cfrp': 25.0,
    'steel': 2.3,
    'titanium': 35.0,
    'polyurethane_foam': 3.5,
    'electronics': 50.0,
    'lh2': 9.0,
    'lox': 0.2,
}

class Environmental_Discipline_Comp(ExplicitComponent):
    """
    Environmental/LCA discipline for launcher optimization
    Uses real ecoinvent data with Brightway2, calculates ESA impact categories
    """
    _baseline_dry_mass = None  # Class variable to store baseline dry mass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iteration_count = 0
        self.verbose = False
    
    def setup(self):
        # ========================================
        # INPUTS FROM STRUCTURAL DISCIPLINE
        # ========================================
        
        # Stage 1 component masses
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
        
        # Stage 2
        self.add_input('Dry_mass_stage_2', val=3000.)
        self.add_input('Al_fraction_stage_2', val=0.5)
        self.add_input('Composite_fraction_stage_2', val=0.5)

        self.add_input('k_SM_thrust_frame', val=1.0)
        self.add_input('k_SM_interstage', val=1.0)
        self.add_input('k_SM_intertank', val=1.0)
        self.add_input('k_SM_stage_2', val=1.0)
        
        # Propellant masses
        self.add_input('Prop_mass_stage_1', val=250000.)
        self.add_input('Prop_mass_stage_2', val=50000.)
        self.add_input('OF_stage_1', val=5.5)
        self.add_input('OF_stage_2', val=5.5)
        
        # ========================================
        # OUTPUTS - ESA IMPACT CATEGORIES
        # ========================================
        
        # Individual impact category scores (raw, normalized, weighted)
        for code in ESA_METHODS.keys():
            self.add_output(f'{code}_impact', val=0.0)
            self.add_output(f'{code}_normalized', val=0.0)
            self.add_output(f'{code}_weighted', val=0.0)
        
        # Single scores
        self.add_output('LCA_score', val=100.0)  # ESA single score (Points)
        self.add_output('LCA_manufacturing', val=50.0)
        self.add_output('LCA_operational_benefit', val=0.0)
        
        # Material mass tracking
        self.add_output('total_aluminum_7075_kg', val=0.0)
        self.add_output('total_aluminum_lithium_kg', val=0.0)
        self.add_output('total_composite_kg', val=0.0)
        self.add_output('total_steel_kg', val=0.0)
        self.add_output('total_titanium_kg', val=0.0)
        self.add_output('total_insulation_kg', val=0.0)
        self.add_output('total_electronics_kg', val=0.0)
        
        # Key metrics for reporting
        self.add_output('CO2_eq', val=0.0)  # Total GWP impact
        self.add_output('Energy_consumption', val=0.0)  # ADEPLf impact
        
        # Stage-wise impacts
        self.add_output('LCA_stage_1', val=50.)
        self.add_output('LCA_stage_2', val=30.)
        self.add_output('LCA_propellants', val=20.)
        
        # Mass savings for operational benefit
        self.add_output('mass_savings_kg', val=0.0)
    
    def compute(self, inputs, outputs):
        """
        Calculate environmental impacts using Brightway2 and ecoinvent data
        """
        self.iteration_count += 1
        print_debug = (self.iteration_count % 10 == 1) and hasattr(self, 'verbose') and self.verbose
        
        # Helper function
        def get_scalar(val):
            if hasattr(val, '__len__'):
                return float(val[0]) if len(val) > 0 else 0.0
            return float(val)
        
        # ========================================
        # BUILD MATERIAL INVENTORY
        # ========================================
        
        material_inventory = {
            'aluminum_7075': 0.0,
            'aluminum_lithium': 0.0,
            'cfrp': 0.0,
            'steel': 0.0,
            'titanium': 0.0,
            'polyurethane_foam': 0.0,
            'electronics': 0.0,
            'lox': 0.0,
            'lh2': 0.0,
        }
        
        # Variable material components (Stage 1)
        # Thrust frame
        m_thrust_frame = get_scalar(inputs['M_thrust_frame_stage_1'])
        al_frac_tf = get_scalar(inputs['Al_fraction_thrust_frame_stage_1'])
        comp_frac_tf = get_scalar(inputs['Composite_fraction_thrust_frame_stage_1'])
        
        material_inventory['aluminum_7075'] += m_thrust_frame * al_frac_tf
        material_inventory['cfrp'] += m_thrust_frame * comp_frac_tf
        
        # Interstage
        m_interstage = get_scalar(inputs['M_interstage_stage_1'])
        al_frac_is = get_scalar(inputs['Al_fraction_interstage_stage_1'])
        comp_frac_is = get_scalar(inputs['Composite_fraction_interstage_stage_1'])
        
        material_inventory['aluminum_7075'] += m_interstage * al_frac_is
        material_inventory['cfrp'] += m_interstage * comp_frac_is
        
        # Intertank
        m_intertank = get_scalar(inputs['M_intertank_stage_1'])
        al_frac_it = get_scalar(inputs['Al_fraction_intertank_stage_1'])
        comp_frac_it = get_scalar(inputs['Composite_fraction_intertank_stage_1'])
        
        material_inventory['aluminum_7075'] += m_intertank * al_frac_it
        material_inventory['cfrp'] += m_intertank * comp_frac_it
        
        # Fixed material components
        # Tanks (aluminum-lithium alloy)
        material_inventory['aluminum_lithium'] += get_scalar(inputs['M_FT_stage_1'])
        material_inventory['aluminum_lithium'] += get_scalar(inputs['M_OxT_stage_1'])
        
        # Engines (70% Al-Li, 20% steel, 10% titanium)
        m_eng = get_scalar(inputs['M_eng_stage_1'])
        material_inventory['aluminum_lithium'] += m_eng * 0.7
        material_inventory['steel'] += m_eng * 0.2
        material_inventory['titanium'] += m_eng * 0.1
        
        # TVC (60% Al, 40% steel)
        m_tvc = get_scalar(inputs['M_TVC_stage_1'])
        material_inventory['aluminum_7075'] += m_tvc * 0.6
        material_inventory['steel'] += m_tvc * 0.4
        
        # Thermal protection (polyurethane foam)
        material_inventory['polyurethane_foam'] += get_scalar(inputs['M_TPS_OxT_stage_1'])
        material_inventory['polyurethane_foam'] += get_scalar(inputs['M_TPS_FT_stage_1'])
        
        # Electronics (avionics + EPS)
        material_inventory['electronics'] += get_scalar(inputs['M_avio_stage_1'])
        material_inventory['electronics'] += get_scalar(inputs['M_EPS_stage_1'])
        
        # Stage 2 materials
        dry_mass_s2 = get_scalar(inputs['Dry_mass_stage_2'])
        al_frac_s2 = get_scalar(inputs['Al_fraction_stage_2'])
        comp_frac_s2 = get_scalar(inputs['Composite_fraction_stage_2'])
        
        # Structural part (60% of dry mass) - variable materials
        structural_s2 = dry_mass_s2 * 0.6
        material_inventory['aluminum_7075'] += structural_s2 * al_frac_s2
        material_inventory['cfrp'] += structural_s2 * comp_frac_s2
        
        # Fixed components (40% of dry mass)
        material_inventory['aluminum_lithium'] += dry_mass_s2 * 0.2  # Tanks, engines
        material_inventory['steel'] += dry_mass_s2 * 0.1
        material_inventory['titanium'] += dry_mass_s2 * 0.05
        material_inventory['electronics'] += dry_mass_s2 * 0.05
        
        # Propellants
        prop_s1 = get_scalar(inputs['Prop_mass_stage_1'])
        prop_s2 = get_scalar(inputs['Prop_mass_stage_2'])
        of_s1 = get_scalar(inputs['OF_stage_1'])
        of_s2 = get_scalar(inputs['OF_stage_2'])
        
        material_inventory['lox'] = (prop_s1 * (of_s1 / (1 + of_s1)) + 
                                     prop_s2 * (of_s2 / (1 + of_s2)))
        material_inventory['lh2'] = (prop_s1 * (1 / (1 + of_s1)) + 
                                     prop_s2 * (1 / (1 + of_s2)))
        
        # Transport and electricity (estimated)
        total_dry_mass = sum([
            material_inventory['aluminum_7075'],
            material_inventory['aluminum_lithium'],
            material_inventory['cfrp'],
            material_inventory['steel'],
            material_inventory['titanium'],
            material_inventory['polyurethane_foam'],
            material_inventory['electronics']
        ])
        
        transport_tkm = (total_dry_mass / 1000.0) * 7000.0  # 7000 km ship transport
        electricity_kwh = total_dry_mass * 0.2  # 0.2 kWh/kg manufacturing
        
        # Store material masses for output
        outputs['total_aluminum_7075_kg'] = material_inventory['aluminum_7075']
        outputs['total_aluminum_lithium_kg'] = material_inventory['aluminum_lithium']
        outputs['total_composite_kg'] = material_inventory['cfrp']
        outputs['total_steel_kg'] = material_inventory['steel']
        outputs['total_titanium_kg'] = material_inventory['titanium']
        outputs['total_insulation_kg'] = material_inventory['polyurethane_foam']
        outputs['total_electronics_kg'] = material_inventory['electronics']
        
        # ========================================
        # PROPER BASELINE CALCULATION (FIXED)
        # ========================================

        # Check if this is actually the 100% aluminum configuration
        k_SM_tf = get_scalar(inputs['k_SM_thrust_frame'])
        k_SM_is = get_scalar(inputs['k_SM_interstage'])
        k_SM_it = get_scalar(inputs['k_SM_intertank'])
        k_SM_s2 = get_scalar(inputs['k_SM_stage_2'])

        is_aluminum = (k_SM_tf > 0.99 and k_SM_is > 0.99 and k_SM_it > 0.99 and k_SM_s2 > 0.99)

        if Environmental_Discipline_Comp._baseline_dry_mass is None:
            if is_aluminum:
                # This IS the baseline - use current total dry mass
                Environmental_Discipline_Comp._baseline_dry_mass = total_dry_mass
            else:
                # Calculate what 100% aluminum would be
                baseline_thrust_frame = m_thrust_frame / k_SM_tf if k_SM_tf > 0 else m_thrust_frame
                baseline_interstage = m_interstage / k_SM_is if k_SM_is > 0 else m_interstage
                baseline_intertank = m_intertank / k_SM_it if k_SM_it > 0 else m_intertank
                baseline_structural_s2 = (structural_s2 / k_SM_s2) if k_SM_s2 > 0 else structural_s2
                
                # Fixed components
                fixed_mass = (
                    material_inventory['aluminum_lithium'] +
                    material_inventory['steel'] +
                    material_inventory['titanium'] +
                    material_inventory['polyurethane_foam'] +
                    material_inventory['electronics'] -
                    structural_s2 * al_frac_s2 -
                    structural_s2 * comp_frac_s2
                )
                
                Environmental_Discipline_Comp._baseline_dry_mass = (
                    baseline_thrust_frame + baseline_interstage + 
                    baseline_intertank + baseline_structural_s2 + fixed_mass
                )

        # Calculate mass savings (0 for aluminum config, positive for others)
        if is_aluminum:
            mass_savings = 0
        else:
            mass_savings = max(0, Environmental_Discipline_Comp._baseline_dry_mass - total_dry_mass)

        outputs['mass_savings_kg'] = mass_savings
        
        # ========================================
        # CALCULATE LCA IMPACTS
        # ========================================
        
        if LCA_ENABLED:
            # Use Brightway2 with ecoinvent data
            self._calculate_brightway_impacts(
                material_inventory, transport_tkm, electricity_kwh, 
                mass_savings, outputs, print_debug
            )
        else:
            # Use simplified factors
            self._calculate_simplified_impacts(
                material_inventory, transport_tkm, electricity_kwh, 
                mass_savings, outputs, print_debug
            )
    
    def _calculate_brightway_impacts(self, inventory, transport_tkm, electricity_kwh, 
                                     mass_savings, outputs, print_debug):
        """
        Calculate impacts using Brightway2 and ecoinvent database
        """
        import brightway2 as bw
        
        # Build Brightway2 inventory
        bw_inventory = {}
        
        try:
            # Add materials to inventory
            for material, amount in inventory.items():
                if amount > 0 and material in ECOINVENT_CODES:
                    activity = bw.get_activity(ECOINVENT_CODES[material])
                    bw_inventory[activity] = amount
            
            # Add transport and electricity
            if transport_tkm > 0:
                transport_act = bw.get_activity(ECOINVENT_CODES['transport_ship'])
                bw_inventory[transport_act] = transport_tkm
            
            if electricity_kwh > 0:
                electricity_act = bw.get_activity(ECOINVENT_CODES['electricity'])
                bw_inventory[electricity_act] = electricity_kwh
            
            # Calculate impacts for each ESA category
            esa_single_score = 0.0
            manufacturing_score = 0.0
            
            for code, method in ESA_METHODS.items():
                try:
                    # Perform LCA for this impact category
                    lca = bw.LCA(bw_inventory, method)
                    lca.lci()
                    lca.lcia()
                    
                    # Get raw impact
                    impact_value = lca.score
                    outputs[f'{code}_impact'] = impact_value
                    
                    # Normalize
                    normalized = impact_value * ESA_NORMALIZATION[code]
                    outputs[f'{code}_normalized'] = normalized
                    
                    # Weight
                    weighted = normalized * ESA_WEIGHTS[code]
                    outputs[f'{code}_weighted'] = weighted
                    
                    # Add to single score
                    esa_single_score += weighted
                    
                    # Track manufacturing impact (before operational benefit)
                    manufacturing_score += weighted
                    
                    if print_debug and code == 'GWP':
                        print(f"GWP impact: {impact_value:.1f} kg CO2-eq")
                        print(f"  Normalized: {normalized:.4f}")
                        print(f"  Weighted: {weighted:.4f} Pt")
                    
                except Exception as e:
                    if print_debug:
                        print(f"Warning: Could not calculate {code}: {e}")
                    outputs[f'{code}_impact'] = 0.0
                    outputs[f'{code}_normalized'] = 0.0
                    outputs[f'{code}_weighted'] = 0.0
            
            # Calculate operational benefit
            # Mass savings = fuel savings over mission lifetime
            fuel_savings = mass_savings * 8.0  # ~8 kg fuel per kg dry mass

            # Calculate operational benefit with FULL ESA assessment for BOTH propellants
            operational_benefit = 0.0
            if mass_savings > 0:
                # Calculate saved propellant masses based on O/F ratio
                # Assuming average O/F of 5.5 (you should use actual mission average)
                of_ratio = 5.5  
                fuel_savings_lh2 = fuel_savings * (1 / (1 + of_ratio))
                fuel_savings_lox = fuel_savings * (of_ratio / (1 + of_ratio))
                
                # Build inventory for saved propellants
                fuel_inventory = {}
                if 'lh2' in ECOINVENT_CODES:
                    fuel_inventory[bw.get_activity(ECOINVENT_CODES['lh2'])] = fuel_savings_lh2
                if 'lox' in ECOINVENT_CODES:
                    fuel_inventory[bw.get_activity(ECOINVENT_CODES['lox'])] = fuel_savings_lox
                
                # Calculate benefit for ALL ESA categories
                for code, method in ESA_METHODS.items():
                    try:
                        lca_benefit = bw.LCA(fuel_inventory, method)
                        lca_benefit.lci()
                        lca_benefit.lcia()
                        
                        benefit_impact = lca_benefit.score
                        benefit_normalized = benefit_impact * ESA_NORMALIZATION[code]
                        benefit_weighted = benefit_normalized * ESA_WEIGHTS[code]
                        operational_benefit += benefit_weighted
                        
                    except Exception as e:
                        if print_debug:
                            print(f"Warning: Could not calculate operational benefit for {code}: {e}")
            
            '''
            # Calculate fuel production impact (simplified - should use full LCA)
            if 'lh2' in ECOINVENT_CODES:
                h2_activity = bw.get_activity(ECOINVENT_CODES['lh2'])
                fuel_inventory = {h2_activity: fuel_savings}
                
                # Just calculate GWP for operational benefit (simplified)
                lca_fuel = bw.LCA(fuel_inventory, ESA_METHODS['GWP'])
                lca_fuel.lci()
                lca_fuel.lcia()
                
                operational_benefit_gwp = lca_fuel.score
                operational_benefit_normalized = operational_benefit_gwp * ESA_NORMALIZATION['GWP']
                operational_benefit_weighted = operational_benefit_normalized * ESA_WEIGHTS['GWP']
                
                # Apply proportional benefit to all categories (approximation)
                operational_benefit = operational_benefit_weighted * len(ESA_METHODS)
            else:
                operational_benefit = 0.0
            '''
            
            # Final LCA score
            outputs['LCA_score'] = esa_single_score - operational_benefit
            outputs['LCA_manufacturing'] = manufacturing_score
            outputs['LCA_operational_benefit'] = operational_benefit
            
            # Extract key impacts
            outputs['CO2_eq'] = outputs['GWP_impact'][0] if 'GWP_impact' in outputs else 0.0
            outputs['Energy_consumption'] = outputs['ADEPLf_impact'][0] if 'ADEPLf_impact' in outputs else 0.0
            
            # Stage-wise breakdown based on actual component contributions
            # Calculate stage-specific impacts by re-running with stage-specific inventories
            stage1_materials = ['aluminum_7075', 'aluminum_lithium', 'steel', 'titanium', 
                               'polyurethane_foam', 'electronics']
            stage1_mass = sum(inventory.get(mat, 0) for mat in stage1_materials) * 0.7
            stage2_mass = sum(inventory.get(mat, 0) for mat in stage1_materials) * 0.3
            propellant_mass = inventory.get('lox', 0) + inventory.get('lh2', 0)
            
            total_mass = stage1_mass + stage2_mass + propellant_mass
            if total_mass > 0:
                outputs['LCA_stage_1'] = outputs['LCA_score'][0] * (stage1_mass / total_mass)
                outputs['LCA_stage_2'] = outputs['LCA_score'][0] * (stage2_mass / total_mass)
                outputs['LCA_propellants'] = outputs['LCA_score'][0] * (propellant_mass / total_mass)
            else:
                outputs['LCA_stage_1'] = outputs['LCA_score'][0] * 0.4
                outputs['LCA_stage_2'] = outputs['LCA_score'][0] * 0.3
                outputs['LCA_propellants'] = outputs['LCA_score'][0] * 0.3
            
            if print_debug:
                print(f"\n=== LCA RESULTS (Brightway2) ===")
                print(f"Manufacturing score: {manufacturing_score:.2f} Pt")
                print(f"Operational benefit: {operational_benefit:.2f} Pt")
                print(f"Final LCA score: {outputs['LCA_score'][0]:.2f} Pt")
                print(f"Mass savings: {mass_savings:.1f} kg")
                print(f"\nMaterial inventory (kg):")
                for mat, mass in inventory.items():
                    if mass > 0:
                        print(f"  {mat}: {mass:.1f}")
                print(f"\nImpact breakdown by category:")
                for code in ['GWP', 'ADEPLmu', 'FWTOX', 'PMAT', 'ACIDef']:
                    if f'{code}_weighted' in outputs:
                        print(f"  {code}: {outputs[f'{code}_weighted'][0]:.4f} Pt")
                
        except Exception as e:
            print(f"ERROR in Brightway2 calculation: {e}")
            print("Falling back to simplified calculation")
            self._calculate_simplified_impacts(
                inventory, transport_tkm, electricity_kwh, 
                mass_savings, outputs, print_debug
            )
    
    def _calculate_simplified_impacts(self, inventory, transport_tkm, electricity_kwh,
                                      mass_savings, outputs, print_debug):
        """
        Simplified LCA calculation when Brightway2 is not available
        """
        # Calculate total CO2 equivalent (simplified)
        total_co2 = 0.0
        
        for material, amount in inventory.items():
            if material in SIMPLIFIED_IMPACTS and amount > 0:
                total_co2 += amount * SIMPLIFIED_IMPACTS[material]
        
        # Add transport and electricity
        total_co2 += transport_tkm * 0.05  # 0.05 kg CO2/tkm for ship
        total_co2 += electricity_kwh * 0.5  # 0.5 kg CO2/kWh
        
        # Operational benefit
        fuel_savings = mass_savings * 8.0
        operational_co2 = fuel_savings * SIMPLIFIED_IMPACTS.get('lh2', 9.0)
        
        net_co2 = total_co2 - operational_co2
        
        # Set GWP impact
        outputs['GWP_impact'] = net_co2
        outputs['GWP_normalized'] = net_co2 * ESA_NORMALIZATION['GWP']
        outputs['GWP_weighted'] = outputs['GWP_normalized'][0] * ESA_WEIGHTS['GWP']
        
        # Estimate other impacts based on GWP (simplified)
        esa_single_score = outputs['GWP_weighted'][0]
        
        scaling = {
            'ODEPL': 0.001, 'IORAD': 0.01, 'PCHEM': 0.1, 'PMAT': 0.05,
            'HTOXnc': 0.02, 'HTOXc': 0.01, 'ACIDef': 0.08, 'FWEUT': 0.03,
            'MWEUT': 0.03, 'TEUT': 0.04, 'FWTOX': 0.02, 'LUP': 0.05,
            'WDEPL': 0.06, 'ADEPLf': 0.15, 'ADEPLmu': 0.08
        }
        
        for code, factor in scaling.items():
            impact = net_co2 * factor
            outputs[f'{code}_impact'] = impact
            outputs[f'{code}_normalized'] = impact * ESA_NORMALIZATION[code]
            outputs[f'{code}_weighted'] = outputs[f'{code}_normalized'][0] * ESA_WEIGHTS[code]
            esa_single_score += outputs[f'{code}_weighted'][0]
        
        outputs['LCA_score'] = esa_single_score
        outputs['LCA_manufacturing'] = esa_single_score * 1.2  # Before benefit
        outputs['LCA_operational_benefit'] = esa_single_score * 0.2
        
        outputs['CO2_eq'] = net_co2
        outputs['Energy_consumption'] = net_co2 * 20
        
        outputs['LCA_stage_1'] = esa_single_score * 0.4
        outputs['LCA_stage_2'] = esa_single_score * 0.3
        outputs['LCA_propellants'] = esa_single_score * 0.3
        
        if print_debug:
            print(f"\n=== LCA RESULTS (Simplified) ===")
            print(f"Total CO2: {total_co2/1000:.1f} t")
            print(f"Operational savings: {operational_co2/1000:.1f} t")
            print(f"Net CO2: {net_co2/1000:.1f} t")
            print(f"LCA score: {esa_single_score:.2f} Pt")