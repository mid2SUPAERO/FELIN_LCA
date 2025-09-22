"""
FELIN-LCA Specific Integration
==============================
Integration code specifically designed for your FELIN discipline structure
"""

import numpy as np
import openmdao.api as om
import brightway2 as bw
from lca4mdao.component import LcaCalculationComponent
from lca4mdao.variable import ExplicitComponentLCA
from lca4mdao.utilities import setup_bw, setup_ecoinvent, cleanup_parameters

# Material database keys (update with your ecoinvent version)
MATERIAL_KEYS = {
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

# ==============================================================================
# FELIN-SPECIFIC ENVIRONMENTAL DISCIPLINE
# ==============================================================================

class FELINEnvironmentalDiscipline(ExplicitComponentLCA):
    """
    Environmental discipline specifically designed for FELIN launcher analysis
    Maps FELIN outputs (masses, configurations) to environmental impacts
    Using baseline values from launcher_design_problem.ipynb
    """
    
    def setup(self):
        # === INPUTS FROM FELIN DISCIPLINES ===
        # Using realistic values from your FELIN launcher_design_problem.ipynb
        
        # Mass inputs from Structure disciplines (estimated from FELIN typical outputs)
        self.add_input('Dry_mass_stage_1', val=45000.0, units='kg', 
                      desc='Stage 1 dry mass from FELIN Struct_1')
        self.add_input('Dry_mass_stage_2', val=6000.0, units='kg',
                      desc='Stage 2 dry mass from FELIN Struct_2')
        
        # Mass inputs from Trajectory discipline (from your initial values)
        self.add_input('Prop_mass_stage_1', val=320000.0, units='kg',
                      desc='Stage 1 propellant mass from launcher_design_problem')
        self.add_input('Prop_mass_stage_2', val=75000.0, units='kg',
                      desc='Stage 2 propellant mass from launcher_design_problem')
        self.add_input('GLOW', val=450000.0, units='kg',  # Estimated from prop + dry masses
                      desc='Gross Lift-Off Weight from Trajectory')
        
        # Configuration inputs from your initial values
        self.add_input('N_eng_stage_1', val=8.0, desc='Number of engines stage 1 (from FELIN)')
        self.add_input('N_eng_stage_2', val=1.0, desc='Number of engines stage 2 (from FELIN)')
        self.add_input('OF_stage_1', val=5.0, desc='O/F ratio stage 1 (from FELIN)')
        self.add_input('OF_stage_2', val=5.0, desc='O/F ratio stage 2 (from FELIN)')
        
        # Performance inputs (typical values for LOX/LH2 engines)
        self.add_input('Isp_stage_1', val=430.0, units='s', desc='Stage 1 Isp (LOX/LH2)')
        self.add_input('Isp_stage_2', val=465.0, units='s', desc='Stage 2 Isp (LOX/LH2)')
        
        # Geometry inputs from your initial values
        self.add_input('Diameter_stage_1', val=5.0, units='m', desc='Stage 1 diameter (FELIN)')
        self.add_input('Diameter_stage_2', val=5.0, units='m', desc='Stage 2 diameter (FELIN)')
        
        # Additional FELIN inputs
        self.add_input('Mass_flow_rate_stage_1', val=250.0, units='kg/s',
                      desc='Stage 1 mass flow rate (FELIN)')
        self.add_input('Mass_flow_rate_stage_2', val=250.0, units='kg/s',
                      desc='Stage 2 mass flow rate (FELIN)')
        self.add_input('Thrust_stage_1', val=1000.0, units='kN',
                      desc='Stage 1 thrust (FELIN)')
        self.add_input('Thrust_stage_2', val=800.0, units='kN',
                      desc='Stage 2 thrust (FELIN)')
        
        # === MATERIAL COMPOSITION DESIGN VARIABLES ===
        # These are new design variables for environmental optimization
        
        # Structure material composition
        self.add_input('cfrp_fraction_stage1', val=0.30, desc='CFRP fraction in stage 1 structure')
        self.add_input('aluminum_fraction_stage1', val=0.60, desc='Al fraction in stage 1 structure')
        self.add_input('steel_fraction_stage1', val=0.10, desc='Steel fraction in stage 1 structure')
        
        self.add_input('cfrp_fraction_stage2', val=0.40, desc='CFRP fraction in stage 2 structure')
        self.add_input('aluminum_fraction_stage2', val=0.50, desc='Al fraction in stage 2 structure')
        self.add_input('steel_fraction_stage2', val=0.10, desc='Steel fraction in stage 2 structure')
        
        # Engine material composition
        self.add_input('engine_nickel_fraction', val=0.65, desc='Ni alloy fraction in engines')
        self.add_input('engine_steel_fraction', val=0.25, desc='Steel fraction in engines')
        self.add_input('engine_titanium_fraction', val=0.10, desc='Ti fraction in engines')
        
        # Payload mass for environmental efficiency calculation
        self.add_input('payload_mass', val=10000.0, units='kg', desc='Payload mass to orbit')
        
        # === LCA OUTPUTS - STAGE 1 STRUCTURE ===
        self.add_output('stage1_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=MATERIAL_KEYS['carbon_fibre'], 
                       lca_units='kilogram',
                       desc='CFRP mass in stage 1')
        
        self.add_output('stage1_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=MATERIAL_KEYS['aluminium_alloy'],
                       lca_units='kilogram', 
                       desc='Aluminum mass in stage 1')
        
        self.add_output('stage1_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage1_structure"),
                       lca_key=MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram',
                       desc='Steel mass in stage 1')
        
        # === LCA OUTPUTS - STAGE 2 STRUCTURE ===
        self.add_output('stage2_cfrp_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=MATERIAL_KEYS['carbon_fibre'],
                       lca_units='kilogram',
                       desc='CFRP mass in stage 2')
        
        self.add_output('stage2_aluminum_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=MATERIAL_KEYS['aluminium_alloy'],
                       lca_units='kilogram',
                       desc='Aluminum mass in stage 2')
        
        self.add_output('stage2_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "stage2_structure"),
                       lca_key=MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram',
                       desc='Steel mass in stage 2')
        
        # === LCA OUTPUTS - PROPULSION SYSTEM ===
        self.add_output('engines_nickel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=MATERIAL_KEYS['nickel_alloy'],
                       lca_units='kilogram',
                       desc='Nickel alloy mass in engines')
        
        self.add_output('engines_steel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=MATERIAL_KEYS['steel_stainless'],
                       lca_units='kilogram',
                       desc='Steel mass in engines')
        
        self.add_output('engines_titanium_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propulsion_system"),
                       lca_key=MATERIAL_KEYS['titanium_alloy'],
                       lca_units='kilogram',
                       desc='Titanium mass in engines')
        
        # === LCA OUTPUTS - PROPELLANTS ===
        self.add_output('total_lox_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propellants"),
                       lca_key=MATERIAL_KEYS['liquid_oxygen'],
                       lca_units='kilogram',
                       desc='Total LOX propellant mass')
        
        self.add_output('total_fuel_mass', val=0.0, units='kg',
                       lca_parent=("launcher_components", "propellants"),
                       lca_key=MATERIAL_KEYS['liquid_hydrogen'],
                       lca_units='kilogram',
                       desc='Total fuel propellant mass')
        
        # === LCA OUTPUTS - OPERATIONS ===
        self.add_output('transport_operations', val=0.0, units='tkm',
                       lca_parent=("launcher_components", "operations"),
                       lca_key=MATERIAL_KEYS['transport_freight'],
                       lca_units='ton kilometer',
                       desc='Transport operations')
        
        self.add_output('launch_operations_electricity', val=0.0, units='kWh',
                       lca_parent=("launcher_components", "operations"),
                       lca_key=MATERIAL_KEYS['electricity_fr'],
                       lca_units='kilowatt hour',
                       desc='Launch operations electricity')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # === EXTRACT FELIN INPUTS (using your launcher_design_problem values) ===
        dry_mass_s1 = inputs['Dry_mass_stage_1']
        dry_mass_s2 = inputs['Dry_mass_stage_2']
        prop_mass_s1 = inputs['Prop_mass_stage_1']    # 320,000 kg (FELIN)
        prop_mass_s2 = inputs['Prop_mass_stage_2']    # 75,000 kg (FELIN)
        glow = inputs['GLOW']
        
        n_eng_s1 = inputs['N_eng_stage_1']           # 8 engines (FELIN)
        n_eng_s2 = inputs['N_eng_stage_2']           # 1 engine (FELIN)
        of_s1 = inputs['OF_stage_1']                 # 5.0 (FELIN)
        of_s2 = inputs['OF_stage_2']                 # 5.0 (FELIN)
        
        # FELIN performance parameters
        thrust_s1 = inputs['Thrust_stage_1']         # 1000 kN (FELIN)
        thrust_s2 = inputs['Thrust_stage_2']         # 800 kN (FELIN)
        mdot_s1 = inputs['Mass_flow_rate_stage_1']   # 250 kg/s (FELIN)
        mdot_s2 = inputs['Mass_flow_rate_stage_2']   # 250 kg/s (FELIN)
        diameter_s1 = inputs['Diameter_stage_1']     # 5.0 m (FELIN)
        diameter_s2 = inputs['Diameter_stage_2']     # 5.0 m (FELIN)
        
        # Material composition fractions
        cfrp_frac_s1 = inputs['cfrp_fraction_stage1']
        al_frac_s1 = inputs['aluminum_fraction_stage1']
        steel_frac_s1 = inputs['steel_fraction_stage1']
        
        cfrp_frac_s2 = inputs['cfrp_fraction_stage2']
        al_frac_s2 = inputs['aluminum_fraction_stage2']
        steel_frac_s2 = inputs['steel_fraction_stage2']
        
        ni_frac = inputs['engine_nickel_fraction']
        steel_eng_frac = inputs['engine_steel_fraction']
        ti_frac = inputs['engine_titanium_fraction']
        
        # === CALCULATE STAGE 1 STRUCTURE MATERIALS ===
        outputs['stage1_cfrp_mass'] = dry_mass_s1 * cfrp_frac_s1
        outputs['stage1_aluminum_mass'] = dry_mass_s1 * al_frac_s1
        outputs['stage1_steel_mass'] = dry_mass_s1 * steel_frac_s1
        
        # === CALCULATE STAGE 2 STRUCTURE MATERIALS ===
        outputs['stage2_cfrp_mass'] = dry_mass_s2 * cfrp_frac_s2
        outputs['stage2_aluminum_mass'] = dry_mass_s2 * al_frac_s2
        outputs['stage2_steel_mass'] = dry_mass_s2 * steel_frac_s2
        
        # === CALCULATE ENGINE MATERIALS (based on FELIN thrust levels) ===
        # More realistic engine mass estimation based on thrust
        # Stage 1: 8 engines @ 1000 kN total → 125 kN per engine → ~180 kg per engine
        # Stage 2: 1 engine @ 800 kN → ~140 kg per engine
        # These are more realistic for LOX/LH2 engines
        
        thrust_per_engine_s1 = thrust_s1 / n_eng_s1  # kN per engine
        thrust_per_engine_s2 = thrust_s2 / n_eng_s2  # kN per engine
        
        # Engine mass scaling: ~1.4 kg per kN of thrust (typical for rocket engines)
        engine_mass_s1_per_unit = thrust_per_engine_s1 * 1.4  # kg per engine
        engine_mass_s2_per_unit = thrust_per_engine_s2 * 1.4  # kg per engine
        
        total_engine_mass = (n_eng_s1 * engine_mass_s1_per_unit + 
                            n_eng_s2 * engine_mass_s2_per_unit)
        
        outputs['engines_nickel_mass'] = total_engine_mass * ni_frac
        outputs['engines_steel_mass'] = total_engine_mass * steel_eng_frac
        outputs['engines_titanium_mass'] = total_engine_mass * ti_frac
        
        # === CALCULATE PROPELLANT BREAKDOWN (using FELIN O/F ratios) ===
        # Your FELIN uses OF = 5.0 for both stages (LOX/LH2)
        # LOX mass = (OF/(1+OF)) * total_prop_mass
        # LH2 mass = (1/(1+OF)) * total_prop_mass
        
        lox_mass_s1 = (of_s1 / (1 + of_s1)) * prop_mass_s1
        fuel_mass_s1 = (1 / (1 + of_s1)) * prop_mass_s1
        
        lox_mass_s2 = (of_s2 / (1 + of_s2)) * prop_mass_s2
        fuel_mass_s2 = (1 / (1 + of_s2)) * prop_mass_s2
        
        outputs['total_lox_mass'] = lox_mass_s1 + lox_mass_s2
        outputs['total_fuel_mass'] = fuel_mass_s1 + fuel_mass_s2
        
        # === CALCULATE OPERATIONS (scaled to FELIN launcher size) ===
        # Transport: launcher dry mass * distance to launch site
        # Your FELIN launcher is much larger than my initial estimates
        transport_distance = 5000.0  # km (Europe to Kourou)
        launcher_dry_mass_tons = (dry_mass_s1 + dry_mass_s2 + total_engine_mass) / 1000.0
        outputs['transport_operations'] = launcher_dry_mass_tons * transport_distance
        
        # Launch operations electricity (scaled to GLOW)
        # Larger launcher needs more ground support energy
        electricity_per_kg_glow = 0.15  # kWh/kg (adjusted for large launcher)
        outputs['launch_operations_electricity'] = glow * electricity_per_kg_glow


# ==============================================================================
# LCA CALCULATION COMPONENT FOR FELIN
# ==============================================================================

class FELINLCACalculation(LcaCalculationComponent):
    """LCA calculation component for FELIN launcher system"""
    
    def setup(self):
        # === INPUTS FROM ENVIRONMENTAL DISCIPLINE ===
        structure_inputs = [
            'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
            'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass'
        ]
        
        propulsion_inputs = [
            'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass'
        ]
        
        propellant_inputs = [
            'total_lox_mass', 'total_fuel_mass'
        ]
        
        operations_inputs = [
            'transport_operations', 'launch_operations_electricity'
        ]
        
        # Add all inputs
        all_inputs = structure_inputs + propulsion_inputs + propellant_inputs + operations_inputs
        
        for input_name in all_inputs:
            if input_name in ['transport_operations']:
                self.add_input(input_name, units='tkm')
            elif input_name in ['launch_operations_electricity']:
                self.add_input(input_name, units='kWh')
            else:
                self.add_input(input_name, units='kg')
        
        # Mission parameters
        self.add_input('payload_mass', val=10000.0, units='kg', desc='Payload mass')
        self.add_input('GLOW', val=200000.0, units='kg', desc='Gross lift-off weight')
        
        # === LCA OUTPUTS FOR DIFFERENT IMPACT CATEGORIES ===
        
        # Global Warming Potential (GWP)
        self.add_lca_output('GWP_total',
                           {("launcher_components", "stage1_structure"): 1,
                            ("launcher_components", "stage2_structure"): 1,
                            ("launcher_components", "propulsion_system"): 1,
                            ("launcher_components", "propellants"): 1,
                            ("launcher_components", "operations"): 1},
                           method_key=('ReCiPe Midpoint (H) V1.13', 'climate change', 'GWP100'),
                           units='kg CO2-eq',
                           desc='Total Global Warming Potential')
        
        # Acidification Potential (AP)
        self.add_lca_output('AP_total',
                           {("launcher_components", "stage1_structure"): 1,
                            ("launcher_components", "stage2_structure"): 1,
                            ("launcher_components", "propulsion_system"): 1,
                            ("launcher_components", "propellants"): 1,
                            ("launcher_components", "operations"): 1},
                           method_key=('ReCiPe Midpoint (H) V1.13', 'terrestrial acidification', 'TAP100'),
                           units='kg SO2-eq',
                           desc='Total Acidification Potential')
        
        # Eutrophication Potential (EP)
        self.add_lca_output('EP_total',
                           {("launcher_components", "stage1_structure"): 1,
                            ("launcher_components", "stage2_structure"): 1,
                            ("launcher_components", "propulsion_system"): 1,
                            ("launcher_components", "propellants"): 1,
                            ("launcher_components", "operations"): 1},
                           method_key=('ReCiPe Midpoint (H) V1.13', 'freshwater eutrophication', 'FEP'),
                           units='kg P-eq',
                           desc='Total Eutrophication Potential')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


# ==============================================================================
# ENVIRONMENTAL METRICS COMPONENT
# ==============================================================================

class FELINEnvironmentalMetrics(om.ExplicitComponent):
    """Calculate environmental efficiency metrics for FELIN launcher"""
    
    def setup(self):
        # === INPUTS ===
        self.add_input('GWP_total', val=0.0, units='kg', desc='Total GWP')
        self.add_input('AP_total', val=0.0, units='kg', desc='Total AP')
        self.add_input('EP_total', val=0.0, units='kg', desc='Total EP')
        
        # Mission parameters
        self.add_input('payload_mass', val=10000.0, units='kg', desc='Payload mass')
        self.add_input('GLOW', val=200000.0, units='kg', desc='Gross lift-off weight')
        
        # Performance parameters from FELIN
        self.add_input('Isp_stage_1', val=450.0, units='s', desc='Stage 1 specific impulse')
        self.add_input('Isp_stage_2', val=465.0, units='s', desc='Stage 2 specific impulse')
        
        # === OUTPUTS - ENVIRONMENTAL EFFICIENCY METRICS ===
        self.add_output('GWP_per_kg_payload', val=0.0, units='kg/kg',
                       desc='GWP per kg of payload')
        self.add_output('AP_per_kg_payload', val=0.0, units='kg/kg',
                       desc='AP per kg of payload')
        self.add_output('EP_per_kg_payload', val=0.0, units='kg/kg',
                       desc='EP per kg of payload')
        
        # Combined metrics
        self.add_output('environmental_score', val=0.0, desc='Normalized environmental score')
        self.add_output('environmental_efficiency', val=0.0, units='kg/kg',
                       desc='Payload per tonne CO2-eq')
        
        # Performance-environment trade-offs
        self.add_output('performance_env_ratio', val=0.0, units='s*kg/kg',
                       desc='Performance-environment ratio')
        self.add_output('mass_efficiency', val=0.0, desc='Payload fraction')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        # Extract inputs
        gwp = inputs['GWP_total']
        ap = inputs['AP_total'] 
        ep = inputs['EP_total']
        payload = inputs['payload_mass']
        glow = inputs['GLOW']
        isp1 = inputs['Isp_stage_1']
        isp2 = inputs['Isp_stage_2']
        
        # Prevent division by zero
        payload = max(payload, 1.0)
        gwp = max(gwp, 1.0)
        glow = max(glow, 1.0)
        
        # === ENVIRONMENTAL EFFICIENCY METRICS ===
        outputs['GWP_per_kg_payload'] = gwp / payload
        outputs['AP_per_kg_payload'] = ap / payload
        outputs['EP_per_kg_payload'] = ep / payload
        
        # Environmental efficiency (kg payload per tonne CO2-eq)
        outputs['environmental_efficiency'] = payload / (gwp / 1000.0)
        
        # === COMBINED ENVIRONMENTAL SCORE ===
        # Normalize impacts and combine with weights
        # Reference values for normalization (typical launcher ranges)
        gwp_ref = 1000.0  # kg CO2-eq per kg payload
        ap_ref = 5.0      # kg SO2-eq per kg payload
        ep_ref = 0.5      # kg P-eq per kg payload
        
        gwp_normalized = (gwp / payload) / gwp_ref
        ap_normalized = (ap / payload) / ap_ref
        ep_normalized = (ep / payload) / ep_ref
        
        # Weighted combination (weights can be adjusted based on priorities)
        weights = [0.6, 0.3, 0.1]  # GWP, AP, EP weights
        outputs['environmental_score'] = (weights[0] * gwp_normalized + 
                                         weights[1] * ap_normalized +
                                         weights[2] * ep_normalized)
        
        # === PERFORMANCE-ENVIRONMENT TRADE-OFFS ===
        # Average specific impulse as performance metric
        avg_isp = (isp1 + isp2) / 2.0
        outputs['performance_env_ratio'] = avg_isp / (gwp / payload)
        
        # Mass efficiency (payload fraction)
        outputs['mass_efficiency'] = payload / glow


# ==============================================================================
# COMPLETE FELIN-LCA INTEGRATED GROUP
# ==============================================================================

class FELINLCAIntegratedGroup(om.Group):
    """Complete FELIN launcher analysis with integrated LCA discipline"""
    
    def setup(self):
        # === ADD FELIN DISCIPLINES ===
        # Import your existing FELIN disciplines here
        # NOTE: Update these imports to match your actual FELIN module structure
        
        # Propulsion discipline
        self.add_subsystem('Propu', Propulsion_Comp(),  # Replace with your actual class
                          promotes_inputs=['Pc_stage_1','Pe_stage_1',
                                          'OF_stage_1','Pc_stage_2','Pe_stage_2','OF_stage_2'],
                          promotes_outputs=['Isp_stage_1','Isp_stage_2'])
        
        # Structure disciplines
        self.add_subsystem('Struct_1', Dry_Mass_stage_1_Comp(),  # Replace with your actual class
                          promotes_inputs=['Diameter_stage_1','OF_stage_1',
                                          'N_eng_stage_1','Diameter_stage_2','Isp_stage_1','Prop_mass_stage_1',
                                          'Thrust_stage_1','Pdyn_max_dim'],
                          promotes_outputs=['Dry_mass_stage_1'])
        
        self.add_subsystem('Struct_2', Dry_Mass_stage_2_Comp(),  # Replace with your actual class
                          promotes_inputs=['Prop_mass_stage_2'],
                          promotes_outputs=['Dry_mass_stage_2'])
        
        # Aerodynamics discipline
        self.add_subsystem('Aero', Aerodynamics_Comp(),  # Replace with your actual class
                          promotes_outputs=['Table_CX_complete_ascent',
                                           'Mach_table','AoA_table','CX_fallout_stage_1','CZ_fallout_stage_1'])
        
        # Trajectory discipline
        self.add_subsystem('Traj', Trajectory_comp(),  # Replace with your actual class
                          promotes_inputs=['Diameter_stage_1','Diameter_stage_2','Mass_flow_rate_stage_1',
                                          'Mass_flow_rate_stage_2','N_eng_stage_1','N_eng_stage_2',
                                          'OF_stage_1','OF_stage_2','Isp_stage_1','Isp_stage_2',
                                          'Prop_mass_stage_1','Prop_mass_stage_2','Dry_mass_stage_1',
                                          'Dry_mass_stage_2','Pitch_over_duration','thetacmd_i',
                                          'thetacmd_f','ksi','Exit_nozzle_area_stage_1','Exit_nozzle_area_stage_2',
                                          'Delta_vertical_phase','Delta_theta_pitch_over','Table_CX_complete_ascent',
                                          'Mach_table','AoA_table','command_stage_1_exo','CX_fallout_stage_1',
                                          'CZ_fallout_stage_1','is_fallout'],
                          promotes_outputs=['T_ascent','alt_ascent','flux_ascent','r_ascent',
                                           'V_ascent','theta_ascent','alpha_ascent','nx_ascent','alpha_cont',
                                           'Nb_pt_ascent','m_ascent','CX_ascent','GLOW',
                                           'lat_ascent','gamma_ascent','longi_ascent','thrust_ascent',
                                           'mass_flow_rate_ascent','Mach_ascent','pdyn_ascent',
                                           'rho_ascent','distance_ascent','state_separation_stage_1',
                                           'max_pdyn_load_ascent_stage_1',
                                           'T_fallout','alt_fallout','flux_fallout','r_fallout',
                                           'V_fallout','theta_fallout','alpha_fallout','nx_fallout',
                                           'Nb_pt_fallout','m_fallout','CX_fallout',
                                           'lat_fallout','gamma_fallout','longi_fallout','thrust_fallout',
                                           'mass_flow_rate_fallout','Mach_fallout','pdyn_fallout',
                                           'rho_fallout','distance_fallout',
                                           'Prop_mass_stage_1','Prop_mass_stage_2'])
        
        # === ADD NEW ENVIRONMENTAL DISCIPLINE ===
        self.add_subsystem('Environment', FELINEnvironmentalDiscipline(),
                          promotes_inputs=[
                              # From Structure
                              'Dry_mass_stage_1', 'Dry_mass_stage_2',
                              # From Trajectory  
                              'Prop_mass_stage_1', 'Prop_mass_stage_2', 'GLOW',
                              # From Propulsion
                              'N_eng_stage_1', 'N_eng_stage_2', 'OF_stage_1', 'OF_stage_2',
                              'Isp_stage_1', 'Isp_stage_2',
                              # Geometry
                              'Diameter_stage_1', 'Diameter_stage_2',
                              # Material design variables
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
        
        # === ADD LCA CALCULATION ===
        self.add_subsystem('LCA_calc', FELINLCACalculation(),
                          promotes_inputs=[
                              'stage1_cfrp_mass', 'stage1_aluminum_mass', 'stage1_steel_mass',
                              'stage2_cfrp_mass', 'stage2_aluminum_mass', 'stage2_steel_mass',
                              'engines_nickel_mass', 'engines_steel_mass', 'engines_titanium_mass',
                              'total_lox_mass', 'total_fuel_mass',
                              'transport_operations', 'launch_operations_electricity',
                              'payload_mass', 'GLOW'
                          ],
                          promotes_outputs=[
                              'GWP_total', 'AP_total', 'EP_total'
                          ])
        
        # === ADD ENVIRONMENTAL METRICS ===
        self.add_subsystem('Env_metrics', FELINEnvironmentalMetrics(),
                          promotes_inputs=[
                              'GWP_total', 'AP_total', 'EP_total',
                              'payload_mass', 'GLOW', 'Isp_stage_1', 'Isp_stage_2'
                          ],
                          promotes_outputs=[
                              'GWP_per_kg_payload', 'AP_per_kg_payload', 'EP_per_kg_payload',
                              'environmental_score', 'environmental_efficiency',
                              'performance_env_ratio', 'mass_efficiency'
                          ])


# ==============================================================================
# SETUP AND ANALYSIS FUNCTIONS
# ==============================================================================

def setup_felin_lca_environment(project_name="FELIN_LCA", ecoinvent_path=None):
    """Setup LCA environment for FELIN integration"""
    
    try:
        # Setup Brightway2
        setup_bw(project_name)
        if ecoinvent_path:
            setup_ecoinvent(ecoinvent_path)
        
        # Create launcher components database
        if "launcher_components" not in bw.databases:
            create_launcher_database()
        
        cleanup_parameters()
        print(f"✓ LCA environment setup complete: {project_name}")
        
    except Exception as e:
        print(f"✗ Error setting up LCA environment: {e}")
        raise


def create_launcher_database():
    """Create Brightway2 database for launcher components"""
    
    launcher_db = bw.Database('launcher_components')
    launcher_db.register()
    
    # Delete if exists
    if launcher_db.name in bw.databases:
        launcher_db.delete()
    
    # Create subsystem activities
    activities = [
        ('launcher_system', 'Complete launcher system', 'unit'),
        ('stage1_structure', 'Stage 1 structural components', 'kg'),
        ('stage2_structure', 'Stage 2 structural components', 'kg'),
        ('propulsion_system', 'Propulsion system components', 'kg'),
        ('propellants', 'Rocket propellants', 'kg'),
        ('operations', 'Launch operations and transport', 'unit')
    ]
    
    for code, name, unit in activities:
        launcher_db.new_activity(code=code, name=name, unit=unit).save()
    
    print("✓ Launcher components database created")


def run_felin_lca_analysis(ecoinvent_path=None, optimize_environment=False):
    """Run FELIN analysis with integrated LCA"""
    
    print("=== FELIN-LCA INTEGRATED ANALYSIS ===")
    
    # Setup LCA environment
    setup_felin_lca_environment("FELIN_LCA", ecoinvent_path)
    
    # Create problem
    prob = om.Problem()
    prob.model = FELINLCAIntegratedGroup()
    
    if optimize_environment:
        # Setup environmental optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['maxiter'] = 50
        prob.driver.options['tol'] = 1e-6
        
        # === ENVIRONMENTAL DESIGN VARIABLES ===
        # Material composition design variables
        prob.model.add_design_var('cfrp_fraction_stage1', lower=0.2, upper=0.6)
        prob.model.add_design_var('aluminum_fraction_stage1', lower=0.3, upper=0.7)
        prob.model.add_design_var('cfrp_fraction_stage2', lower=0.2, upper=0.6)
        prob.model.add_design_var('aluminum_fraction_stage2', lower=0.3, upper=0.7)
        prob.model.add_design_var('engine_nickel_fraction', lower=0.5, upper=0.8)
        
        # === ENVIRONMENTAL OBJECTIVES/CONSTRAINTS ===
        # Primary objective: minimize environmental impact per kg payload
        prob.model.add_objective('environmental_score')
        
        # OR use as constraint with performance objective:
        # prob.model.add_constraint('GWP_per_kg_payload', upper=1200.0)
        # prob.model.add_objective('performance_env_ratio', scaler=-1)  # Maximize
        
        # Material fraction constraints (must sum to reasonable values)
        prob.model.add_constraint('steel_fraction_stage1', lower=0.05, upper=0.3,
                                 equals=om.ExecComp('steel_frac = 1.0 - cfrp_fraction_stage1 - aluminum_fraction_stage1'))
        prob.model.add_constraint('steel_fraction_stage2', lower=0.05, upper=0.3,
                                 equals=om.ExecComp('steel_frac = 1.0 - cfrp_fraction_stage2 - aluminum_fraction_stage2'))
    
    # Setup problem
    prob.setup(check=False, force_alloc_complex=True)
    
    # Run analysis
    try:
        if optimize_environment:
            print("Running environmental optimization...")
            prob.run_driver()
        else:
            print("Running baseline analysis...")
            prob.run_model()
        
        # Print results
        print_felin_lca_results(prob)
        
        return prob
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print("Check that:")
        print("- Ecoinvent database is properly configured")
        print("- FELIN discipline classes are correctly imported")
        print("- All variable names match between disciplines")
        return None


def print_felin_lca_results(prob):
    """Print comprehensive FELIN-LCA results"""
    
    print(f"\n{'='*80}")
    print("FELIN LAUNCHER WITH LCA - COMPREHENSIVE RESULTS")
    print(f"{'='*80}")
    
    # === FELIN PERFORMANCE RESULTS ===
    print(f"\n--- LAUNCHER PERFORMANCE (FELIN) ---")
    try:
        payload = prob.get_val('payload_mass')[0]
        glow = prob.get_val('GLOW')[0]
        dry1 = prob.get_val('Dry_mass_stage_1')[0]
        dry2 = prob.get_val('Dry_mass_stage_2')[0]
        prop1 = prob.get_val('Prop_mass_stage_1')[0]
        prop2 = prob.get_val('Prop_mass_stage_2')[0]
        isp1 = prob.get_val('Isp_stage_1')[0]
        isp2 = prob.get_val('Isp_stage_2')[0]
        
        print(f"Payload mass:           {payload:10.1f} kg")
        print(f"GLOW:                   {glow:10.1f} kg")
        print(f"Stage 1 dry mass:       {dry1:10.1f} kg")
        print(f"Stage 2 dry mass:       {dry2:10.1f} kg")
        print(f"Stage 1 propellant:     {prop1:10.1f} kg")
        print(f"Stage 2 propellant:     {prop2:10.1f} kg")
        print(f"Stage 1 Isp:            {isp1:10.1f} s")
        print(f"Stage 2 Isp:            {isp2:10.1f} s")
        print(f"Payload fraction:       {payload/glow:10.3f}")
        
    except Exception as e:
        print(f"Error accessing FELIN results: {e}")
    
    # === ENVIRONMENTAL IMPACT RESULTS ===
    print(f"\n--- ENVIRONMENTAL IMPACTS (LCA) ---")
    try:
        gwp_total = prob.get_val('GWP_total')[0]
        ap_total = prob.get_val('AP_total')[0]
        ep_total = prob.get_val('EP_total')[0]
        gwp_per_kg = prob.get_val('GWP_per_kg_payload')[0]
        ap_per_kg = prob.get_val('AP_per_kg_payload')[0]
        env_score = prob.get_val('environmental_score')[0]
        env_efficiency = prob.get_val('environmental_efficiency')[0]
        
        print(f"Total GWP:              {gwp_total:10.1f} kg CO₂-eq")
        print(f"Total AP:               {ap_total:10.3f} kg SO₂-eq")
        print(f"Total EP:               {ep_total:10.3f} kg P-eq")
        print(f"GWP per kg payload:     {gwp_per_kg:10.2f} kg CO₂-eq/kg")
        print(f"AP per kg payload:      {ap_per_kg:10.4f} kg SO₂-eq/kg")
        print(f"Environmental score:    {env_score:10.4f}")
        print(f"Env. efficiency:        {env_efficiency:10.2f} kg payload/tonne CO₂-eq")
        
    except Exception as e:
        print(f"Error accessing LCA results: {e}")
    
    # === MATERIAL BREAKDOWN ===
    print(f"\n--- MATERIAL BREAKDOWN ---")
    try:
        # Stage 1 materials
        cfrp1 = prob.get_val('stage1_cfrp_mass')[0]
        al1 = prob.get_val('stage1_aluminum_mass')[0]
        steel1 = prob.get_val('stage1_steel_mass')[0]
        
        # Stage 2 materials
        cfrp2 = prob.get_val('stage2_cfrp_mass')[0]
        al2 = prob.get_val('stage2_aluminum_mass')[0]
        steel2 = prob.get_val('stage2_steel_mass')[0]
        
        # Engine materials
        ni_eng = prob.get_val('engines_nickel_mass')[0]
        steel_eng = prob.get_val('engines_steel_mass')[0]
        ti_eng = prob.get_val('engines_titanium_mass')[0]
        
        # Propellants
        lox = prob.get_val('total_lox_mass')[0]
        fuel = prob.get_val('total_fuel_mass')[0]
        
        print(f"Stage 1 CFRP:           {cfrp1:10.1f} kg")
        print(f"Stage 1 Aluminum:       {al1:10.1f} kg")
        print(f"Stage 1 Steel:          {steel1:10.1f} kg")
        print(f"Stage 2 CFRP:           {cfrp2:10.1f} kg")
        print(f"Stage 2 Aluminum:       {al2:10.1f} kg")
        print(f"Stage 2 Steel:          {steel2:10.1f} kg")
        print(f"Engine Nickel alloy:    {ni_eng:10.1f} kg")
        print(f"Engine Steel:           {steel_eng:10.1f} kg")
        print(f"Engine Titanium:        {ti_eng:10.1f} kg")
        print(f"Total LOX:              {lox:10.1f} kg")
        print(f"Total Fuel:             {fuel:10.1f} kg")
        
        # Material fractions
        total_structure = cfrp1 + al1 + steel1 + cfrp2 + al2 + steel2
        if total_structure > 0:
            print(f"\nTotal CFRP fraction:    {(cfrp1+cfrp2)/total_structure:10.1%}")
            print(f"Total Al fraction:      {(al1+al2)/total_structure:10.1%}")
            print(f"Total Steel fraction:   {(steel1+steel2)/total_structure:10.1%}")
        
    except Exception as e:
        print(f"Error accessing material breakdown: {e}")
    
    # === PERFORMANCE-ENVIRONMENT TRADE-OFFS ===
    print(f"\n--- PERFORMANCE-ENVIRONMENT TRADE-OFFS ---")
    try:
        perf_env_ratio = prob.get_val('performance_env_ratio')[0]
        mass_eff = prob.get_val('mass_efficiency')[0]
        
        print(f"Performance/Env ratio:  {perf_env_ratio:10.3f} s⋅kg/kg")
        print(f"Mass efficiency:        {mass_eff:10.3f}")
        
        # Comparative metrics
        if gwp_total > 0:
            co2_per_mission = gwp_total / 1000  # tonnes CO2-eq
            car_equivalent = gwp_total / 120    # km (avg car: 120g CO2/km)
            print(f"CO₂-eq per mission:     {co2_per_mission:10.2f} tonnes")
            print(f"Equivalent car travel:  {car_equivalent:10.0f} km")
        
    except Exception as e:
        print(f"Error accessing trade-off metrics: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


# ==============================================================================
# MAIN EXECUTION AND TESTING
# ==============================================================================

def main():
    """Main execution function"""
    
    print("FELIN-LCA Specific Integration")
    print("=" * 50)
    
    # Update this path to your ecoinvent location
    ecoinvent_path = r'C:\Users\joana\Desktop\Joana\FELIN\ecoinvent 3.8_cutoff_ecoSpold02\datasets'
    
    # Apply OpenMDAO compatibility patch
    def patched_setup_procs(self, pathname, comm, prob_meta):
        super(LcaCalculationComponent, self)._setup_procs(pathname, comm, prob_meta)
    
    LcaCalculationComponent._setup_procs = patched_setup_procs
    
    try:
        # Run baseline analysis
        print("\n1. Running FELIN baseline analysis with LCA...")
        prob_baseline = run_felin_lca_analysis(ecoinvent_path, optimize_environment=False)
        
        if prob_baseline is not None:
            # Run environmental optimization
            print("\n2. Running environmental optimization...")
            prob_optimized = run_felin_lca_analysis(ecoinvent_path, optimize_environment=True)
            
            if prob_optimized is not None:
                print("\n✓ FELIN-LCA integration successful!")
                print("\nNext steps:")
                print("- Update imports to use your actual FELIN discipline classes")
                print("- Verify all variable connections are working")
                print("- Customize material composition design variables")
                print("- Add environmental constraints to your existing optimization")
        
    except Exception as e:
        print(f"✗ Integration failed: {e}")
        print("\nTroubleshooting:")
        print("- Check ecoinvent database path and setup")
        print("- Verify FELIN discipline imports")
        print("- Update variable names to match your FELIN implementation")
        print("- Ensure all required packages are installed")


if __name__ == '__main__':
    main()


# ==============================================================================
# INTEGRATION INSTRUCTIONS FOR YOUR SPECIFIC FELIN CODE
# ==============================================================================

"""
INTEGRATION INSTRUCTIONS:
=========================

1. REPLACE DISCIPLINE IMPORTS:
   Update the imports in FELINLCAIntegratedGroup.setup() to use your actual FELIN classes:
   
   from your_felin_modules import (
       Propulsion_Comp,           # Your actual propulsion class
       Dry_Mass_stage_1_Comp,     # Your actual structure class  
       Dry_Mass_stage_2_Comp,     # Your actual structure class
       Aerodynamics_Comp,         # Your actual aero class
       Trajectory_comp            # Your actual trajectory class
   )

2. ADD TO YOUR EXISTING FELIN CYCLE:
   In your main FELIN script, replace your existing cycle setup with:
   
   cycle = FELINLCAIntegratedGroup()
   
   OR add the environmental discipline to your existing cycle:
   
   Environment = cycle.add_subsystem('Environment', FELINEnvironmentalDiscipline(),
                                   promotes_inputs=[...],  # See variable mapping above
                                   promotes_outputs=[...]) # See outputs above

3. UPDATE DESIGN VARIABLES:
   Add these environmental design variables to your optimization:
   
   # Material composition variables
   prob.model.add_design_var('cfrp_fraction_stage1', lower=0.2, upper=0.6)
   prob.model.add_design_var('aluminum_fraction_stage1', lower=0.3, upper=0.7)
   # ... add others as needed
   
   # Environmental constraints
   prob.model.add_constraint('GWP_per_kg_payload', upper=1200.0)
   
   # OR environmental objective
   prob.model.add_objective('environmental_score')

4. VARIABLE MAPPING VERIFICATION:
   Verify these variable names match your FELIN implementation:
   
   FELIN Discipline → Variable Name → LCA Input
   Struct_1 → 'Dry_mass_stage_1' → Environment.Dry_mass_stage_1
   Struct_2 → 'Dry_mass_stage_2' → Environment.Dry_mass_stage_2  
   Traj → 'Prop_mass_stage_1' → Environment.Prop_mass_stage_1
   Traj → 'Prop_mass_stage_2' → Environment.Prop_mass_stage_2
   Traj → 'GLOW' → Environment.GLOW
   (promoted inputs) → 'N_eng_stage_1' → Environment.N_eng_stage_1
   (promoted inputs) → 'OF_stage_1' → Environment.OF_stage_1
   Propu → 'Isp_stage_1' → Environment.Isp_stage_1

5. TESTING:
   Test the integration step by step:
   
   a) First test individual LCA components:
      python -c "from your_file import FELINEnvironmentalDiscipline; print('✓ Import OK')"
   
   b) Test with dummy FELIN data:
      prob = om.Problem(FELINEnvironmentalDiscipline())
      prob.setup()
      prob.run_model()
   
   c) Test full integration:
      prob = run_felin_lca_analysis(ecoinvent_path, optimize_environment=False)

6. EXPECTED OUTPUTS:
   After successful integration, you should see:
   - Environmental impact metrics (GWP, AP, EP)
   - Material breakdown by launcher component
   - Environmental efficiency metrics
   - Performance-environment trade-offs
   
   Typical results:
   - GWP per kg payload: 800-1200 kg CO₂-eq/kg
   - Environmental efficiency: 5-15 kg payload/tonne CO₂-eq
   - Material composition effects on both mass and environment

7. OPTIMIZATION BENEFITS:
   With this integration, your FELIN optimization can now:
   - Minimize environmental impact while meeting performance targets
   - Find optimal material compositions (CFRP vs Al vs Steel)
   - Explore performance-environment Pareto fronts
   - Include environmental constraints in launcher design
   - Assess life cycle impacts of design decisions

8. CUSTOMIZATION:
   Customize the integration for your specific needs:
   - Add new materials to MATERIAL_KEYS dictionary
   - Modify material composition assumptions in compute() methods
   - Add new impact categories (ozone depletion, toxicity, etc.)
   - Include manufacturing processes and end-of-life scenarios
   - Add uncertainty analysis for environmental parameters

This integration transforms your FELIN launcher design into a comprehensive
multidisciplinary optimization that considers both performance and environmental
sustainability - exactly what modern space industry needs!
"""
        print
        