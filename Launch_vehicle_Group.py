"""
Launch Vehicle Group with Environmental/LCA Discipline Integration
CORRECTED VERSION - proper path names for connections
"""

import numpy as np
from openmdao.api import Group, IndepVarComp
import propulsion.Propulsion as Propulsion
import trajectory.Trajectory as Trajectory
import structure.Dry_Mass_stage_1 as Dry_Mass_stage_1 
import structure.Dry_Mass_stage_2 as Dry_Mass_stage_2
import aerodynamics.Aerodynamics as Aerodynamics
import environment.Environmental_Discipline as Environmental_Discipline

class Launcher_vehicle(Group):
           
    def setup(self):
        
        # Independent variables
        indeps = self.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        
        # Existing design variables
        indeps.add_output('Diameter_stage_1',4.6)
        indeps.add_output('Diameter_stage_2',4.6)
        indeps.add_output('Mass_flow_rate_stage_1',300.)
        indeps.add_output('Mass_flow_rate_stage_2',200.)
        indeps.add_output('Thrust_stage_1',1000.)
        indeps.add_output('Thrust_stage_2',1000.)
        indeps.add_output('Pc_stage_1',80.)
        indeps.add_output('Pc_stage_2',60.)
        indeps.add_output('Pe_stage_1',1.)
        indeps.add_output('Pe_stage_2',1.)
        indeps.add_output('OF_stage_1',5.0)
        indeps.add_output('OF_stage_2',5.5)
        indeps.add_output('N_eng_stage_1',6.)
        indeps.add_output('N_eng_stage_2',1.)
        indeps.add_output('Prop_mass_stage_1',350000.)
        indeps.add_output('Prop_mass_stage_2',75000.)
        indeps.add_output('Pdyn_max_dim',40.)
        indeps.add_output('thetacmd_i',2.72)
        indeps.add_output('thetacmd_f',20.)
        indeps.add_output('ksi',0.293)
        indeps.add_output('Pitch_over_duration',5.)
        indeps.add_output('Exit_nozzle_area_stage_1',0.79)
        indeps.add_output('Exit_nozzle_area_stage_2',3.6305)
        indeps.add_output('Delta_vertical_phase',10.)
        indeps.add_output('Delta_theta_pitch_over',1.)
        indeps.add_output('is_fallout',0.)        
        indeps.add_output('command_stage_1_exo',np.array([1.,1.]))
        
        # NEW: Material composition design variables
        indeps.add_output('k_SM_thrust_frame', 0.81)  # Default 50% Al, 50% Composite
        indeps.add_output('k_SM_interstage', 0.85)    # Default 50% Al, 50% Composite
        indeps.add_output('k_SM_intertank', 0.9)      # Default 50% Al, 50% Composite

        # Create cycle group with promotion
        cycle = self.add_subsystem('cycle', Group(), promotes=['*'])
        
        # Propulsion discipline
        Propu = cycle.add_subsystem('Propu', Propulsion.Propulsion_Comp(),
                                   promotes_inputs=['Pc_stage_1','Pe_stage_1',
                                                   'OF_stage_1','Pc_stage_2','Pe_stage_2','OF_stage_2'],
                                   promotes_outputs=['Isp_stage_1','Isp_stage_2'])
           
        # Structure Stage 1 - DON'T promote the individual mass outputs
        Struct_1 = cycle.add_subsystem('Struct_1', Dry_Mass_stage_1.Dry_Mass_stage_1_Comp(),
                                       promotes_inputs=['Diameter_stage_1','OF_stage_1',
                                                       'N_eng_stage_1','Diameter_stage_2','Isp_stage_1',
                                                       'Prop_mass_stage_1','Thrust_stage_1','Pdyn_max_dim',
                                                       'k_SM_thrust_frame','k_SM_interstage','k_SM_intertank'],
                                       promotes_outputs=['Dry_mass_stage_1'])
        
        # Structure Stage 2
        Struct_2 = cycle.add_subsystem('Struct_2', Dry_Mass_stage_2.Dry_Mass_stage_2_Comp(),
                                       promotes_inputs=['Prop_mass_stage_2'],
                                       promotes_outputs=['Dry_mass_stage_2'])

        # Aerodynamics discipline
        Aero = cycle.add_subsystem('Aero', Aerodynamics.Aerodynamics_Comp(),
                                  promotes_outputs=['Table_CX_complete_ascent',
                                                   'Mach_table','AoA_table','CX_fallout_stage_1',
                                                   'CZ_fallout_stage_1'])

        # Trajectory discipline
        Traj = cycle.add_subsystem('Traj', Trajectory.Trajectory_comp(),
                                   promotes_inputs=['Diameter_stage_1','Diameter_stage_2',
                                                   'Mass_flow_rate_stage_1','Mass_flow_rate_stage_2',
                                                   'N_eng_stage_1','N_eng_stage_2',
                                                   'OF_stage_1','OF_stage_2','Isp_stage_1','Isp_stage_2',
                                                   'Prop_mass_stage_1','Prop_mass_stage_2',
                                                   'Dry_mass_stage_1','Dry_mass_stage_2',
                                                   'Pitch_over_duration','thetacmd_i',
                                                   'thetacmd_f','ksi','Exit_nozzle_area_stage_1',
                                                   'Exit_nozzle_area_stage_2',
                                                   'Delta_vertical_phase','Delta_theta_pitch_over',
                                                   'Table_CX_complete_ascent',
                                                   'Mach_table','AoA_table','command_stage_1_exo',
                                                   'CX_fallout_stage_1','CZ_fallout_stage_1','is_fallout'],
                                   promotes_outputs=['T_ascent','alt_ascent','flux_ascent','r_ascent',
                                                    'V_ascent','theta_ascent','alpha_ascent','nx_ascent',
                                                    'alpha_cont','Nb_pt_ascent','m_ascent','CX_ascent',
                                                    'GLOW','lat_ascent','gamma_ascent','longi_ascent',
                                                    'thrust_ascent','mass_flow_rate_ascent','Mach_ascent',
                                                    'pdyn_ascent','rho_ascent','distance_ascent',
                                                    'state_separation_stage_1',
                                                    'max_pdyn_load_ascent_stage_1',
                                                    'T_fallout','alt_fallout','flux_fallout','r_fallout',
                                                    'V_fallout','theta_fallout','alpha_fallout','nx_fallout',
                                                    'Nb_pt_fallout','m_fallout','CX_fallout',
                                                    'CZ_fallout','lat_fallout','gamma_fallout','longi_fallout',
                                                    'thrust_fallout','mass_flow_rate_fallout','Mach_fallout',
                                                    'pdyn_fallout','rho_fallout','distance_fallout'])
        
        # Environmental/LCA Discipline - DON'T promote
        Environmental = cycle.add_subsystem('Environmental', 
                                           Environmental_Discipline.Environmental_Discipline_Comp())
        
        # Connect Stage 1 component masses to Environmental discipline
        # Based on the error message, the correct paths are:
        # FROM: 'Struct_1.X' (not 'cycle.Struct_1.X')
        # TO: 'Environmental.Y' (not 'cycle.Environmental.Y')
        
        self.connect('Struct_1.M_eng', 'Environmental.M_eng_stage_1')
        self.connect('Struct_1.M_thrust_frame', 'Environmental.M_thrust_frame_stage_1')
        self.connect('Struct_1.M_FT', 'Environmental.M_FT_stage_1')
        self.connect('Struct_1.M_OxT', 'Environmental.M_OxT_stage_1')
        self.connect('Struct_1.M_TPS_OxT', 'Environmental.M_TPS_OxT_stage_1')
        self.connect('Struct_1.M_TPS_FT', 'Environmental.M_TPS_FT_stage_1')
        self.connect('Struct_1.M_TVC', 'Environmental.M_TVC_stage_1')
        self.connect('Struct_1.M_avio', 'Environmental.M_avio_stage_1')
        self.connect('Struct_1.M_EPS', 'Environmental.M_EPS_stage_1')
        self.connect('Struct_1.M_intertank', 'Environmental.M_intertank_stage_1')
        self.connect('Struct_1.M_interstage', 'Environmental.M_interstage_stage_1')
        
        # Connect material fractions
        self.connect('Struct_1.Al_fraction_thrust_frame', 
                    'Environmental.Al_fraction_thrust_frame_stage_1')
        self.connect('Struct_1.Composite_fraction_thrust_frame', 
                    'Environmental.Composite_fraction_thrust_frame_stage_1')
        self.connect('Struct_1.Al_fraction_interstage', 
                    'Environmental.Al_fraction_interstage_stage_1')
        self.connect('Struct_1.Composite_fraction_interstage', 
                    'Environmental.Composite_fraction_interstage_stage_1')
        self.connect('Struct_1.Al_fraction_intertank', 
                    'Environmental.Al_fraction_intertank_stage_1')
        self.connect('Struct_1.Composite_fraction_intertank', 
                    'Environmental.Composite_fraction_intertank_stage_1')
        
        # Connect Stage 2 and propellant masses
        # These are promoted to the top level, so use direct names
        self.connect('Dry_mass_stage_2', 'Environmental.Dry_mass_stage_2')
        self.connect('Prop_mass_stage_1', 'Environmental.Prop_mass_stage_1')
        self.connect('Prop_mass_stage_2', 'Environmental.Prop_mass_stage_2')
        self.connect('OF_stage_1', 'Environmental.OF_stage_1')
        self.connect('OF_stage_2', 'Environmental.OF_stage_2')