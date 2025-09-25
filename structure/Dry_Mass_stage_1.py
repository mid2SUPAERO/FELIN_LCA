# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:31:11 2018

@author: lbrevaul
"""
from __future__ import print_function
import numpy as np
from openmdao.api import ExplicitComponent
from scipy import integrate
from subprocess import Popen
import structure.Mass_models as mmf
import specifications as Spec


class Dry_Mass_stage_1_Comp(ExplicitComponent):
    def setup(self):

        ###Inputs definition
        self.add_input('Diameter_stage_1', val=3.0)
        self.add_input('OF_stage_1',val=1.)
        self.add_input('N_eng_stage_1', val=1.)
        self.add_input('Diameter_stage_2',val=1.)
        self.add_input('Isp_stage_1',val=1.)
        self.add_input('Prop_mass_stage_1',val=1.)
        self.add_input('Thrust_stage_1',val=1.)
        self.add_input('Pdyn_max_dim',val=1.)

        # Add k_SM as input variables with component-specific ranges
        # Thrust frame: 1.0 = 100% Al, 0.62 = 100% Composite
        self.add_input('k_SM_thrust_frame', val=1.0)
        # Interstage: 1.0 = 100% Al, 0.7 = 100% Composite  
        self.add_input('k_SM_interstage', val=1.0)
        # Intertank: 1.0 = 100% Al, 0.8 = 100% Composite
        self.add_input('k_SM_intertank', val=1.0)

        ###Output definition
        self.add_output('Dry_mass_stage_1',val=3000.)

        # Add individual mass outputs for LCA analysis
        self.add_output('M_eng', val=100.)
        self.add_output('M_thrust_frame', val=100.)
        self.add_output('M_FT', val=100.)
        self.add_output('M_OxT', val=100.)
        self.add_output('M_TPS_OxT', val=100.)
        self.add_output('M_TPS_FT', val=100.)
        self.add_output('M_TVC', val=100.)
        self.add_output('M_avio', val=100.)
        self.add_output('M_EPS', val=100.)
        self.add_output('M_intertank', val=100.)
        self.add_output('M_interstage', val=100.)

        # Material composition outputs for LCA
        self.add_output('Al_fraction_thrust_frame', val=1.0)
        self.add_output('Composite_fraction_thrust_frame', val=0.0)
        self.add_output('Al_fraction_interstage', val=1.0)
        self.add_output('Composite_fraction_interstage', val=0.0)
        self.add_output('Al_fraction_intertank', val=1.0)
        self.add_output('Composite_fraction_intertank', val=0.0)
        
    def compute(self, inputs, outputs):
        
        Constants={}
        
        Constants['Type_prop']='Cryogenic'
        Constants['Engine_cycle']='SC'
        Constants['P_tank_Ox'] = 3.0
        Constants['P_tank_F'] = 3.0
        Constants['Tank_config']='intertank'
        Constants['Stage_in_staging']='lower'
        #Constants['Type_struct_intertank']='Al'
        Constants['Techno_TVC']='electromechanic'
        Constants['Thrust_frame_material']='Al'
        Constants['Redundancy_level'] = 3
        Constants['Type_fuel']='LH2'                 #Type of fuel (here LH2)
        Constants['S_interstage']=0.0
        Constants['g0']=9.80665
        Constants['Type_interstage']='lower'
        Constants['SSM_TF_1st_stage'] = 1.25
        Constants['Masse_aux_stage_1'] = 3000.
        Constants['NX_max_dim'] = Spec.specifications['command']['ascent']['nx_max']
        
        Thrust_1 = inputs['Thrust_stage_1'][0]*1e3       
        Total_Thrust= inputs['N_eng_stage_1'][0]*Thrust_1
         
        
        #Sizing of the LOx and fuel tanks
        S_Ox,S_F,S_totale,S_dome,S_exterieur,L_total = mmf.sizing(inputs['Prop_mass_stage_1'][0], inputs['OF_stage_1'][0], inputs['Diameter_stage_1'][0],Constants['Type_fuel'])
        
        #Computation of Engine mass using Mass Estimation Regression
        M_engine = mmf.engine_mass(Thrust_1, Constants['Type_prop'],Constants['Engine_cycle'])
        M_eng = inputs['N_eng_stage_1'][0]*M_engine

        # Computation of thrust frame mass with variable k_SM
        # Instead of fixed material, use k_SM as continuous variable
        k_SM_tf = inputs['k_SM_thrust_frame'][0]
        #Computation of thrust frame mass (structure that support the engines and connect to the tanks)
        M_thrust_frame = mmf.thrust_frame_mass_variable(Total_Thrust/inputs['N_eng_stage_1'][0]/1000,M_eng/inputs['N_eng_stage_1'][0],Constants['NX_max_dim'],inputs['N_eng_stage_1'][0],k_SM_tf,Constants['SSM_TF_1st_stage'])
        
        #Computation of the tank masses
        M_F = inputs['Prop_mass_stage_1'][0]/(1+inputs['OF_stage_1'][0])
        M_OX = inputs['Prop_mass_stage_1'][0] - M_F
        mu_LOX = 1141.
        if Constants['Type_fuel'] == 'LH2':
        	mu_F = 70.85
        elif Constants['Type_fuel'] =='CH4':
        	mu_F = 422.36
        elif Constants['Type_fuel'] == 'RP1':
        	mu_F = 810. 
        
        k_SM_it = inputs['k_SM_intertank'][0]
        V_Ox = M_OX / mu_LOX
        V_FT = M_F / mu_F
        M_FT,M_OxT,M_TPS_OxT,M_TPS_FT,M_intertank = mmf.tank_mass_variable(inputs['Pdyn_max_dim'][0],Constants['NX_max_dim'],Constants['P_tank_Ox'],Constants['P_tank_F'],V_FT,V_Ox,inputs['Diameter_stage_1'][0],S_Ox,S_F,2*S_dome,S_totale,Constants['Type_prop'],Constants['Tank_config'],Constants['Stage_in_staging'],k_SM_it)
                    
        #Computation of the Thrust Vector Control (TVC) mass (to control the nozzle orientation)
        M_TVC = inputs['N_eng_stage_1'][0]*mmf.TVC_mass(Thrust_1,Constants['Techno_TVC'])
        
        #Computation of avionics and power equipment masses
        M_avio, M_EPS = mmf.EPS_avio_mass(S_exterieur,Constants['Redundancy_level'])
        	
        #Computation of interstage mass
        k_SM_is = inputs['k_SM_interstage'][0]
        M_interstage = mmf.mass_interstage_variable(Constants['S_interstage'],inputs['Diameter_stage_1'][0],inputs['Diameter_stage_2'][0],Constants['Type_interstage'],k_SM_is)

        #Computation of the structural dry mass (without the mass of the propellants)
        Dry_mass_stage_1 = M_eng+M_thrust_frame+M_FT+M_OxT+M_TPS_OxT+\
                        M_TPS_FT+M_TVC+M_avio+M_EPS+M_intertank+\
                        M_interstage+Constants['Masse_aux_stage_1']
                        
        outputs['Dry_mass_stage_1']=Dry_mass_stage_1
        
        # Output individual masses for LCA
        outputs['M_eng'] = M_eng
        outputs['M_thrust_frame'] = M_thrust_frame
        outputs['M_FT'] = M_FT
        outputs['M_OxT'] = M_OxT
        outputs['M_TPS_OxT'] = M_TPS_OxT
        outputs['M_TPS_FT'] = M_TPS_FT
        outputs['M_TVC'] = M_TVC
        outputs['M_avio'] = M_avio
        outputs['M_EPS'] = M_EPS
        outputs['M_intertank'] = M_intertank
        outputs['M_interstage'] = M_interstage
        
        '''
        # Calculate material fractions based on k_SM values
        # Thrust frame: k_SM ranges from 1.0 (100% Al) to 0.62 (100% Composite)
        # Linear interpolation: Al_fraction = (k_SM - 0.62) / (1.0 - 0.62)
        outputs['Al_fraction_thrust_frame'] = (k_SM_tf - 0.62) / 0.38
        outputs['Composite_fraction_thrust_frame'] = 1.0 - outputs['Al_fraction_thrust_frame'][0]
        
        # Interstage: k_SM ranges from 1.0 (100% Al) to 0.7 (100% Composite)
        # Linear interpolation: Al_fraction = (k_SM - 0.7) / (1.0 - 0.7)
        outputs['Al_fraction_interstage'] = (k_SM_is - 0.7) / 0.3
        outputs['Composite_fraction_interstage'] = 1.0 - outputs['Al_fraction_interstage'][0]

        # Intertank: k_SM ranges from 1.0 (100% Al) to 0.8 (100% Composite)  
        # Linear interpolation: Al_fraction = (k_SM - 0.8) / (1.0 - 0.8)
        outputs['Al_fraction_intertank'] = (k_SM_it - 0.8) / 0.2
        outputs['Composite_fraction_intertank'] = 1.0 - outputs['Al_fraction_intertank'][0]
        '''
        # After computing k_SM_*:
        al_tf = float(np.clip((k_SM_tf - 0.62)/0.38, 0.0, 1.0))
        outputs['Al_fraction_thrust_frame'] = al_tf
        outputs['Composite_fraction_thrust_frame'] = 1.0 - al_tf

        al_is = float(np.clip((k_SM_is - 0.7)/0.3, 0.0, 1.0))
        outputs['Al_fraction_interstage'] = al_is
        outputs['Composite_fraction_interstage'] = 1.0 - al_is

        al_it = float(np.clip((k_SM_it - 0.8)/0.2, 0.0, 1.0))
        outputs['Al_fraction_intertank'] = al_it
        outputs['Composite_fraction_intertank'] = 1.0 - al_it
