# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 15:08:27 2017

@author: lbrevaul
"""

from __future__ import print_function
import numpy as np
from openmdao.api import ExplicitComponent

'''
class Dry_Mass_stage_2_Comp(ExplicitComponent):
    def setup(self):
        self.add_input('Prop_mass_stage_2',val=1.)
		
        self.add_output('Dry_mass_stage_2',val=3000.)

    def compute(self, inputs, outputs):
        
        #Regression to estimate the dry mass (without the propellant) of the second stage as a function of the propellant mass
        outputs['Dry_mass_stage_2'] = (80.*(inputs['Prop_mass_stage_2']/1e3)**(-0.5))/100*inputs['Prop_mass_stage_2'] ## Transcost MODEL
'''

class Dry_Mass_stage_2_Comp(ExplicitComponent):
    def setup(self):
        # Input: Propellant mass
        self.add_input('Prop_mass_stage_2', val=75000.)
        
        # NEW: Material factor for Stage 2
        # k_SM ranges from 0.75 (100% Composite) to 1.0 (100% Aluminum)
        self.add_input('k_SM_stage_2', val=0.875)  # Default 50/50 mix
        
        # Output: Dry mass
        self.add_output('Dry_mass_stage_2', val=3000.)
        
        # NEW: Material fractions for LCA
        self.add_output('Al_fraction_stage_2', val=0.5)
        self.add_output('Composite_fraction_stage_2', val=0.5)
        
        # NEW: Component mass breakdown (simplified)
        self.add_output('M_structure_stage_2', val=1800.)
        self.add_output('M_propulsion_stage_2', val=600.)
        self.add_output('M_avionics_stage_2', val=300.)
        self.add_output('M_other_stage_2', val=300.)

    def compute(self, inputs, outputs):
        """
        Calculate Stage 2 dry mass with material optimization
        """
        # Get inputs
        prop_mass = inputs['Prop_mass_stage_2'][0]
        k_SM = inputs['k_SM_stage_2'][0]
        
        # Base mass from Transcost regression (100% aluminum baseline)
        base_mass_aluminum = (80. * (prop_mass/1e3)**(-0.5)) / 100 * prop_mass
        
        # Apply material factor
        # k_SM = 1.0 -> 100% aluminum (heavier)
        # k_SM = 0.75 -> 100% composite (25% lighter)
        # The structural part (60% of mass) is affected by material choice
        # The rest (engines, avionics, etc.) remains constant
        
        structural_fraction = 0.6  # 60% of dry mass is structure
        non_structural_fraction = 0.4  # 40% is fixed (engines, avionics, etc.)
        
        # Calculate component masses
        base_structural = base_mass_aluminum * structural_fraction
        base_non_structural = base_mass_aluminum * non_structural_fraction
        
        # Apply material factor only to structural components
        actual_structural = base_structural * k_SM
        
        # Total dry mass
        total_dry_mass = actual_structural + base_non_structural
        
        outputs['Dry_mass_stage_2'] = total_dry_mass
        
        # Calculate material fractions
        # k_SM ranges from 0.75 (pure composite) to 1.0 (pure aluminum)
        # Linear interpolation
        k_SM_min = 0.75  # 100% composite
        k_SM_max = 1.00  # 100% aluminum
        
        # Ensure k_SM is within bounds
        k_SM_bounded = np.clip(k_SM, k_SM_min, k_SM_max)
        
        # Calculate aluminum fraction
        al_fraction = (k_SM_bounded - k_SM_min) / (k_SM_max - k_SM_min)
        comp_fraction = 1.0 - al_fraction
        
        outputs['Al_fraction_stage_2'] = float(al_fraction)
        outputs['Composite_fraction_stage_2'] = float(comp_fraction)
        
        # Component mass breakdown
        outputs['M_structure_stage_2'] = actual_structural
        outputs['M_propulsion_stage_2'] = base_non_structural * 0.5  # ~50% of non-structural
        outputs['M_avionics_stage_2'] = base_non_structural * 0.25  # ~25% of non-structural
        outputs['M_other_stage_2'] = base_non_structural * 0.25  # ~25% of non-structural
        
        # Debug output
        if False:  # Set to True for debugging
            print(f"\n=== Stage 2 Mass Calculation ===")
            print(f"Propellant mass: {prop_mass:.1f} kg")
            print(f"k_SM: {k_SM:.3f}")
            print(f"Base mass (100% Al): {base_mass_aluminum:.1f} kg")
            print(f"Structural mass: {actual_structural:.1f} kg")
            print(f"Non-structural mass: {base_non_structural:.1f} kg")
            print(f"Total dry mass: {total_dry_mass:.1f} kg")
            print(f"Material mix: {al_fraction*100:.1f}% Al, {comp_fraction*100:.1f}% Composite")