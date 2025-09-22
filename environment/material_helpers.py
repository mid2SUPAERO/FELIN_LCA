# -*- coding: utf-8 -*-
"""
Material conversion helper functions for FELIN launcher optimization
Handles conversions between k_SM values and material fractions
"""

import numpy as np

class MaterialConverter:
    """
    Converts between k_SM structural factors and material mass fractions
    for launcher components with variable materials
    """
    
    # Component-specific k_SM ranges from literature
    K_SM_RANGES = {
        'thrust_frame': {
            'Al': 1.0,
            'Composite': 0.62  # 38% mass reduction with composite
        },
        'interstage': {
            'Al': 1.0,
            'Composite': 0.7   # 30% mass reduction with composite
        }
    }
    
    @staticmethod
    def k_SM_to_fractions(k_SM, component):
        """
        Convert k_SM value to aluminum and composite mass fractions
        
        Parameters:
        -----------
        k_SM : float
            Structural mass factor (between k_SM_composite and k_SM_Al)
        component : str
            Component name ('thrust_frame' or 'interstage')
            
        Returns:
        --------
        tuple : (Al_fraction, Composite_fraction)
            Mass fractions that sum to 1.0
        """
        if component not in MaterialConverter.K_SM_RANGES:
            raise ValueError(f"Unknown component: {component}")
        
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # Validate input
        if not (k_Comp <= k_SM <= k_Al):
            raise ValueError(f"k_SM {k_SM} out of range [{k_Comp}, {k_Al}] for {component}")
        
        # Linear interpolation
        Al_fraction = (k_SM - k_Comp) / (k_Al - k_Comp)
        Composite_fraction = 1.0 - Al_fraction
        
        return Al_fraction, Composite_fraction
    
    @staticmethod
    def fractions_to_k_SM(Al_fraction, component):
        """
        Convert material fractions to k_SM value
        
        Parameters:
        -----------
        Al_fraction : float
            Aluminum mass fraction (0 to 1)
        component : str
            Component name
            
        Returns:
        --------
        float : k_SM value
        """
        if component not in MaterialConverter.K_SM_RANGES:
            raise ValueError(f"Unknown component: {component}")
        
        if not (0 <= Al_fraction <= 1):
            raise ValueError(f"Al_fraction {Al_fraction} must be between 0 and 1")
        
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # Linear interpolation
        k_SM = Al_fraction * k_Al + (1 - Al_fraction) * k_Comp
        
        return k_SM
    
    @staticmethod
    def get_bounds(component):
        """
        Get the valid k_SM bounds for a component
        
        Parameters:
        -----------
        component : str
            Component name
            
        Returns:
        --------
        tuple : (lower_bound, upper_bound)
        """
        if component not in MaterialConverter.K_SM_RANGES:
            raise ValueError(f"Unknown component: {component}")
        
        return (MaterialConverter.K_SM_RANGES[component]['Composite'], 
                MaterialConverter.K_SM_RANGES[component]['Al'])
    
    @staticmethod
    def calculate_mass_reduction(k_SM, component):
        """
        Calculate the mass reduction percentage for given k_SM
        
        Parameters:
        -----------
        k_SM : float
            Structural mass factor
        component : str
            Component name
            
        Returns:
        --------
        float : Mass reduction percentage (0-100)
        """
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        mass_reduction = (1 - k_SM/k_Al) * 100
        return mass_reduction
    
    @staticmethod
    def optimal_k_SM_for_lca(lca_al, lca_composite, component):
        """
        Calculate optimal k_SM to minimize LCA impact
        Simple linear optimization assuming linear LCA relationship
        
        Parameters:
        -----------
        lca_al : float
            LCA impact per kg of aluminum
        lca_composite : float
            LCA impact per kg of composite
        component : str
            Component name
            
        Returns:
        --------
        float : Optimal k_SM value
        """
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # If composite has lower impact per effective kg
        if lca_composite * k_Comp < lca_al * k_Al:
            return k_Comp  # Use 100% composite
        else:
            return k_Al  # Use 100% aluminum
    
    @staticmethod
    def print_component_info(component):
        """
        Print information about material efficiency for a component
        """
        if component not in MaterialConverter.K_SM_RANGES:
            print(f"Unknown component: {component}")
            return
        
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        mass_reduction = (1 - k_Comp/k_Al) * 100
        
        print(f"\n{component.upper()} Material Properties:")
        print(f"  Aluminum k_SM: {k_Al}")
        print(f"  Composite k_SM: {k_Comp}")
        print(f"  Mass reduction with 100% composite: {mass_reduction:.1f}%")
        print(f"  Valid k_SM range: [{k_Comp}, {k_Al}]")


def validate_k_SM_design(k_SM_thrust_frame, k_SM_interstage):
    """
    Validate k_SM values for both variable components
    
    Parameters:
    -----------
    k_SM_thrust_frame : float
        k_SM value for thrust frame
    k_SM_interstage : float
        k_SM value for interstage
        
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    # Check thrust frame bounds
    tf_bounds = MaterialConverter.get_bounds('thrust_frame')
    if not (tf_bounds[0] <= k_SM_thrust_frame <= tf_bounds[1]):
        return False
    
    # Check interstage bounds
    is_bounds = MaterialConverter.get_bounds('interstage')
    if not (is_bounds[0] <= k_SM_interstage <= is_bounds[1]):
        return False
    
    return True


def calculate_environmental_tradeoff(base_mass, k_SM, component, 
                                    impact_al=8.5, impact_composite=25.0):
    """
    Calculate the environmental impact trade-off for a component
    
    Parameters:
    -----------
    base_mass : float
        Mass if component were 100% aluminum with k_SM = 1.0
    k_SM : float
        Current structural mass factor
    component : str
        Component name
    impact_al : float
        Environmental impact per kg aluminum (default: kg CO2-eq)
    impact_composite : float
        Environmental impact per kg composite (default: kg CO2-eq)
        
    Returns:
    --------
    dict : Environmental metrics
    """
    # Get actual mass with current k_SM
    actual_mass = base_mass * k_SM
    
    # Get material fractions
    al_frac, comp_frac = MaterialConverter.k_SM_to_fractions(k_SM, component)
    
    # Calculate impacts
    al_mass = actual_mass * al_frac
    comp_mass = actual_mass * comp_frac
    
    total_impact = al_mass * impact_al + comp_mass * impact_composite
    
    # Compare with 100% aluminum case
    al_only_mass = base_mass * MaterialConverter.K_SM_RANGES[component]['Al']
    al_only_impact = al_only_mass * impact_al
    
    # Compare with 100% composite case
    comp_only_mass = base_mass * MaterialConverter.K_SM_RANGES[component]['Composite']
    comp_only_impact = comp_only_mass * impact_composite
    
    return {
        'actual_mass': actual_mass,
        'al_mass': al_mass,
        'comp_mass': comp_mass,
        'total_impact': total_impact,
        'al_only_impact': al_only_impact,
        'comp_only_impact': comp_only_impact,
        'mass_saved_vs_al': al_only_mass - actual_mass,
        'impact_vs_al': total_impact - al_only_impact,
        'impact_vs_comp': total_impact - comp_only_impact
    }


# Example usage and testing
if __name__ == "__main__":
    
    print("="*60)
    print("FELIN LAUNCHER MATERIAL CONVERSION UTILITIES")
    print("="*60)
    
    # Show component information
    for component in ['thrust_frame', 'interstage']:
        MaterialConverter.print_component_info(component)
    
    print("\n" + "="*60)
    print("CONVERSION EXAMPLES")
    print("="*60)
    
    # Example 1: Different k_SM for same material mix
    print("\nExample: 50% Al, 50% Composite material mix")
    
    # Thrust frame
    k_SM_tf = MaterialConverter.fractions_to_k_SM(0.5, 'thrust_frame')
    print(f"Thrust frame: k_SM = {k_SM_tf:.3f}")
    
    # Interstage  
    k_SM_is = MaterialConverter.fractions_to_k_SM(0.5, 'interstage')
    print(f"Interstage: k_SM = {k_SM_is:.3f}")
    print("Note: Same material mix gives different k_SM values!")
    
    # Example 2: Mass reduction
    print("\nMass reduction for different k_SM values:")
    test_k_values = [1.0, 0.9, 0.8, 0.7, 0.62]
    for k in test_k_values:
        if k >= 0.62:  # Valid for thrust frame
            reduction = MaterialConverter.calculate_mass_reduction(k, 'thrust_frame')
            al_frac, _ = MaterialConverter.k_SM_to_fractions(k, 'thrust_frame')
            print(f"  Thrust frame k_SM={k:.2f}: {reduction:.1f}% reduction, "
                  f"{al_frac*100:.0f}% Al")
    
    # Example 3: Environmental trade-off
    print("\n" + "="*60)
    print("ENVIRONMENTAL TRADE-OFF ANALYSIS")
    print("="*60)
    
    base_mass = 1000  # kg
    
    # Analyze thrust frame
    print(f"\nThrust Frame (base mass = {base_mass} kg):")
    for al_percent in [100, 75, 50, 25, 0]:
        al_frac = al_percent / 100
        k_SM = MaterialConverter.fractions_to_k_SM(al_frac, 'thrust_frame')
        metrics = calculate_environmental_tradeoff(base_mass, k_SM, 'thrust_frame')
        
        print(f"  {al_percent}% Al: mass={metrics['actual_mass']:.0f} kg, "
              f"CO2={metrics['total_impact']:.0f} kg")
    
    # Find break-even point
    print("\n" + "="*60)
    print("OPTIMIZATION INSIGHTS")
    print("="*60)
    
    # When is composite better than aluminum?
    impact_al = 8.5  # kg CO2/kg
    impact_comp = 25.0  # kg CO2/kg
    
    print(f"\nWith Al impact={impact_al} and Composite impact={impact_comp}:")
    
    for component in ['thrust_frame', 'interstage']:
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # Break-even point where impacts are equal
        # impact_al * mass_al = impact_comp * mass_comp
        # impact_al * k_Al = impact_comp * k_Comp (for same base mass)
        
        ratio = (impact_al * k_Al) / (impact_comp * k_Comp)
        
        if ratio > 1:
            print(f"  {component}: Composite is better (ratio={ratio:.2f})")
        elif ratio < 1:
            print(f"  {component}: Aluminum is better (ratio={ratio:.2f})")
        else:
            print(f"  {component}: Equal impact")
        
        # Critical impact ratio where they break even
        critical_ratio = k_Al / k_Comp
        critical_impact_comp = impact_al * critical_ratio
        print(f"    Break-even when composite impact < {critical_impact_comp:.1f} kg CO2/kg")