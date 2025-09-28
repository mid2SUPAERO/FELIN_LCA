"""
Material conversion helper functions for FELIN launcher optimization
Complete version with all components and LCA integration
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
            'Composite': 0.62,  # 38% mass reduction with composite
            'name': 'Thrust Frame'
        },
        'interstage': {
            'Al': 1.0,
            'Composite': 0.70,  # 30% mass reduction with composite
            'name': 'Interstage'
        },
        'intertank': {
            'Al': 1.0,
            'Composite': 0.80,  # 20% mass reduction with composite
            'name': 'Intertank'
        },
        'stage_2': {
            'Al': 1.0,
            'Composite': 0.75,  # 25% mass reduction with composite
            'name': 'Stage 2 Structure'
        }
    }
    
    # Environmental impact factors (kg CO2-eq per kg material)
    # These can be updated based on specific LCA database values
    IMPACT_FACTORS = {
        'aluminum_primary': 8.5,       # Primary aluminum
        'aluminum_recycled': 1.5,      # Recycled aluminum (if using)
        'cfrp_virgin': 25.0,           # Virgin carbon fiber composite
        'cfrp_recycled': 12.0,         # Recycled CFRP (if available)
        'steel': 2.3,                  # Steel
        'titanium': 35.0,              # Titanium alloy
        # Operational factors
        'fuel_multiplier': 8.0,        # kg fuel saved per kg structure over mission
        'lh2_impact': 9.0,             # kg CO2 per kg H2 (grey hydrogen)
        'lox_impact': 0.2,             # kg CO2 per kg O2
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
            Component name ('thrust_frame', 'interstage', 'intertank', 'stage_2')
            
        Returns:
        --------
        tuple : (Al_fraction, Composite_fraction)
            Mass fractions that sum to 1.0
        """
        if component not in MaterialConverter.K_SM_RANGES:
            raise ValueError(f"Unknown component: {component}")
        
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # Ensure k_SM is within valid range (with small tolerance for numerical errors)
        k_SM = np.clip(k_SM, k_Comp - 0.001, k_Al + 0.001)
        
        # Linear interpolation
        Al_fraction = (k_SM - k_Comp) / (k_Al - k_Comp) if k_Al != k_Comp else 1.0
        Al_fraction = np.clip(Al_fraction, 0.0, 1.0)
        Composite_fraction = 1.0 - Al_fraction
        
        return float(Al_fraction), float(Composite_fraction)
    
    @classmethod
    def reset_baseline(cls):
        """Reset baseline for new optimization run"""
        cls._baseline_dry_mass = None
    
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
        
        Al_fraction = np.clip(Al_fraction, 0.0, 1.0)
        
        k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
        k_Comp = MaterialConverter.K_SM_RANGES[component]['Composite']
        
        # Linear interpolation
        k_SM = Al_fraction * k_Al + (1 - Al_fraction) * k_Comp
        
        return float(k_SM)
    
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
        return max(0, mass_reduction)
    
    @staticmethod
    def calculate_lca_impact(mass, k_SM, component, include_operational=True):
        """
        Calculate LCA impact for a component with given mass and material mix
        
        Parameters:
        -----------
        mass : float
            Component mass (kg) with current k_SM
        k_SM : float
            Structural mass factor
        component : str
            Component name
        include_operational : bool
            Whether to include operational benefits from mass savings
            
        Returns:
        --------
        dict : LCA impact breakdown
        """
        # Get material fractions
        al_frac, comp_frac = MaterialConverter.k_SM_to_fractions(k_SM, component)
        
        # Calculate material masses
        al_mass = mass * al_frac
        comp_mass = mass * comp_frac
        
        # Manufacturing impacts
        impact_al = MaterialConverter.IMPACT_FACTORS['aluminum_primary']
        impact_comp = MaterialConverter.IMPACT_FACTORS['cfrp_virgin']
        
        manufacturing_impact = al_mass * impact_al + comp_mass * impact_comp
        
        # Operational benefit (if applicable)
        operational_benefit = 0
        if include_operational:
            # Compare with all-aluminum baseline
            k_Al = MaterialConverter.K_SM_RANGES[component]['Al']
            baseline_mass = mass / k_SM * k_Al  # What mass would be with 100% Al
            mass_saved = baseline_mass - mass
            
            # Fuel savings over mission lifetime
            fuel_saved = mass_saved * MaterialConverter.IMPACT_FACTORS['fuel_multiplier']
            operational_benefit = fuel_saved * MaterialConverter.IMPACT_FACTORS['lh2_impact']
        
        # Net impact
        net_impact = manufacturing_impact - operational_benefit
        
        return {
            'manufacturing_impact': manufacturing_impact,
            'operational_benefit': operational_benefit,
            'net_impact': net_impact,
            'al_mass': al_mass,
            'comp_mass': comp_mass,
            'mass_savings': baseline_mass - mass if include_operational else 0
        }
    
    @staticmethod
    def find_optimal_k_SM(base_mass, component, include_operational=True):
        """
        Find optimal k_SM to minimize LCA impact for a component
        
        Parameters:
        -----------
        base_mass : float
            Mass if component were 100% aluminum with k_SM = 1.0
        component : str
            Component name
        include_operational : bool
            Whether to include operational benefits
            
        Returns:
        --------
        dict : Optimal k_SM and related metrics
        """
        bounds = MaterialConverter.get_bounds(component)
        
        # Test range of k_SM values
        k_SM_values = np.linspace(bounds[0], bounds[1], 100)
        impacts = []
        
        for k_SM in k_SM_values:
            mass = base_mass * k_SM
            result = MaterialConverter.calculate_lca_impact(
                mass, k_SM, component, include_operational
            )
            impacts.append(result['net_impact'])
        
        # Find minimum
        min_idx = np.argmin(impacts)
        optimal_k_SM = k_SM_values[min_idx]
        optimal_impact = impacts[min_idx]
        
        # Get fractions at optimum
        al_frac, comp_frac = MaterialConverter.k_SM_to_fractions(optimal_k_SM, component)
        
        return {
            'optimal_k_SM': optimal_k_SM,
            'optimal_impact': optimal_impact,
            'al_fraction': al_frac,
            'comp_fraction': comp_frac,
            'all_k_SM': k_SM_values,
            'all_impacts': impacts
        }
    
    @staticmethod
    def validate_design(k_SM_dict):
        """
        Validate a complete design (all k_SM values)
        
        Parameters:
        -----------
        k_SM_dict : dict
            Dictionary with component names as keys and k_SM values as values
            
        Returns:
        --------
        tuple : (is_valid, error_messages)
        """
        errors = []
        
        for component, k_SM in k_SM_dict.items():
            if component not in MaterialConverter.K_SM_RANGES:
                errors.append(f"Unknown component: {component}")
                continue
                
            bounds = MaterialConverter.get_bounds(component)
            if not (bounds[0] <= k_SM <= bounds[1]):
                errors.append(f"{component}: k_SM={k_SM:.3f} outside bounds {bounds}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def print_design_summary(k_SM_dict, base_masses=None):
        """
        Print a summary of the design configuration
        
        Parameters:
        -----------
        k_SM_dict : dict
            Dictionary with component names as keys and k_SM values as values
        base_masses : dict, optional
            Base masses for each component (100% Al case)
        """
        print("\n" + "="*60)
        print("DESIGN CONFIGURATION SUMMARY")
        print("="*60)
        
        total_manufacturing = 0
        total_operational = 0
        total_mass = 0
        total_base_mass = 0
        
        for component, k_SM in k_SM_dict.items():
            if component not in MaterialConverter.K_SM_RANGES:
                continue
                
            name = MaterialConverter.K_SM_RANGES[component]['name']
            al_frac, comp_frac = MaterialConverter.k_SM_to_fractions(k_SM, component)
            mass_reduction = MaterialConverter.calculate_mass_reduction(k_SM, component)
            
            print(f"\n{name}:")
            print(f"  k_SM = {k_SM:.3f}")
            print(f"  Material mix: {al_frac*100:.1f}% Al, {comp_frac*100:.1f}% Composite")
            print(f"  Mass reduction: {mass_reduction:.1f}%")
            
            if base_masses and component in base_masses:
                base_mass = base_masses[component]
                actual_mass = base_mass * k_SM
                result = MaterialConverter.calculate_lca_impact(
                    actual_mass, k_SM, component, include_operational=True
                )
                
                print(f"  Mass: {actual_mass:.1f} kg (saved {base_mass-actual_mass:.1f} kg)")
                print(f"  Manufacturing impact: {result['manufacturing_impact']:.1f} kg CO2-eq")
                print(f"  Operational benefit: -{result['operational_benefit']:.1f} kg CO2-eq")
                print(f"  Net impact: {result['net_impact']:.1f} kg CO2-eq")
                
                total_manufacturing += result['manufacturing_impact']
                total_operational += result['operational_benefit']
                total_mass += actual_mass
                total_base_mass += base_mass
        
        if base_masses:
            print("\n" + "-"*60)
            print("TOTAL IMPACTS:")
            print(f"  Total mass: {total_mass:.1f} kg (saved {total_base_mass-total_mass:.1f} kg)")
            print(f"  Total manufacturing: {total_manufacturing:.1f} kg CO2-eq")
            print(f"  Total operational benefit: -{total_operational:.1f} kg CO2-eq")
            print(f"  Net impact: {total_manufacturing - total_operational:.1f} kg CO2-eq")


# Utility functions for optimization
def create_bounds_arrays():
    """
    Create bounds arrays for optimization algorithms
    
    Returns:
    --------
    tuple : (lower_bounds, upper_bounds) as numpy arrays
    """
    components = ['thrust_frame', 'interstage', 'intertank', 'stage_2']
    lower = []
    upper = []
    
    for comp in components:
        bounds = MaterialConverter.get_bounds(comp)
        lower.append(bounds[0])
        upper.append(bounds[1])
    
    return np.array(lower), np.array(upper)


def normalize_k_SM(k_SM_values):
    """
    Normalize k_SM values to [0, 1] range for optimization
    
    Parameters:
    -----------
    k_SM_values : array-like
        k_SM values in actual range
        
    Returns:
    --------
    numpy array : Normalized values
    """
    components = ['thrust_frame', 'interstage', 'intertank', 'stage_2']
    normalized = []
    
    for i, comp in enumerate(components):
        bounds = MaterialConverter.get_bounds(comp)
        norm_val = (k_SM_values[i] - bounds[0]) / (bounds[1] - bounds[0])
        normalized.append(norm_val)
    
    return np.array(normalized)


def denormalize_k_SM(normalized_values):
    """
    Convert normalized [0, 1] values back to actual k_SM range
    
    Parameters:
    -----------
    normalized_values : array-like
        Normalized values [0, 1]
        
    Returns:
    --------
    numpy array : Actual k_SM values
    """
    components = ['thrust_frame', 'interstage', 'intertank', 'stage_2']
    actual = []
    
    for i, comp in enumerate(components):
        bounds = MaterialConverter.get_bounds(comp)
        k_SM = bounds[0] + normalized_values[i] * (bounds[1] - bounds[0])
        actual.append(k_SM)
    
    return np.array(actual)


# Example usage and testing
if __name__ == "__main__":
    
    print("="*60)
    print("FELIN LAUNCHER MATERIAL OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Show all component information
    for component in MaterialConverter.K_SM_RANGES.keys():
        bounds = MaterialConverter.get_bounds(component)
        print(f"\n{MaterialConverter.K_SM_RANGES[component]['name']}:")
        print(f"  k_SM range: [{bounds[0]:.2f}, {bounds[1]:.2f}]")
        print(f"  Max mass reduction: {(1-bounds[0])*100:.0f}%")
    
    # Test design configurations
    print("\n" + "="*60)
    print("DESIGN CONFIGURATIONS")
    print("="*60)
    
    # Define test configurations
    configs = {
        '100% Aluminum': {'thrust_frame': 1.0, 'interstage': 1.0, 
                         'intertank': 1.0, 'stage_2': 1.0},
        '100% Composite': {'thrust_frame': 0.62, 'interstage': 0.70, 
                          'intertank': 0.80, 'stage_2': 0.75},
        '50/50 Mix': {'thrust_frame': 0.81, 'interstage': 0.85, 
                     'intertank': 0.90, 'stage_2': 0.875},
    }
    
    # Example base masses (kg)
    base_masses = {
        'thrust_frame': 2000,
        'interstage': 1500,
        'intertank': 1000,
        'stage_2': 3000  # structural portion only
    }
    
    # Analyze each configuration
    for name, k_SM_dict in configs.items():
        print(f"\n### {name} ###")
        MaterialConverter.print_design_summary(k_SM_dict, base_masses)
    
    # Find optimal configuration
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS")
    print("="*60)
    
    for component, base_mass in base_masses.items():
        print(f"\nOptimizing {MaterialConverter.K_SM_RANGES[component]['name']}:")
        result = MaterialConverter.find_optimal_k_SM(base_mass, component, 
                                                    include_operational=True)
        print(f"  Optimal k_SM: {result['optimal_k_SM']:.3f}")
        print(f"  Material mix: {result['al_fraction']*100:.0f}% Al, "
              f"{result['comp_fraction']*100:.0f}% Composite")
        print(f"  Optimal impact: {result['optimal_impact']:.1f} kg CO2-eq")
    
    # Test normalization functions
    print("\n" + "="*60)
    print("NORMALIZATION TEST")
    print("="*60)
    
    test_k_SM = np.array([0.81, 0.85, 0.90, 0.875])
    print(f"Original k_SM: {test_k_SM}")
    
    normalized = normalize_k_SM(test_k_SM)
    print(f"Normalized: {normalized}")
    
    denormalized = denormalize_k_SM(normalized)
    print(f"Denormalized: {denormalized}")
    
    print(f"Match: {np.allclose(test_k_SM, denormalized)}")