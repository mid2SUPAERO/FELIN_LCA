# FELIN_LCA

### Prerequisites

Two python 3.8.8 packages are required: OpenMDAO 3.13.1 and CMA 3.1.0 

```
pip install openmdao
pip install cma
```

The interesting files created/updated are Launcher_Design_Problem.ipynb, where the optimization is performed, Launch_Vehicle_Group.py where all 4 disciplines are coupled, and the files inside Structure, Propulsion and Environment Folders, the modified files Dry_mass_stage_1.py, Mass_models.py, Propulsion.py, Environmental_Disicipline.py and Material_helpers.py.

# Ecoinvent setup
```
import brightway2 as bw

bw.projects.set_current("Project_name")
bw.bw2setup()

fp = r'directory'
if bw.Database("ecoinvent 3.8 cutoff").random() is None :
    ei = bw.SingleOutputEcospold2Importer(fp, "ecoinvent 3.8 cutoff")
    ei.apply_strategies()
    ei.statistics()
    ei.write_database()
else :
    print("ecoinvent 3.8 cutoff already imported")
eco = bw.Database("ecoinvent 3.8 cutoff")
print(eco.random())
```


### Workflow Overview

1. Design Variable Flow: k_SM (Structural Mass factor) represents material composition:

Each component has specific range:

Thrust frame: [0.62, 1.00]
Interstage: [0.70, 1.00]
Intertank: [0.80, 1.00]
Stage 2: [0.75, 1.00]

k_SM = 1.0 means 100% aluminum, lower bound means 100% composite
Mass calculation: Component_mass = Base_mass × k_SM

2. Structural Mass Calculation Chain
k_SM → Material Fractions → Component Masses → Total Dry Mass
In Dry_Mass_stage_1.py:
```
Convert k_SM to material fractions
Al_fraction = (k_SM - k_SM_min) / (k_SM_max - k_SM_min)
Composite_fraction = 1.0 - Al_fraction

# Calculate component mass with material factor
M_thrust_frame = base_calculation × k_SM_thrust_frame
```
Mass breakdown:

Variable components (affected by k_SM): thrust frame, interstage, intertank, Stage 2 structure
Fixed components (always same material): tanks, engines, avionics, TPS

3. Environmental Impact Calculation
Material Inventory Building:
```
#For each component
material_inventory['aluminum_7075'] += component_mass × Al_fraction
material_inventory['cfrp'] += component_mass × Composite_fraction
```
LCA Impact Assessment using Brightway2:
```
#For each material in inventory
activity = bw.get_activity(ECOINVENT_CODES[material])
bw_inventory[activity] = mass_kg
```
```
# For each ESA impact category
lca = bw.LCA(bw_inventory, method)
lca.lci()  # Life Cycle Inventory
lca.lcia() # Life Cycle Impact Assessment
impact = lca.score
```
ESA Method Processing:

16 impact categories (GWP, acidification, toxicity...)
Each normalized to common units
Weighted by importance factors (took from simapro)
Summed to single score in Points (Pt)

4. Operational Benefit Calculation (mass savings by using cfrp)
```
#Mass savings compared to baseline (100% aluminum)
baseline_mass = component_mass / k_SM  # What it would be at k_SM=1.0
mass_savings = baseline_mass - actual_mass

# Fuel savings over mission lifetime
fuel_savings = mass_savings × 8  # Multiplier from rocket equation

# Environmental benefit from saved propellant production
benefit = LCA(saved_LOX + saved_LH2)
```
5. Complete Integration Loop
CMA-ES Optimizer
    ↓ (proposes k_SM values)
OpenMDAO Problem
    ↓
Structural Discipline
    → Calculates masses based on k_SM
    → Outputs material fractions
    ↓
Environmental Discipline  
    → Builds material inventory
    → Maps ecoinvent entries via Brightway2
    → Calculates manufacturing impacts
    → Calculates operational benefits
    → Returns LCA score
    ↓
Trajectory Discipline
    → Simulates flight with updated masses
    → Returns performance metrics
    ↓
Objective Function
    → Combines LCA score and performance
    → Returns to optimizer
    
6. Balanced Optimization Approach
Single Weighted Objective -> Alpha* mass + (1-alpha)*score
```
Normalize both objectives to [0,1]
lca_normalized = (lca_score - 250) / (285 - 250)
glow_normalized = (glow - 378000) / (382000 - 378000)

# Weighted combination
objective = w_env × lca_normalized + w_perf × glow_normalized
Key Trade-offs:

Aluminum: Lower environmental impact (14 kg CO₂/kg), heavier structure
CFRP: Higher environmental impact (85 kg CO₂/kg), 20-38% mass reduction
Operational benefit: Rarely compensates for CFRP's 6× higher manufacturing impact
```

7. Why Aluminum Dominates
Manufacturing impact ratio: CFRP/Al = 6:1 (much more pollutant - should i find some greener material?)
Mass reduction: Maximum 38% (thrust frame)
Fuel savings multiplier: 8× over mission
BUT Environmental penalty exceeds operational benefit
->The balanced optimization (60% environment, 40% performance) still chose aluminum because the performance gain (0.7% GLOW reduction) is minimal compared to the environmental cost (13% LCA increase).
