"""
Simplified SLSQP Optimization for FELIN-LCA
Minimizes trajectory failures and maximizes performance
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import openmdao.api as om
import Launch_vehicle_Group
import result_vizualization

# Cache to avoid redundant evaluations
evaluation_cache = {}
eval_count = 0
best_esa = np.inf
best_design = None

def evaluate_design(x, P_obj):
    """
    Evaluate a design and return all needed values
    Caches results to avoid redundant evaluations
    """
    global eval_count, evaluation_cache, best_esa, best_design
    
    # Check cache first
    x_key = tuple(np.round(x, 6))  # Round for numerical tolerance
    if x_key in evaluation_cache:
        return evaluation_cache[x_key]
    
    eval_count += 1
    
    # Setup problem if needed
    if not hasattr(evaluate_design, 'setup_done'):
        P_obj.setup(check=False)
        evaluate_design.setup_done = True
    
    # Set all design variables
    P_obj['Prop_mass_stage_1'] = x[0] * 1e3
    P_obj['Prop_mass_stage_2'] = x[1] * 1e3
    P_obj['OF_stage_1'] = x[2]
    P_obj['OF_stage_2'] = x[3]
    P_obj['stage1_cfrp_fraction'] = x[4]
    P_obj['stage1_aluminum_fraction'] = x[5]
    P_obj['stage1_steel_fraction'] = 1.0 - x[4] - x[5]
    P_obj['stage2_cfrp_fraction'] = x[6]
    P_obj['stage2_aluminum_fraction'] = x[7]
    P_obj['stage2_steel_fraction'] = 1.0 - x[6] - x[7]
    P_obj['fairing_cfrp_fraction'] = x[8]
    P_obj['thetacmd_i'] = x[9]
    P_obj['thetacmd_f'] = x[10]
    P_obj['ksi'] = x[11]
    P_obj['Pitch_over_duration'] = x[12]
    P_obj['Delta_vertical_phase'] = x[13]
    P_obj['Delta_theta_pitch_over'] = x[14]
    P_obj['command_stage_1_exo'] = x[15:17]
    
    # Fixed parameters
    P_obj['Diameter_stage_1'] = 5.0
    P_obj['Diameter_stage_2'] = 5.0
    P_obj['Mass_flow_rate_stage_1'] = 250.
    P_obj['Mass_flow_rate_stage_2'] = 250.
    P_obj['Thrust_stage_1'] = 1000.
    P_obj['Thrust_stage_2'] = 800.
    P_obj['Pc_stage_1'] = 100.0
    P_obj['Pc_stage_2'] = 100.0
    P_obj['Pe_stage_1'] = 1.0
    P_obj['Pe_stage_2'] = 1.0
    P_obj['N_eng_stage_1'] = 8.
    P_obj['N_eng_stage_2'] = 1.
    P_obj['Exit_nozzle_area_stage_1'] = 0.79
    P_obj['Exit_nozzle_area_stage_2'] = 3.6305
    P_obj['is_fallout'] = 0.
    P_obj['payload_mass'] = 5000.0
    
    # Fixed Point Iteration
    error = 100.
    P_obj['Pdyn_max_dim'] = 36.24
    k = 0
    
    while error > 1. and k < 15:
        try:
            P_obj.run_model()
            new_pdyn = P_obj['max_pdyn_load_ascent_stage_1']/1e3
            error = abs(P_obj['Pdyn_max_dim'] - new_pdyn)
            P_obj['Pdyn_max_dim'] += 0.5 * (new_pdyn - P_obj['Pdyn_max_dim'])
            k += 1
        except:
            # FPI failed
            result = {
                'converged': False,
                'esa': 1e6,
                'altitude': -1e6,
                'delta_v': -1e6,
                'max_g': 10.0
            }
            evaluation_cache[x_key] = result
            return result
    
    if error > 1.:
        # FPI didn't converge
        result = {
            'converged': False,
            'esa': 1e6,
            'altitude': -1e6,
            'delta_v': -1e6,
            'max_g': 10.0
        }
        evaluation_cache[x_key] = result
        return result
    
    # Extract results
    try:
        nb_pts = int(P_obj['Nb_pt_ascent'][0])
        if nb_pts > 0:
            esa = P_obj['ESA_single_score'][0]
            altitude = P_obj['alt_ascent'][nb_pts-1]
            delta_v = P_obj['V_ascent'][nb_pts-1] - P_obj['V_ascent'][0]
            max_g = P_obj['max_acceleration_g'][0]
            
            # Track best feasible
            if altitude >= 300000 and delta_v >= 7400 and esa < best_esa:
                best_esa = esa
                best_design = x.copy()
                print(f"Eval {eval_count}: NEW BEST ESA = {esa:.1f} (alt={altitude/1000:.1f}km, ΔV={delta_v:.1f}m/s)")
            elif eval_count % 20 == 0:
                print(f"Eval {eval_count}: ESA = {esa:.1f} (alt={altitude/1000:.1f}km, ΔV={delta_v:.1f}m/s)")
            
            result = {
                'converged': True,
                'esa': esa,
                'altitude': altitude,
                'delta_v': delta_v,
                'max_g': max_g,
                'P_obj': P_obj
            }
        else:
            result = {
                'converged': False,
                'esa': 1e6,
                'altitude': -1e6,
                'delta_v': -1e6,
                'max_g': 10.0
            }
    except:
        result = {
            'converged': False,
            'esa': 1e6,
            'altitude': -1e6,
            'delta_v': -1e6,
            'max_g': 10.0
        }
    
    evaluation_cache[x_key] = result
    return result

def objective(x):
    """Simple objective: just ESA score"""
    result = evaluate_design(x, P_obj)
    return result['esa']

def all_constraints(x):
    """
    All constraints in one function (more efficient)
    Returns array where each element must be >= 0
    """
    result = evaluate_design(x, P_obj)
    
    return np.array([
        result['altitude'] - 300000,    # Altitude >= 300 km
        result['delta_v'] - 7400,        # Delta-v >= 7400 m/s
        4.5 - result['max_g'],           # Max g <= 4.5
        0.95 - (x[4] + x[5]),           # Stage 1 materials <= 95%
        0.95 - (x[6] + x[7]),           # Stage 2 materials <= 95%
        x[1]/x[0] - 0.15,               # Prop ratio >= 0.15
        0.30 - x[1]/x[0]                # Prop ratio <= 0.30
    ])

# ============================================================
# MAIN OPTIMIZATION
# ============================================================

print("="*60)
print("SIMPLIFIED SLSQP OPTIMIZATION FOR FELIN-LCA")
print("="*60)

# Create problem
P_obj = om.Problem()
P_obj.model = Launch_vehicle_Group.Launcher_vehicle()

# TIGHTER BOUNDS to avoid infeasible regions
bounds = [
    (285, 310),    # Prop_mass_stage_1 - narrower range
    (52, 62),      # Prop_mass_stage_2 - narrower range
    (5.1, 5.3),    # OF_stage_1 - very tight
    (5.1, 5.3),    # OF_stage_2 - very tight
    (0.08, 0.20),  # stage1_cfrp_fraction - realistic
    (0.50, 0.65),  # stage1_aluminum_fraction - traditional
    (0.10, 0.22),  # stage2_cfrp_fraction - realistic
    (0.50, 0.65),  # stage2_aluminum_fraction - traditional
    (0.40, 0.60),  # fairing_cfrp_fraction - moderate
    (1.95, 2.05),  # thetacmd_i - tight
    (9.95, 10.05), # thetacmd_f - tight
    (0.29, 0.30),  # ksi - very tight
    (9.9, 10.1),   # Pitch_over_duration - tight
    (9.9, 10.1),   # Delta_vertical_phase - tight
    (1.95, 2.05),  # Delta_theta_pitch_over - tight
    (19, 21),      # command_stage_1_exo[0] - tight
    (-11, -9)      # command_stage_1_exo[1] - tight
]

# GOOD INITIAL GUESS (from previous successful runs)
x0 = np.array([
    295., 56.,      # Propellant masses that work
    5.2, 5.2,       # Standard O/F
    0.12, 0.60,     # Stage 1: traditional
    0.15, 0.60,     # Stage 2: traditional
    0.50,           # Fairing: moderate CFRP
    2.0, 10., 0.295,  # Known good trajectory
    10., 10., 2.,     # Known good pitch
    20., -10.         # Known good exo
])

print("\nInitial guess (proven baseline):")
print(f"  Stage 1: {x0[4]:.1%} CFRP, {x0[5]:.1%} Al, {(1-x0[4]-x0[5]):.1%} Steel")
print(f"  Stage 2: {x0[6]:.1%} CFRP, {x0[7]:.1%} Al, {(1-x0[6]-x0[7]):.1%} Steel")

# Test initial point
print("\nTesting initial design...")
initial_result = evaluate_design(x0, P_obj)
print(f"Initial: ESA={initial_result['esa']:.1f}, Alt={initial_result['altitude']/1000:.1f}km, ΔV={initial_result['delta_v']:.1f}m/s")

# OPTIMIZATION with conservative settings
print("\n" + "="*60)
print("STARTING OPTIMIZATION")
print("="*60)

result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints={'type': 'ineq', 'fun': all_constraints},
    options={
        'maxiter': 100,        # Fewer iterations
        'ftol': 1e-5,          # Looser tolerance
        'disp': True,
        'finite_diff_rel_step': 1e-4  # Smaller step for gradients
    }
)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

print(f"\nTotal evaluations: {eval_count}")
print(f"Cache hits: {len(evaluation_cache)}")
print(f"Success: {result.success}")
print(f"Message: {result.message}")

# FINAL RESULTS
if best_design is not None:
    print("\n=== BEST FEASIBLE DESIGN FOUND ===")
    final_result = evaluate_design(best_design, P_obj)
    
    print(f"ESA Score: {final_result['esa']:.1f}")
    print(f"Altitude: {final_result['altitude']/1000:.1f} km")
    print(f"Delta-V: {final_result['delta_v']:.1f} m/s")
    print(f"Max g: {final_result['max_g']:.2f}")
    
    print(f"\nMaterial distribution:")
    print(f"  Stage 1: {best_design[4]:.1%} CFRP, {best_design[5]:.1%} Al, {(1-best_design[4]-best_design[5]):.1%} Steel")
    print(f"  Stage 2: {best_design[6]:.1%} CFRP, {best_design[7]:.1%} Al, {(1-best_design[6]-best_design[7]):.1%} Steel")
    print(f"  Fairing: {best_design[8]:.1%} CFRP")
    
    # Get final P_obj for plotting
    if 'P_obj' in final_result:
        P_opt = final_result['P_obj']
        
        # ESA breakdown
        print("\n=== ESA BREAKDOWN ===")
        print(f"ADEPLmu: {P_opt['ESA_ADEPLmu_normalized'][0]:.0f} (dominates)")
        print(f"GWP: {P_opt['GWP_total'][0]/1e3:.1f} tCO2eq")
        
        # Generate plots
        try:
            result_vizualization.plots_output(P_opt)
            print("\nTrajectory plots generated")
        except:
            pass
        
        # Save results
        pd.DataFrame([{
            'ESA': final_result['esa'],
            'Altitude_km': final_result['altitude']/1000,
            'DeltaV_ms': final_result['delta_v'],
            'CFRP_s1': best_design[4],
            'CFRP_s2': best_design[6]
        }]).to_csv("slsqp_results.csv", index=False)
        print("Results saved to slsqp_results.csv")
else:
    print("\nNo feasible solution found")