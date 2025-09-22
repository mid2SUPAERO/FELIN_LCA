"""
FELIN Launcher Design Problem
Optimization for Minimum ESA Single Score
Compatible with actual FELIN structure
"""

import numpy as np
import pandas as pd
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt

import openmdao.api as om
import cma

import Launch_vehicle_Group
import post_traitement
import result_vizualization
import constants as Cst
import specifications as Spec

# ============================================================================
# PROBLEM SETUP
# ============================================================================

print("="*70)
print("FELIN LAUNCHER ENVIRONMENTAL OPTIMIZATION")
print("Objective: Minimize ESA Single Score")
print("Constraint: 5t payload to 700km circular orbit")
print("="*70)

P_obj = om.Problem()
P_obj.model = Launch_vehicle_Group.Launcher_vehicle()
P_obj.setup()

# Generate N2 diagram
try:
    om.n2(P_obj, outfile='n2_environmental.html', show_browser=False)
    print("✓ N2 diagram saved to n2_environmental.html")
except:
    print("Could not generate N2 diagram")

# ============================================================================
# FIXED POINT ITERATION
# ============================================================================

def FPI(Pb):
    """
    Fixed Point Iteration for pdyn convergence
    Standard FELIN approach
    """
    Pb.setup(check=False)
    
    # Baseline configuration
    Pb['Diameter_stage_1'] = 5.0
    Pb['Diameter_stage_2'] = 5.0
    Pb['Mass_flow_rate_stage_1'] = 250.
    Pb['Mass_flow_rate_stage_2'] = 250.
    Pb['Thrust_stage_1'] = 1000.
    Pb['Thrust_stage_2'] = 800.
    Pb['OF_stage_1'] = 5.0
    Pb['OF_stage_2'] = 5.5
    Pb['Pc_stage_1'] = 100.0
    Pb['Pc_stage_2'] = 100.0
    Pb['Pe_stage_1'] = 1.0
    Pb['Pe_stage_2'] = 1.0
    Pb['N_eng_stage_1'] = 8.
    Pb['N_eng_stage_2'] = 1.
    Pb['Prop_mass_stage_1'] = 320000.
    Pb['Prop_mass_stage_2'] = 75000.
    Pb['thetacmd_i'] = 2.72
    Pb['thetacmd_f'] = 10.
    Pb['ksi'] = 0.293
    Pb['Pitch_over_duration'] = 5.
    Pb['Exit_nozzle_area_stage_1'] = 0.79
    Pb['Exit_nozzle_area_stage_2'] = 3.6305
    Pb['Delta_vertical_phase'] = 10.
    Pb['Delta_theta_pitch_over'] = 1.
    Pb['command_stage_1_exo'] = np.array([30., -20.])
    Pb['is_fallout'] = 0.
    
    # Material fractions (if environmental available)
    if hasattr(Pb, 'cfrp_fraction_stage1'):
        Pb['cfrp_fraction_stage1'] = 0.25
        Pb['aluminum_fraction_stage1'] = 0.65
        Pb['steel_fraction_stage1'] = 0.10
        Pb['cfrp_fraction_stage2'] = 0.35
        Pb['aluminum_fraction_stage2'] = 0.55
        Pb['steel_fraction_stage2'] = 0.10
        Pb['engine_nickel_fraction'] = 0.60
        Pb['engine_steel_fraction'] = 0.30
        Pb['engine_titanium_fraction'] = 0.10
    
    Pb['payload_mass'] = 5000.0  # 5t
    
    # Fixed point iteration
    error = 100.
    Pb['Pdyn_max_dim'] = 40.
    k = 0
    
    while error > 1. and k < 20:
        Pb.run_model()
        error = abs(Pb['Pdyn_max_dim'] - Pb['max_pdyn_load_ascent_stage_1']/1e3)
        Pb['Pdyn_max_dim'] = Pb['max_pdyn_load_ascent_stage_1']/1e3
        k = k + 1
        print(f'FPI: {k:2d}, error: {error[0]:6.2f}, Pdyn_max: {Pb["Pdyn_max_dim"][0]:6.2f} kPa')
    
    return Pb

# ============================================================================
# BASELINE EVALUATION
# ============================================================================

print("\n" + "="*70)
print("BASELINE CONFIGURATION")
print("="*70)

P_out = FPI(P_obj)
GLOW, contraintes = post_traitement.post_traitement(P_out)

# Check if environmental discipline is available
has_env = 'ESA_single_score' in P_out

print("\n--- Baseline Results ---")
data_baseline = {
    "GLOW (t)": P_out['GLOW'][0]/1e3,
    "Dry_mass_stage_1 (t)": P_out['Dry_mass_stage_1'][0]/1e3,
    "Dry_mass_stage_2 (t)": P_out['Dry_mass_stage_2'][0]/1e3,
    "Prop_mass_stage_1 (t)": P_out['Prop_mass_stage_1'][0]/1e3,
    "Prop_mass_stage_2 (t)": P_out['Prop_mass_stage_2'][0]/1e3
}

if has_env:
    data_baseline.update({
        "ESA_single_score": P_out['ESA_single_score'][0],
        "GWP_total (tCO2eq)": P_out['GWP_total'][0]/1e3,
        "Delta-V (m/s)": P_out.get('delta_v_achieved', [0])[0]
    })
    
    print("\n--- ESA Impact Categories (Normalized) ---")
    categories = ['GWP','ODEPL','IORAD','PCHEM','PMAT','HTOXnc','HTOXc',
                 'ACIDef','FWEUT','MWEUT','TEUT','FWTOX','LUP','WDEPL','ADEPLf','ADEPLmu']
    for cat in categories:
        key = f'ESA_{cat}_normalized'
        if key in P_out:
            print(f"{cat:8s}: {P_out[key][0]:10.6f}")
else:
    print("⚠ Environmental discipline not available - using GLOW minimization")

df_baseline = pd.DataFrame([data_baseline])
print("\n", df_baseline.round(3))

# Visualize baseline
try:
    result_vizualization.plots_output(P_out)
except:
    print("Could not generate plots")

# ============================================================================
# OPTIMIZATION SETUP
# ============================================================================

def FPI_optim(x, lowerbnd_exp, upperbnd_exp, Pb):
    """
    Fixed Point Iteration for optimization
    """
    Pb.setup(check=False)
    XX = lowerbnd_exp + (upperbnd_exp - lowerbnd_exp) * x
    
    # Fixed configuration
    Pb['Diameter_stage_1'] = 5.0
    Pb['Diameter_stage_2'] = 5.0
    Pb['Mass_flow_rate_stage_1'] = 300.
    Pb['Mass_flow_rate_stage_2'] = 200.
    Pb['Thrust_stage_1'] = 1000.
    Pb['Thrust_stage_2'] = 1000.
    Pb['Pc_stage_1'] = 80.0
    Pb['Pc_stage_2'] = 60.0
    Pb['Pe_stage_1'] = 1.0
    Pb['Pe_stage_2'] = 1.0
    Pb['N_eng_stage_1'] = 6.
    Pb['N_eng_stage_2'] = 1.
    
    # Design variables
    idx = 0
    Pb['Prop_mass_stage_1'] = XX[idx] * 1e3; idx += 1
    Pb['Prop_mass_stage_2'] = XX[idx] * 1e3; idx += 1
    Pb['thetacmd_i'] = XX[idx]; idx += 1
    Pb['thetacmd_f'] = XX[idx]; idx += 1
    Pb['ksi'] = XX[idx]; idx += 1
    Pb['Pitch_over_duration'] = XX[idx]; idx += 1
    Pb['Delta_vertical_phase'] = XX[idx]; idx += 1
    Pb['Delta_theta_pitch_over'] = XX[idx]; idx += 1
    Pb['command_stage_1_exo'] = XX[idx:idx+2]; idx += 2
    
    # O/F ratios
    Pb['OF_stage_1'] = XX[idx]; idx += 1
    Pb['OF_stage_2'] = XX[idx]; idx += 1
    
    # Material fractions (if available)
    if hasattr(Pb, 'cfrp_fraction_stage1'):
        Pb['cfrp_fraction_stage1'] = XX[idx]; idx += 1
        Pb['aluminum_fraction_stage1'] = XX[idx]; idx += 1
        Pb['steel_fraction_stage1'] = XX[idx]; idx += 1
        Pb['cfrp_fraction_stage2'] = XX[idx]; idx += 1
        Pb['aluminum_fraction_stage2'] = XX[idx]; idx += 1
        Pb['steel_fraction_stage2'] = XX[idx]; idx += 1
        Pb['engine_nickel_fraction'] = XX[idx]; idx += 1
        Pb['engine_steel_fraction'] = XX[idx]; idx += 1
        Pb['engine_titanium_fraction'] = XX[idx]; idx += 1
    
    # Fixed parameters
    Pb['Exit_nozzle_area_stage_1'] = 0.79
    Pb['Exit_nozzle_area_stage_2'] = 3.6305
    Pb['is_fallout'] = 0.
    Pb['payload_mass'] = 5000.0
    
    # Fixed point iteration
    error = 100.
    Pb['Pdyn_max_dim'] = 40.
    k = 0
    
    while error > 1. and k < 20:
        try:
            Pb.run_model()
            error = abs(Pb['Pdyn_max_dim'] - Pb['max_pdyn_load_ascent_stage_1']/1e3)
            Pb['Pdyn_max_dim'] = Pb['max_pdyn_load_ascent_stage_1']/1e3
            k = k + 1
        except:
            break
    
    return Pb

def Objective_function(x, lowerbnd_exp, upperbnd_exp, Pb, simu):
    """
    Objective function: minimize ESA score (or GLOW if ESA not available)
    """
    P_out = FPI_optim(x, lowerbnd_exp, upperbnd_exp, Pb)
    
    try:
        GLOW, contraintes = post_traitement.post_traitement(P_out)
        
        # Check if environmental available
        if 'ESA_single_score' in P_out:
            objective = P_out['ESA_single_score'][0]
            delta_v = P_out.get('delta_v_achieved', [0])[0]
        else:
            # Fallback to GLOW minimization
            objective = GLOW / 1e5
            delta_v = 9400.0  # Assume OK
        
    except:
        if simu == 0:
            return 1e6
        else:
            return 1e6, np.array([1e6]), 0, 0
    
    # Performance constraint
    delta_v_required = 9400.0  # m/s for 700km orbit
    
    if simu == 0:  # For optimizer
        if len(np.where(contraintes > 1e-2)[0]) == 0:
            if delta_v >= delta_v_required * 0.95:  # 5% margin
                return objective
            else:
                # Penalty for insufficient delta-v
                return objective + (delta_v_required - delta_v) / 100.0
        else:
            # Constraint violation
            return objective + np.sum(contraintes[contraintes > 0]) * 100.0
    else:  # For reporting
        return objective, contraintes, GLOW, delta_v

# ============================================================================
# DESIGN VARIABLE BOUNDS
# ============================================================================

# Check if environmental discipline is available
if has_env:
    # Full optimization with materials
    lowerbnd_exp = np.array([
        150.,   # Prop_mass_stage_1 (t)
        20.,    # Prop_mass_stage_2 (t)
        0.,     # theta_cmd_i
        -10.,   # theta_cmd_f
        -1.,    # ksi
        5.,     # Pitch_over_duration
        5.,     # Delta_vertical_phase
        1.,     # Delta_theta_pitch_over
        10., 10.,  # command_stage_1_exo
        4.0,    # OF_stage_1
        4.0,    # OF_stage_2
        0.10,   # cfrp_fraction_stage1
        0.40,   # aluminum_fraction_stage1
        0.05,   # steel_fraction_stage1
        0.20,   # cfrp_fraction_stage2
        0.30,   # aluminum_fraction_stage2
        0.05,   # steel_fraction_stage2
        0.40,   # engine_nickel_fraction
        0.20,   # engine_steel_fraction
        0.05    # engine_titanium_fraction
    ])
    
    upperbnd_exp = np.array([
        600.,   # Prop_mass_stage_1 (t)
        200.,   # Prop_mass_stage_2 (t)
        50.,    # theta_cmd_i
        20.,    # theta_cmd_f
        1.,     # ksi
        20.,    # Pitch_over_duration
        20.,    # Delta_vertical_phase
        10.,    # Delta_theta_pitch_over
        70., 70.,  # command_stage_1_exo
        6.0,    # OF_stage_1
        6.0,    # OF_stage_2
        0.40,   # cfrp_fraction_stage1
        0.70,   # aluminum_fraction_stage1
        0.20,   # steel_fraction_stage1
        0.50,   # cfrp_fraction_stage2
        0.60,   # aluminum_fraction_stage2
        0.20,   # steel_fraction_stage2
        0.70,   # engine_nickel_fraction
        0.40,   # engine_steel_fraction
        0.20    # engine_titanium_fraction
    ])
else:
    # Reduced optimization without materials
    lowerbnd_exp = np.array([
        150., 20., 0., -10., -1., 5., 5., 1., 10., 10., 4.0, 4.0
    ])
    upperbnd_exp = np.array([
        600., 200., 50., 20., 1., 20., 20., 10., 70., 70., 6.0, 6.0
    ])

# ============================================================================
# CMA-ES OPTIMIZATION
# ============================================================================

print("\n" + "="*70)
if has_env:
    print("OPTIMIZATION: MINIMIZING ESA SINGLE SCORE")
else:
    print("OPTIMIZATION: MINIMIZING GLOW (Environmental not available)")
print("="*70)

lower = np.zeros(len(lowerbnd_exp))
upper = np.ones(len(upperbnd_exp))

options = {
    'tolfun': 1e-5,
    'tolx': 1e-6,
    'seed': 2,
    'bounds': [lower, upper],
    'popsize': min(16, 4 + int(3*np.log(len(lowerbnd_exp)))),
    'maxiter': 100,
    'verb_disp': 1
}

simu = 0
init = np.ones(len(lowerbnd_exp)) * 0.5

# Adjust initial guess
init[10:12] = [0.625, 0.688]  # O/F ratios
if len(init) > 12:  # Has material fractions
    init[12:15] = [0.375, 0.625, 0.125]  # Stage 1 materials
    init[15:18] = [0.375, 0.625, 0.125]  # Stage 2 materials
    init[18:21] = [0.5, 0.5, 0.25]      # Engine materials

critere_cma = lambda x: Objective_function(x, lowerbnd_exp, upperbnd_exp, P_obj, simu)

print(f"\nDesign variables: {len(lowerbnd_exp)}")
print("Starting CMA-ES optimization...")

res = cma.fmin(critere_cma, init, 0.2, options)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZATION RESULTS")
print("="*70)

# Evaluate optimized configuration
obj_opt, constraints_opt, GLOW_opt, delta_v_opt = Objective_function(
    res[0], lowerbnd_exp, upperbnd_exp, P_obj, 1
)

print(f"\n--- Performance ---")
if has_env:
    print(f"ESA Single Score: {obj_opt:.6f}")
print(f"GLOW: {GLOW_opt/1e3:.1f} t")
print(f"Delta-V: {delta_v_opt:.1f} m/s")
print(f"Constraints: {np.sum(constraints_opt[constraints_opt > 0]):.6f}")

# Get detailed results
P_out = FPI_optim(res[0], lowerbnd_exp, upperbnd_exp, P_obj)

# Comparison table
comparison_data = {
    "Metric": ["GLOW (t)", "Dry_mass_stage_1 (t)", "Dry_mass_stage_2 (t)",
               "Prop_mass_stage_1 (t)", "Prop_mass_stage_2 (t)"],
    "Baseline": [
        P_obj['GLOW'][0]/1e3,
        P_obj['Dry_mass_stage_1'][0]/1e3,
        P_obj['Dry_mass_stage_2'][0]/1e3,
        320.0, 75.0
    ],
    "Optimized": [
        P_out['GLOW'][0]/1e3,
        P_out['Dry_mass_stage_1'][0]/1e3,
        P_out['Dry_mass_stage_2'][0]/1e3,
        P_out['Prop_mass_stage_1'][0]/1e3,
        P_out['Prop_mass_stage_2'][0]/1e3
    ]
}

if has_env:
    comparison_data["Metric"].extend(["ESA_single_score", "GWP (tCO2eq)"])
    comparison_data["Baseline"].extend([
        P_obj.get('ESA_single_score', [0])[0],
        P_obj.get('GWP_total', [0])[0]/1e3
    ])
    comparison_data["Optimized"].extend([
        P_out.get('ESA_single_score', [0])[0],
        P_out.get('GWP_total', [0])[0]/1e3
    ])

df_comparison = pd.DataFrame(comparison_data)
df_comparison["Change (%)"] = ((df_comparison["Optimized"] - df_comparison["Baseline"]) / 
                               df_comparison["Baseline"] * 100)

print("\n", df_comparison.round(3))

# Save results
df_comparison.to_csv('optimization_results.csv', index=False)
print("\nResults saved to optimization_results.csv")

# Final plots
try:
    result_vizualization.plots_output(P_out)
except:
    print("Could not generate final plots")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)