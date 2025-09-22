import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt

import openmdao.api as om
import cma

import Launch_vehicle_Group
import post_traitement
import result_vizualization
import constants as Cst
import specifications as Spec

def FPI_with_environment(x, lowerbnd, upperbnd, Pb):
    """
    Fixed Point Iteration with environmental discipline
    x: normalized design variables [0,1]
    """
    
    Pb.setup(check=False)
    
    # Denormalize design variables
    XX = lowerbnd + (upperbnd - lowerbnd) * x
    
    # SET DESIGN VARIABLES
    
    # Propellant masses
    Pb['Prop_mass_stage_1'] = XX[0] * 1e3  # Convert to kg
    Pb['Prop_mass_stage_2'] = XX[1] * 1e3
    
    # O/F ratios
    Pb['OF_stage_1'] = XX[2]
    Pb['OF_stage_2'] = XX[3]
    
    # NORMALIZE MATERIAL FRACTIONS TO SUM TO 1
    # Stage 1 materials
    mat_s1_sum = XX[4] + XX[5] + XX[6]
    if mat_s1_sum > 0:
        Pb['cfrp_fraction_stage1'] = XX[4] / mat_s1_sum
        Pb['aluminum_fraction_stage1'] = XX[5] / mat_s1_sum
        Pb['steel_fraction_stage1'] = XX[6] / mat_s1_sum
    
    # Stage 2 materials
    mat_s2_sum = XX[7] + XX[8] + XX[9]
    if mat_s2_sum > 0:
        Pb['cfrp_fraction_stage2'] = XX[7] / mat_s2_sum
        Pb['aluminum_fraction_stage2'] = XX[8] / mat_s2_sum
        Pb['steel_fraction_stage2'] = XX[9] / mat_s2_sum
    
    # Engine materials
    mat_eng_sum = XX[10] + XX[11] + XX[12]
    if mat_eng_sum > 0:
        Pb['engine_nickel_fraction'] = XX[10] / mat_eng_sum
        Pb['engine_steel_fraction'] = XX[11] / mat_eng_sum
        Pb['engine_titanium_fraction'] = XX[12] / mat_eng_sum
    
    # Trajectory parameters
    Pb['thetacmd_i'] = XX[13]
    Pb['thetacmd_f'] = XX[14]
    Pb['ksi'] = XX[15]
    Pb['Pitch_over_duration'] = XX[16]
    Pb['Delta_vertical_phase'] = XX[17]
    Pb['Delta_theta_pitch_over'] = XX[18]
    Pb['command_stage_1_exo'] = XX[19:21]
    
    # FIXED PARAMETERS
    Pb['Diameter_stage_1'] = 5.0
    Pb['Diameter_stage_2'] = 5.0
    Pb['Mass_flow_rate_stage_1'] = 250.
    Pb['Mass_flow_rate_stage_2'] = 250.
    Pb['Thrust_stage_1'] = 1000.
    Pb['Thrust_stage_2'] = 800.
    Pb['Pc_stage_1'] = 100.0
    Pb['Pc_stage_2'] = 100.0
    Pb['Pe_stage_1'] = 1.0
    Pb['Pe_stage_2'] = 1.0
    Pb['N_eng_stage_1'] = 8.
    Pb['N_eng_stage_2'] = 1.
    Pb['Exit_nozzle_area_stage_1'] = 0.79
    Pb['Exit_nozzle_area_stage_2'] = 3.6305
    Pb['is_fallout'] = 0.
    Pb['payload_mass'] = 5000.0
    
    # FIXED POINT ITERATION FOR PDYN
    error = 100.
    Pb['Pdyn_max_dim'] = 36.24
    k = 0
    
    while error > 1. and k < 10:  # Reduced iterations for faster convergence, should increse for better results
        try:
            Pb.run_model()
            error = abs(Pb['Pdyn_max_dim'] - Pb['max_pdyn_load_ascent_stage_1']/1e3)
            Pb['Pdyn_max_dim'] = Pb['max_pdyn_load_ascent_stage_1']/1e3
            k += 1
        except:
            return Pb, False
    
    return Pb, (error <= 1.)

def objective_ESA(x, lowerbnd, upperbnd, Pb):
    """
    Objective: Minimize ESA Single Score
    Subject to: Reaching orbital velocity and realistic design
    """

    if not hasattr(objective_ESA, 'best_esa'):
        objective_ESA.best_esa = 1e10
        objective_ESA.count = 0

    objective_ESA.count += 1
    
    # Run FPI
    P_out, converged = FPI_with_environment(x, lowerbnd, upperbnd, Pb)
    
    if not converged:
        return 1e10
    
    try:
        # Get trajectory performance
        nb_pts = int(P_out['Nb_pt_ascent'][0])
        if nb_pts > 0:
            v_initial = P_out['V_ascent'][0]
            v_final = P_out['V_ascent'][nb_pts-1]
            delta_v = v_final - v_initial
        else:
            delta_v = 0.0
        
        # Get ESA score
        esa_score = P_out['ESA_single_score'][0]

        # After calculating esa_score:
        if esa_score < objective_ESA.best_esa and delta_v >= 7800:
            objective_ESA.best_esa = esa_score
            if objective_ESA.count % 10 == 0:
                print(f"New best ESA: {esa_score:.1f} at ΔV={delta_v:.1f} m/s")
        
        #  CONSTRAINTS 
        penalties = 0.0
        
        # 1. Must reach orbital velocity (reduced penalty)
        if delta_v < 7800:
            penalties += 10 * (7800 - delta_v)  # Reduced from 1000
        
        # 2. Propellant ratio constraint (stage 2 should be 15-30% of stage 1)
        XX = lowerbnd + (upperbnd - lowerbnd) * x
        prop_ratio = XX[1] / XX[0]  # Stage2/Stage1
        if prop_ratio < 0.15 or prop_ratio > 0.30:
            penalties += 100 * abs(prop_ratio - 0.20)
        
        # 3. Total propellant mass constraint (reasonable launcher size)
        total_prop = XX[0] + XX[1]
        if total_prop > 450:  # Max 450t total propellant
            penalties += 10 * (total_prop - 450)
        
        # 4. Check GLOW constraint
        glow = P_out['GLOW'][0] / 1000  # Convert to tons
        if glow > 500:  # Max 500t GLOW
            penalties += 5 * (glow - 500)
        
        return esa_score + penalties
        
    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e10
    
def print_status(P_out, iteration=None):
    """Print current optimization status"""
    
    try:
        nb_pts = int(P_out['Nb_pt_ascent'][0])
        delta_v = P_out['V_ascent'][nb_pts-1] - P_out['V_ascent'][0]
        
        if iteration is not None:
            print(f"\n Iteration {iteration} ===")
        else:
            print("\n=== Current Status ===")
            
        print(f"ESA Single Score: {P_out['ESA_single_score'][0]:.1f}")
        print(f"GWP Total: {P_out['GWP_total'][0]/1e3:.1f} tCO2eq")
        print(f"GLOW: {P_out['GLOW'][0]/1e3:.1f} t")
        print(f"Delta-V: {delta_v:.1f} m/s")
        print(f"Max acceleration: {P_out['max_acceleration_g'][0]:.2f} g")
        
        # Material breakdown
        print(f"\nMaterial masses (t):")
        print(f"  Stage 1: CFRP={P_out['stage1_cfrp_mass'][0]/1e3:.1f}, "
              f"Al={P_out['stage1_aluminum_mass'][0]/1e3:.1f}, "
              f"Steel={P_out['stage1_steel_mass'][0]/1e3:.1f}")
        print(f"  Stage 2: CFRP={P_out['stage2_cfrp_mass'][0]/1e3:.1f}, "
              f"Al={P_out['stage2_aluminum_mass'][0]/1e3:.1f}, "
              f"Steel={P_out['stage2_steel_mass'][0]/1e3:.1f}")
        
        # Propellants
        print(f"\nPropellants (t):")
        print(f"  LOX: {P_out['total_lox_mass'][0]/1e3:.1f}")
        print(f"  LH2: {P_out['total_lh2_mass'][0]/1e3:.1f}")
        print(f"  O/F ratios: {P_out['OF_stage_1'][0]:.2f}/{P_out['OF_stage_2'][0]:.2f}")
        
    except Exception as e:
        print(f"Error printing status: {e}")

# Create problem
P_obj = om.Problem()
P_obj.model = Launch_vehicle_Group.Launcher_vehicle()

# Design variable bounds
lowerbnd = np.array([
    200.,   # Prop_mass_stage_1 (t)
    30.,    # Prop_mass_stage_2 (t)
    4.0,    # OF_stage_1
    4.0,    # OF_stage_2
    0.10,   # cfrp_fraction_stage1
    0.40,   # aluminum_fraction_stage1
    0.05,   # steel_fraction_stage1
    0.10,   # cfrp_fraction_stage2
    0.40,   # aluminum_fraction_stage2
    0.05,   # steel_fraction_stage2
    0.40,   # engine_nickel_fraction
    0.20,   # engine_steel_fraction
    0.05,   # engine_titanium_fraction
    0.,     # thetacmd_i
    5.,     # thetacmd_f
    0.2,    # ksi
    5.,     # Pitch_over_duration
    5.,     # Delta_vertical_phase
    1.,     # Delta_theta_pitch_over
    -30., -30.  # command_stage_1_exo
])

upperbnd = np.array([
    400.,   # Prop_mass_stage_1 (t)
    100.,   # Prop_mass_stage_2 (t)
    6.0,    # OF_stage_1
    6.0,    # OF_stage_2
    0.40,   # cfrp_fraction_stage1
    0.70,   # aluminum_fraction_stage1
    0.20,   # steel_fraction_stage1
    0.40,   # cfrp_fraction_stage2
    0.70,   # aluminum_fraction_stage2
    0.20,   # steel_fraction_stage2
    0.70,   # engine_nickel_fraction
    0.40,   # engine_steel_fraction
    0.20,   # engine_titanium_fraction
    5.,     # thetacmd_i
    15.,    # thetacmd_f
    0.4,    # ksi
    10.,    # Pitch_over_duration
    15.,    # Delta_vertical_phase
    5.,     # Delta_theta_pitch_over
    70., 70.  # command_stage_1_exo
])

print('Problem and bounds initialised.')

# Reference baseline 
x_baseline = np.array([
    250., 50.6,            # Propellant masses (t)
    5.5, 5.5,              # O/F ratios
    0.10, 0.70, 0.20,      # Stage 1 materials
    0.10, 0.70, 0.20,      # Stage 2 materials
    0.60, 0.30, 0.10,      # Engine materials
    2.72, 10., 0.293,      # Trajectory
    5., 10., 1.,           # More trajectory
    30., -20.              # Command
])

x_norm = (x_baseline - lowerbnd) / (upperbnd - lowerbnd)

P_baseline, converged = FPI_with_environment(x_norm, lowerbnd, upperbnd, P_obj)
if converged:
    print_status(P_baseline)
else:
    print("Baseline failed to converge!")

'''=== Current Status ===
ESA Single Score: 266.8
GWP Total: 1166.1 tCO2eq
GLOW: 343.8 t
Delta-V: 7562.2 m/s
Max acceleration: 4.50 g

Material masses (t):
  Stage 1: CFRP=3.0, Al=21.3, Steel=6.1
  Stage 2: CFRP=0.6, Al=4.0, Steel=1.1

Propellants (t):
  LOX: 254.4
  LH2: 46.2
  O/F ratios: 5.50/5.50'''

print("\n" + "="*60)
print("STARTING CMA-ES OPTIMIZATION")
print("="*60)

options = {
    'bounds': [np.zeros(len(lowerbnd)), np.ones(len(upperbnd))],
    'popsize': 12,
    'maxiter': 30,
    'tolfun': 1e-3,
    'tolx': 1e-4,
    'verb_disp': 1,
    'seed': 42,
    'CMA_stds': 0.1
}

init = x_norm.copy()

critere_cma = lambda x: objective_ESA(x, lowerbnd, upperbnd, P_obj)
res = cma.fmin(critere_cma, init, 0.2, options)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

'''============================================================
STARTING CMA-ES OPTIMIZATION
============================================================
(6_w,12)-aCMA-ES (mu_w=3.7,w_1=40%) in dimension 21 (seed=42, Fri Aug 29 11:26:14 2025)
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
    1     12 2.410449167403392e+03 1.0e+00 1.84e-01  2e-02  2e-02 1:06.5
    2     24 2.760901662969989e+03 1.1e+00 1.78e-01  2e-02  2e-02 2:12.6
    3     36 2.669255592116614e+02 1.1e+00 1.69e-01  2e-02  2e-02 3:17.3
    4     48 2.667201783538147e+02 1.1e+00 1.63e-01  2e-02  2e-02 4:10.1
    5     60 2.672669369912905e+02 1.2e+00 1.63e-01  2e-02  2e-02 5:05.7
    6     72 5.500421227602051e+02 1.2e+00 1.63e-01  2e-02  2e-02 6:05.0
    7     84 2.668415997415191e+02 1.2e+00 1.62e-01  2e-02  2e-02 7:27.5
    8     96 2.668081109125185e+02 1.2e+00 1.54e-01  1e-02  2e-02 8:41.3
    9    108 2.667022082151540e+02 1.2e+00 1.48e-01  1e-02  2e-02 9:57.5
   10    120 2.653306813331924e+02 1.2e+00 1.44e-01  1e-02  1e-02 11:13.7
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
   11    132 2.667456355219028e+02 1.2e+00 1.39e-01  1e-02  1e-02 12:30.3
   12    144 2.670311522502480e+02 1.3e+00 1.37e-01  1e-02  1e-02 13:48.9
   13    156 2.661781003912224e+02 1.3e+00 1.44e-01  1e-02  1e-02 15:11.7
   14    168 2.660470753849982e+02 1.4e+00 1.46e-01  1e-02  2e-02 16:30.9
   15    180 2.669307612555463e+02 1.4e+00 1.45e-01  1e-02  2e-02 17:55.4
   16    192 2.665231596156264e+02 1.4e+00 1.44e-01  1e-02  2e-02 19:18.2
   17    204 2.664815992919807e+02 1.4e+00 1.45e-01  1e-02  2e-02 20:41.9
   18    216 2.660951162198336e+02 1.4e+00 1.47e-01  1e-02  2e-02 22:02.8
   19    228 2.669994741460367e+02 1.4e+00 1.53e-01  1e-02  2e-02 23:15.6
   20    240 3.735168446790035e+02 1.4e+00 1.56e-01  1e-02  2e-02 24:12.4
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
   21    252 2.669765013975288e+02 1.5e+00 1.51e-01  1e-02  2e-02 25:09.9
   22    264 2.663568822811110e+02 1.5e+00 1.49e-01  1e-02  2e-02 26:09.9
   23    276 2.665246598837821e+02 1.5e+00 1.46e-01  1e-02  2e-02 27:06.9
   24    288 2.669604469760463e+02 1.5e+00 1.44e-01  1e-02  2e-02 28:03.3
   25    300 2.675914454513061e+02 1.5e+00 1.37e-01  1e-02  1e-02 28:59.8
   26    312 1.413832141226649e+03 1.5e+00 1.25e-01  1e-02  1e-02 29:56.6
   27    324 2.668763686993811e+02 1.5e+00 1.18e-01  1e-02  1e-02 30:54.4
   28    336 2.665967603146824e+02 1.5e+00 1.12e-01  1e-02  1e-02 31:51.1
   29    348 2.666082038824566e+02 1.6e+00 1.08e-01  1e-02  1e-02 32:47.5
   30    360 2.663646223909973e+02 1.6e+00 1.03e-01  9e-03  1e-02 33:43.8
termination on maxiter=30 (Fri Aug 29 12:00:03 2025)
final/bestever f-value = 2.668590e+02 2.653307e+02
incumbent solution: [2.49512367e-01 2.90928040e-01 7.47438509e-01 7.15908011e-01
 2.22349989e-04 9.98637307e-01 9.90891473e-01 1.17861650e-03 ...]
std deviations: [0.00936895 0.01069306 0.01022887 0.0100788  0.00974736 0.0104747
 0.01087009 0.00991915 ...]

============================================================
OPTIMIZATION COMPLETE
============================================================'''

x_opt = res[0]

P_opt, converged = FPI_with_environment(x_opt, lowerbnd, upperbnd, P_obj)

if converged:
    print("\n=== OPTIMAL CONFIGURATION ===")
    print_status(P_opt)

    XX_opt = lowerbnd + (upperbnd - lowerbnd) * x_opt
    print("\n=== OPTIMAL DESIGN VARIABLES ===")
    print(f"Propellant masses: {XX_opt[0]:.1f}t / {XX_opt[1]:.1f}t")
    print(f"O/F ratios: {XX_opt[2]:.2f} / {XX_opt[3]:.2f}")
    print(f"Stage 1 materials: CFRP={XX_opt[4]:.2f}, Al={XX_opt[5]:.2f}, Steel={XX_opt[6]:.2f}")
    print(f"Stage 2 materials: CFRP={XX_opt[7]:.2f}, Al={XX_opt[8]:.2f}, Steel={XX_opt[9]:.2f}")
    print(f"Engine materials: Ni={XX_opt[10]:.2f}, Steel={XX_opt[11]:.2f}, Ti={XX_opt[12]:.2f}")

    print("\n=== ESA CATEGORY BREAKDOWN ===")
    for cat in ['GWP', 'ODEPL', 'PMAT', 'LUP', 'WDEPL']:
        try:
            raw = P_opt[f'ESA_{cat}'][0]
            norm = P_opt[f'ESA_{cat}_normalized'][0]
            print(f"{cat}: raw={raw:.2e}, normalized={norm:.2e}")
        except KeyError:
            print(f"{cat}: (keys not found in model outputs)")

    # Plot outputs if helper is available
    try:
        result_vizualization.plots_output(P_opt)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    data = [[
        P_opt['ESA_single_score'][0],
        P_opt['GWP_total'][0]/1e3,
        P_opt['GLOW'][0]/1e3,
        P_opt['Dry_mass_stage_1'][0]/1e3,
        P_opt['Dry_mass_stage_2'][0]/1e3,
        P_opt['Prop_mass_stage_1'][0]/1e3,
        P_opt['Prop_mass_stage_2'][0]/1e3,
        P_opt['delta_v_achieved'][0]
    ]]

    df = pd.DataFrame(data, columns=[
        "ESA Score", "GWP (tCO2eq)", "GLOW (t)",
        "Dry_mass_stage_1 (t)", "Dry_mass_stage_2 (t)",
        "Prop_mass_stage_1 (t)", "Prop_mass_stage_2 (t)",
        "Delta-V (m/s)"
    ]).round(2)

    print("\n=== SUMMARY TABLE ===")
    display(df)

    df.to_csv("optimization_results.csv", index=False)
    print("\nResults saved to optimization_results.csv")

else:
    print("ERROR: Optimal solution failed to converge!")

'''=== OPTIMAL CONFIGURATION ===

=== Current Status ===
ESA Single Score: 265.3
GWP Total: 1157.1 tCO2eq
GLOW: 336.9 t
Delta-V: 7908.9 m/s
Max acceleration: 4.50 g

Material masses (t):
  Stage 1: CFRP=3.0, Al=21.2, Steel=6.1
  Stage 2: CFRP=0.6, Al=4.0, Steel=1.2

Propellants (t):
  LOX: 248.8
  LH2: 45.0
  O/F ratios: 5.54/5.50

=== OPTIMAL DESIGN VARIABLES ===
Propellant masses: 241.7t / 52.2t
O/F ratios: 5.54 / 5.50
Stage 1 materials: CFRP=0.10, Al=0.70, Steel=0.20
Stage 2 materials: CFRP=0.10, Al=0.70, Steel=0.20
Engine materials: Ni=0.60, Steel=0.31, Ti=0.10

=== ESA CATEGORY BREAKDOWN ===
GWP: raw=1.16e+06, normalized=1.43e+02
ODEPL: raw=7.04e-02, normalized=1.31e+00
PMAT: raw=9.28e-02, normalized=1.56e+02
LUP: raw=4.10e+06, normalized=5.00e+00
WDEPL: raw=1.24e+06, normalized=1.08e+02'''