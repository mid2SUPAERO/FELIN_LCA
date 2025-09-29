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


#Changes made for changing the ratio of materials in the structure (k_SM) in order to minime environmental score

There are two launcher components where k_SM influences: Thrust frame and Interstage.

In dry_mass_stage_1.py:
```
        # Add k_SM as input variables with component-specific ranges
        # Thrust frame: 1.0 = 100% Al, 0.62 = 100% Composite
        self.add_input('k_SM_thrust_frame', val=1.0)
        # Interstage: 1.0 = 100% Al, 0.7 = 100% Composite  
        self.add_input('k_SM_interstage', val=1.0)
        # Intertank: 1.0 = 100% Al, 0.8 = 100% Composite
        self.add_input('k_SM_intertank', val=1.0)

        # Material composition outputs for LCA
        self.add_output('Al_fraction_thrust_frame', val=1.0)
        self.add_output('Composite_fraction_thrust_frame', val=0.0)
        self.add_output('Al_fraction_interstage', val=1.0)
        self.add_output('Composite_fraction_interstage', val=0.0)
        self.add_output('Al_fraction_intertank', val=1.0)
        self.add_output('Composite_fraction_intertank', val=0.0)

        # Computation of thrust frame mass with variable k_SM
        # Instead of fixed material, use k_SM as continuous variable
        k_SM_tf = inputs['k_SM_thrust_frame'][0]
        #Computation of thrust frame mass (structure that support the engines and connect to the tanks)
        M_thrust_frame = mmf.thrust_frame_mass_variable(Total_Thrust/inputs['N_eng_stage_1'][0]/1000,M_eng/inputs['N_eng_stage_1'][0],Constants['NX_max_dim'],inputs['N_eng_stage_1'][0],k_SM_tf,Constants['SSM_TF_1st_stage'])

        #Computation of intertank mass
        k_SM_it = inputs['k_SM_intertank'][0]
        M_FT,M_OxT,M_TPS_OxT,M_TPS_FT,M_intertank = mmf.tank_mass_variable(inputs['Pdyn_max_dim'][0],Constants['NX_max_dim'],Constants['P_tank_Ox'],Constants['P_tank_F'],V_FT,V_Ox,inputs['Diameter_stage_1']            [0],S_Ox,S_F,2*S_dome,S_totale,Constants['Type_prop'],Constants['Tank_config'],Constants['Stage_in_staging'],k_SM_it)

        #Computation of interstage mass
        k_SM_is = inputs['k_SM_interstage'][0]
        M_interstage = mmf.mass_interstage_variable(Constants['S_interstage'],inputs['Diameter_stage_1'][0],inputs['Diameter_stage_2'][0],Constants['Type_interstage'],k_SM_is)

        Dry_mass_stage_1 = M_eng+M_thrust_frame+M_FT+M_OxT+M_TPS_OxT+\
                        M_TPS_FT+M_TVC+M_avio+M_EPS+M_intertank+\
                        M_interstage+Constants['Masse_aux_stage_1']

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
```

In Environmental_Discipline.py:
```
        # Material fractions for variable components
        self.add_input('Al_fraction_thrust_frame_stage_1', val=1.0)
        self.add_input('Composite_fraction_thrust_frame_stage_1', val=0.0)
        self.add_input('Al_fraction_interstage_stage_1', val=1.0)
        self.add_input('Composite_fraction_interstage_stage_1', val=0.0)
        self.add_input('Al_fraction_intertank_stage_1', val=1.0)
        self.add_input('Composite_fraction_intertank_stage_1', val=0.0)

        # Stage 1 components with variable materials
        # Thrust frame (variable Al/Composite)
        aluminum_total += inputs['M_thrust_frame_stage_1'][0] * inputs['Al_fraction_thrust_frame_stage_1'][0]
        composite_total += inputs['M_thrust_frame_stage_1'][0] * inputs['Composite_fraction_thrust_frame_stage_1'][0]
        
        # Interstage (variable Al/Composite)
        aluminum_total += inputs['M_interstage_stage_1'][0] * inputs['Al_fraction_interstage_stage_1'][0]
        composite_total += inputs['M_interstage_stage_1'][0] * inputs['Composite_fraction_interstage_stage_1'][0]

       # Intertank (variable Al/Composite)
        aluminum_total += inputs['M_intertank_stage_1'][0] * inputs['Al_fraction_intertank_stage_1'][0]
        composite_total += inputs['M_intertank_stage_1'][0] * inputs['Composite_fraction_intertank_stage_1'][0]

```
