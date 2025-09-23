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

<img width="974" height="202" alt="image" src="https://github.com/user-attachments/assets/50fe5a26-c33e-4e00-91b4-46eb097bb7ba" />
