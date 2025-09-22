import os
# 1) Point Brightway to the correct base folder
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

import brightway2 as bw

print("Base BW dir (should be ...\\Brightway3):", bw.projects.dir)

# 2) Create the target project if it doesn't exist yet
if "LCA_FELIN" not in [p.name for p in bw.projects]:
    bw.projects.create_project("LCA_FELIN")  # (some BW versions ignore activate=True)

# 3) Now ACTIVATE the project explicitly
bw.projects.set_current("LCA_FELIN")
print("Current project:", bw.projects.current)

# 4) THIS is the per-project folder you want (should look like LCA_FELIN.<hash>)
target_dir = bw.projects.dir
print("Target project dir:", target_dir)
