import os
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

import brightway2 as bw
bw.projects.create_project("LCA_FELIN", activate=True)
target_dir = bw.projects.dir
print("Target project dir:", target_dir)  # e.g. ...\LCA_FELIN.<hash>