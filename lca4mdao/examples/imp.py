import os
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

import brightway2 as bw
bw.projects.set_current("LCA_FELIN")
print("Databases:", list(bw.databases))
print("Ecoinvent activities:", len(bw.Database("ecoinvent 3.8 cutoff")))  # expect ~19565
