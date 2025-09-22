import os
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

import brightway2 as bw
bw.projects.set_current("LCA_FELIN")   # <-- this is the key line

print("Project:", bw.projects.current)
print("DBs:", list(bw.databases))
print("Methods count:", len(list(bw.methods)))

# now search for IPCC 100a
print([m for m in bw.methods if 'ipcc' in str(m).lower() and '100a' in str(m).lower()])

# if still empty, broaden the search:
print([m for m in bw.methods if 'ipcc' in str(m).lower()])
print([m for m in bw.methods if 'climate change' in str(m).lower()])
