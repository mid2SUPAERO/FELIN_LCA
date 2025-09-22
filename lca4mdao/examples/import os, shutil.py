import os, shutil
import brightway2 as bw

# Always point to the same Brightway base
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

# Source (your GOOD project with ecoinvent)
SRC = r"C:\Users\joana\AppData\Local\pylca\Brightway3\Ariana_5_LCA.1f9202b8d3ddf35f04d8ae2176d379f3"

# Target (your NEW project)
bw.projects.set_current("LCA_FELIN")
DST = bw.projects.dir
print("Copying into:", DST)

# 1) Copy 'databases' folder
src_db = os.path.join(SRC, "databases")
dst_db = os.path.join(DST, "databases")
if os.path.exists(dst_db):
    shutil.rmtree(dst_db)
shutil.copytree(src_db, dst_db)

# 2) Copy 'databases.json'
shutil.copy2(os.path.join(SRC, "databases.json"),
             os.path.join(DST, "databases.json"))

# (Optional) copy 'processed' and 'intermediate' if present
for name in ["processed", "intermediate"]:
    s = os.path.join(SRC, name)
    d = os.path.join(DST, name)
    if os.path.exists(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)

print("Copy complete.")
