import os, shutil, brightway2 as bw
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"

SRC = r"C:\Users\joana\AppData\Local\pylca\Brightway3\Ariana_5_LCA.1f9202b8d3ddf35f04d8ae2176d379f3"
bw.projects.set_current("LCA_FELIN")
DST = bw.projects.dir

print("Copying from:\n ", SRC, "\nTo:\n ", DST)

# Copy everything (folders + files). Overwrite if exists.
for name in os.listdir(SRC):
    s = os.path.join(SRC, name)
    d = os.path.join(DST, name)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print("Copy complete.")