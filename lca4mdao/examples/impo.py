# clone_bw_project.py  (run with: python clone_bw_project.py)
import os, shutil

SRC = r"C:\Users\joana\AppData\Local\pylca\Brightway3\Ariana_5_LCA.1f9202b8d3ddf35f04d8ae2176d379f3"
DST = r"C:\Users\joana\AppData\Local\pylca\Brightway3\LCA_FELIN.b1133b8f7c1854b5aa2d091753cb4ec0"

print("Copying from:\n ", SRC, "\nTo:\n ", DST)

# 1) Clear destination (no BW imported → no locks)
for name in os.listdir(DST):
    p = os.path.join(DST, name)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            os.remove(p)
        except PermissionError:
            pass

# 2) Copy EVERYTHING (folders + files)
for name in os.listdir(SRC):
    s = os.path.join(SRC, name)
    d = os.path.join(DST, name)
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print("Copy complete.")
