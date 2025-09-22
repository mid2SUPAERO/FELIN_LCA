import os, brightway2 as bw
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"
bw.projects.set_current("LCA_FELIN")
target_dir = bw.projects.dir
print("Target:", target_dir)  # e.g. ...\LCA_FELIN.b1133b8f7c1854...
