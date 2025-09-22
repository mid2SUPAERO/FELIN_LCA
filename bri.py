import os, brightway2 as bw
os.environ["BW2DIR"] = r"C:\Users\joana\AppData\Local\pylca\Brightway3"
bw.projects.set_current("LCA_FELIN")

def show(q):
    print(q, "->")
    print([m for m in bw.methods if q in str(m).lower()], "\n")

for q in [
    'human toxicity', 'non-cancer', 'cancer', 'ionising', 'particulate',
    'ozone depletion', 'acidification', 'eutrophication', 'freshwater ecotoxicity',
    'land use', 'water use', 'resource use', 'fossil', 'minerals'
]:
    show(q)
