import brightway2 as bw
print([m for m in bw.methods if 'ipcc' in str(m).lower() and '100a' in str(m).lower()])
