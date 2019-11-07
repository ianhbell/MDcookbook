import os, json, pandas

o = []
for subfolder in [f.path for f in os.scandir(os.path.abspath(os.curdir)) if f.is_dir() ]:
    if subfolder.endswith('.backup'):
        continue
    fname = subfolder+'/results.json'
    if not os.path.exists(fname):
        continue
    o.append(json.load(open(fname)))
df = pandas.DataFrame(o).sort_values(by=['rho^*','T^*'])
df.to_csv('Meier.csv')
print(df)