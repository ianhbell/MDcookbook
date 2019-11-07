import pandas, glob

dfs = []
for fname in glob.glob('repli*.csv'):
    df = pandas.read_csv(fname)
    df['rep'] = int(fname.split('.')[0][-1])
    df['Tnom^*'] = df.apply(lambda row: round(row['T^*'],3), axis=1)
    dfs.append(df)

df = pandas.concat(dfs).sort_values(by=['Tnom^*','rho^*'])
for (Tstar,rhostar),gp in df.groupby(['Tnom^*','rho^*']):
    eta = gp['eta^*']
    D = gp['D^*']
    print(Tstar, rhostar, eta.mean(), '+-', eta.std()*2/eta.mean()*100, '% (k=2)', D.mean(), '+-', D.std()*2/D.mean()*100, '% (k=2)')
