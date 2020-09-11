import subprocess, pandas
subprocess.check_call('rumd_bonds -n 10000 -t start.top', shell=True)
df = pandas.read_csv('bonds.dat', sep=' ', comment='#', names = ['length','count','dummy'])
del df['dummy']
# df.info()
stiffness = 3000
print(stiffness/2.0*((df['length']-1.0)**2*df['count']).sum()/df['count'].sum()*1100/1200*12)