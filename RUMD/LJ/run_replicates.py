import subprocess, shutil
import os, json, pandas

for i in range(10):
    #subprocess.check_call('docker run --gpus device=1 --user $(id -u):$(id -g) -v "${PWD}":/output rumd bash -c "cd /output && python validate_Meier.py"', shell=True)
    subprocess.check_call('docker run --gpus device=1  -v "${PWD}":/output rumd bash -c "cd /output && python validate_Meier.py"', shell=True)
    o = []
    for subfolder in [f.path for f in os.scandir(os.path.abspath(os.curdir)) if f.is_dir() ]:
        if subfolder.endswith('.backup'):
            continue
        fname = subfolder+'/results.json'
        if not os.path.exists(fname):
            continue
        o.append(json.load(open(fname)))
        # Copy files
        sub = os.path.split(subfolder)[1]
        shutil.copy2(sub+'/energies.csv', 'rep'+str(i)+sub+'-energies.csv')
        shutil.copy2(sub+'/GreenKubo_fitting.pdf', 'rep'+str(i)+sub+'-GreenKubo.pdf')
    df = pandas.DataFrame(o).sort_values(by=['rho^*','T^*'])
    df.to_csv('Meier.csv')
    print(df)
    shutil.move('Meier.csv', 'replicate'+str(i)+'.csv')
