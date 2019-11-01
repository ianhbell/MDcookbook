from __future__ import print_function

import os, json, shutil
from collections import namedtuple
EMDPoint = namedtuple('EMDPoint', ['Tstar','rhostar','etastar','Dstar'])

# Points from Meier's publications
pts = [
    EMDPoint(0.9,0.025,0.09764,5.1107),
    EMDPoint(4,0.025,0.387,0.47277),
    EMDPoint(1.5,0.3,0.3275,0.20146),
    EMDPoint(6.0,0.3,0.6465,0.53799),
    EMDPoint(0.722,0.8442,3.258,0.325),  # Estimated 
]

import EMD

here = os.path.abspath(os.path.dirname(__file__))
for pt in pts:
    os.chdir(here)
    folder = 'T{0:g}D{1:g}'.format(pt.Tstar, pt.rhostar)
    if os.path.exists(folder):
        newfolder = folder+'.backup'
        if os.path.exists(newfolder):
            shutil.rmtree(newfolder)
        print('Moving', folder, 'to', newfolder)
        shutil.move(folder, newfolder)
    os.makedirs(folder)
    os.chdir(folder)
    print('Running RUMD from', os.path.abspath(os.curdir))

    EMD.do_run(Tstar=pt.Tstar, rhostar=pt.rhostar)
    EMD.post_process()