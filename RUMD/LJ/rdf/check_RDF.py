from __future__ import print_function
import sys

import os, json, shutil
from collections import namedtuple
EMDPoint = namedtuple('EMDPoint', ['Tstar','rhostar','etastar','Dstar'])

pts = []
for Tstar in [2, 3, 4, 5, 6]:
    for rho in [0.1, 0.5, 1.0, 1.5]:
        pts.append(EMDPoint(Tstar, rhostar, None, None))

sys.path.insert(0, '..')
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