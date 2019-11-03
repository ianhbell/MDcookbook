docker build -t lam3d .
docker run -v "${PWD}":/output -d lam3d bash -c "cd /output && mpirun -np 12 --allow-run-as-root lammps -sf opt -in LJ_GK.3d -var rhostar 0.3 -var Tstar 1.3"
