## Use "docker-compose up --build" to build and run

FROM centos:7.3.1611

RUN yum install -y --setopt=skip_missing_names_on_install=False cmake git zip gcc make openmpi openmpi-devel gcc-c++ numpy

# This ADD block forces a build (invalidates the cache) if the git repo contents have changed, otherwise leaves it untouched.
# See https://stackoverflow.com/a/39278224
ADD https://api.github.com/repos/dwsideriusNIST/LAMMPS_Examples/git/refs/heads/master examples-version.json
RUN git clone --recursive https://github.com/dwsideriusNIST/LAMMPS_Examples.git
ADD https://api.github.com/repos/lammps/lammps/git/refs/heads/master examples-version.json
RUN git clone --recursive https://github.com/lammps/lammps.git

# Add necessary paths
ENV PATH="/lammps/src:/usr/lib64/openmpi/bin:${PATH}"

# Build the LAMMPS executable
WORKDIR /lammps/src
RUN git checkout r15061 && \
    make package-update && \
    make yes-user-misc && \
    make yes-opt && \
    make -j8 mpi

# Run the test
WORKDIR /LAMMPS_Examples/LJ_example
RUN ln -s ../LJ_initial_cfgs/in.nvt.dens_*.* ./

# The task to run
CMD python -c "contents = open('example.sh').read(); fo = open('example.sh', 'w'); fo.write(contents.replace('-np 8','--allow-run-as-root -np 8')); fo.close()" && \
    mpirun -np 12 --allow-run-as-root lmp_mpi -sf opt -in NVT.startfromrestart -var rho 0.4000 -var temp 1.5 && \
    ls -al && \
    ../analysis/block_analysis.py -f ave.dens_0.4000.out -b 5 -m 100000