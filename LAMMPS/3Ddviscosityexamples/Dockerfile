## Use "docker build -t lampvis ." to build and "docker run -it -t lampvis" to run

# 18.04 ubuntu + CUDA stuff for ultimate use with GPU acceleration
FROM nvidia/cuda:10.1-devel 
#FROM costrouc/lammps:patch_11May2018-debian-mpi-all

RUN apt-get -y -m update && apt-get install -y cmake python3-numpy g++ gfortran git zip libopenmpi-dev nano
#RUN apt-get -y -m update && apt-get install -y nano

# This ADD block forces a build (invalidates the cache) if the git repo contents have changed, otherwise leaves it untouched.
# See https://stackoverflow.com/a/39278224
ADD https://api.github.com/repos/lammps/lammps/git/refs/heads/master examples-version.json
RUN git clone --recursive https://github.com/lammps/lammps.git && git checkout stable

# Add necessary paths
ENV PATH="/lammps/src:/usr/lib64/openmpi/bin:${PATH}"

# Build the LAMMPS executable
WORKDIR /lammps/src
RUN make package-update && \
   make yes-user-misc && \
   make yes-opt && \
   make -j8 mpi
   # make yes-gpu && \

# Copy over the script
COPY Ar_GK.3d /

# The task to run
WORKDIR /
CMD for fname in *.3d; do mpirun -np 12 --allow-run-as-root lammps -sf opt -in $fname; done