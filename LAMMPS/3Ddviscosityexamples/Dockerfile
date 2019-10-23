# Largely based on https://gitlab.com/costrouc/public-docker/blob/master/lammps/Dockerfile.debian 
# but uses a different CentOS-based base image.
# 
# Use "docker build -t lampvis ." to build and "docker run -it -t lampvis" to run
#
# These images don't work: ubuntu:18.04 ; nvidia/cuda:10.1-devel ; debian:buster

# CentOS 7 + CUDA stuff for ultimate use with GPU acceleration
FROM nvidia/cuda:10.1-devel-centos7
MAINTAINER Ian Bell "ian.bell@nist.gov"

RUN yum -y update && \
    yum -y install  wget ssh zlib-devel \
                    make gcc gcc-c++ \
                    openmpi-devel openblas-devel fftw3-devel git nano && \
    yum -y clean all

ENV PATH="${PATH}:/usr/lib64/openmpi/bin/"

# ARG PACKAGES="asphere body class2 colloid compress coreshell dipole granular kspace manybody mc misc molecule opt peri qeq replica rigid shock snap srd user-reaxc"
ARG PACKAGES="opt"
ARG BUILD="mpi"
ARG LMP_INCLUDES="-DLAMMPS_EXCEPTIONS -DLAMMPS_GZIP -DLAMMPS_MEMALIGN=64"

# This ADD block forces a build (invalidates the cache) if the git repo contents have changed, otherwise leaves it untouched.
# See https://stackoverflow.com/a/39278224
ADD https://api.github.com/repos/lammps/lammps/git/refs/heads/stable stable-meta.json
RUN git clone --recursive https://github.com/lammps/lammps.git && cd lammps && git checkout stable

# Build the LAMMPS executable
WORKDIR /lammps/src
RUN for pack in $PACKAGES; do make "yes-$pack"; done && \
   make mode=lib $BUILD -j12 LMP_INC="$LMP_INCLUDES" && \
   make mode=shlib $BUILD -j12 LMP_INC="$LMP_INCLUDES" && \
   make $BUILD && \
   cp lmp_$BUILD /usr/local/bin/lammps && \
   cp liblammps_$BUILD.so /usr/local/lib/liblammps.so && \
   cp liblammps_$BUILD.a /usr/local/lib/liblammps.a && \
   mkdir /usr/local/include/lammps && \
   cp *.h /usr/local/include/lammps

# Copy over the script
COPY *.3d /

# The task to run
WORKDIR /
CMD for fname in LJ_*.3d; do mpirun -np 12 --allow-run-as-root lammps -sf opt -in $fname; done
