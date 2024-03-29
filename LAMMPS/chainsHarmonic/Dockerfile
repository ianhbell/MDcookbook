# Largely based on https://gitlab.com/costrouc/public-docker/blob/master/lammps/Dockerfile.debian 
# but uses a different CentOS-based base image.
# 
# Use "docker build -t lampvis ." to build and "docker run -it -t lampvis" to run
#
# These images don't work: ubuntu:18.04 ; nvidia/cuda:10.1-devel ; debian:buster

# CentOS 8 + CUDA stuff for ultimate use with GPU acceleration
FROM nvidia/cuda:11.0-devel-centos8
MAINTAINER Ian Bell "ian.bell@nist.gov"

RUN yum -y update && \
    yum -y install  wget zlib-devel \
                    make gcc gcc-c++ \
                    openmpi-devel fftw3-devel \
                    git nano python3 \
    && \
    yum -y clean all

ENV PATH="${PATH}:/usr/lib64/openmpi/bin/"

# ARG PACKAGES="asphere body class2 colloid compress coreshell dipole granular kspace manybody mc misc molecule opt peri qeq replica rigid shock snap srd user-reaxc"
ARG PACKAGES="opt molecule user-tally "
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

RUN python3 -m pip install pandas

# Copy over the run scripts
COPY do_*.lammps / 

# Build the chain-building executable
WORKDIR /lammps/tools
RUN gfortran -o chain chain.f && cp chain /

# Copy over the scripts
COPY chain_builder.py go.py /

# The task to run
WORKDIR /
CMD python3 go.py
