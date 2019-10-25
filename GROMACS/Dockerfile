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

# Build and install cmake 3.x
RUN wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz && \
    tar zxvf cmake-3.* && \
    cd cmake-3.* && \
    ./bootstrap --prefix=/usr/local && \
    make -j$(nproc) && \
    make install

RUN cmake --version

#ENV PATH="${PATH}:/usr/lib64/openmpi/bin/"

## Build the GROMACS executable
WORKDIR /
# See http://manual.gromacs.org/documentation/2019/install-guide/index.html
RUN wget http://ftp.gromacs.org/pub/gromacs/gromacs-2019.tar.gz && \
  tar xfz gromacs-2019.tar.gz && \
  cd gromacs-2019 && \
  mkdir build && \
  cd build && \
  cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON && \
  make && \
  make check && \
  make install && \
  source /usr/local/gromacs/bin/GMXRC

# The task to run
#WORKDIR /
#CMD for fname in *.3d; do mpirun -np 12 --allow-run-as-root lammps -sf opt -in $fname; done
