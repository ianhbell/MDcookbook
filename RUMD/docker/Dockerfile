FROM nvidia/cuda:10.1-devel

RUN apt-get -y -m update && \
    apt-get install -y gcc g++ cmake git wget zlib1g-dev xz-utils python3.7 python3.7-dev libboost-all-dev swig subversion nano python3-pip python3-numpy-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -m pip install numpy

RUN wget http://rumd.org/downloads/rumd_3.5.tar.xz && \
    tar xJvf rumd_3.5.tar.xz && \
    cd rumd-3.5 && \
    sed -i 's/# INCLUDE/INCLUDE/g' Makefile && \
    sed -i 's/# LDFLAGS/LDFLAGS/g' Makefile && \
    sed -i 's/61/75/g' Makefile && \
    make

WORKDIR /rumd-3.5
COPY setup.py /rumd-3.5
RUN python3 setup.py install && \
    apt-get -y remove python3-numpy && \
    python3 -m pip install numpy pandas matplotlib scipy && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
