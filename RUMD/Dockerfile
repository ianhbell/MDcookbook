## Use "docker-compose up --build" to build and run

FROM nvidia/cuda:10.1-devel

RUN apt-get -y -m update && \
    apt-get install -y gcc g++ cmake git wget zlib1g-dev xz-utils python libboost-all-dev swig python-numpy python-numpy-dev subversion nano python-pip

RUN wget http://rumd.org/downloads/rumd_3.4.tar.xz && \
    tar xJvf rumd_3.4.tar.xz && \
    cd rumd-3.4 && \
    sed -i 's/# INCLUDE/INCLUDE/g' Makefile && \
    sed -i 's/# LDFLAGS/LDFLAGS/g' Makefile && \
    sed -i 's/61/75/g' Makefile && \
    make && \
    python setup.py install && \
    pip install pandas matplotlib scipy

# Add necessary paths
ENV PATH="/rumd-3.4/Tools:${PATH}"
ENV PYTHONPATH="/rumd-3.4/lib:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/rumd-3.4/lib:${LD_LIBRARY_PATH}"