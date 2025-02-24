# Define image with OpenMDAO and Dymos

FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

# Install updates
RUN apt-get update -y && apt-get -y install wget git g++ gfortran make libblas-dev liblapack-dev

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb ;\
    apt-get install -y ./google-chrome-stable_current_amd64.deb

# Create user
ENV USER=omdao
RUN adduser --shell /bin/bash --disabled-password ${USER}
RUN adduser ${USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# Install Miniforge
RUN wget -q -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" ;\
    bash Miniforge3.sh -b ;\
    rm Miniforge3.sh ;\
    export PATH=$HOME/miniforge3/bin:$PATH ;\
    conda init bash

# Create conda environment
RUN source $HOME/miniforge3/etc/profile.d/conda.sh ;\
    #
    # Create conda environment
    #
    conda create -n mdaowork python=3.12 'numpy<2' scipy cython swig -y ;\
    conda activate mdaowork ;\
    conda install matplotlib graphviz -y ;\
    conda install mpi4py petsc4py=3.20 -y ;\
    python -m pip install pyparsing psutil objgraph plotly pyxdsm pydot ;\
    #
    # Install pyoptsparse
    #
    python -m pip install git+https://github.com/openmdao/build_pyoptsparse ;\
    build_pyoptsparse -v ;\
    #
    # Install OpenMDAO
    #
    git clone https://github.com/OpenMDAO/OpenMDAO.git ;\
    python -m pip install -e 'OpenMDAO[all]' ;\
    #
    # Install Dymos
    #
    git clone https://github.com/OpenMDAO/dymos.git ;\
    python -m pip install -e 'dymos[all]'

# Modify .bashrc
RUN echo "## Always activate mdaowork environment on startup ##" >> ~/.bashrc ;\
    echo "conda activate mdaowork" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc ;\
    echo "## OpenMPI settings" >> ~/.bashrc ;\
    echo "export OMPI_MCA_rmaps_base_oversubscribe=1" >> ~/.bashrc ;\
    echo "export OMPI_MCA_btl=^openib" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc ;\
    echo "## Required for some newer MPI / libfabric instances" >> ~/.bashrc ;\
    echo "export RDMAV_FORK_SAFE=true" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc

# Set up a work directory that can be shared with the host operating system
WORKDIR /home/${USER}/work

ENTRYPOINT ["tail", "-f", "/dev/null"]
