# Define image with OpenMDAO and Dymos

FROM ubuntu:24.04

SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Install updates
RUN apt-get update -y && apt-get -y install sudo vim curl wget git g++ gfortran make cmake graphviz libblas-dev liblapack-dev

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb ;\
    apt-get install -y ./google-chrome-stable_current_amd64.deb

# Create user
ENV USER=omdao
RUN adduser --shell /bin/bash --disabled-password ${USER}
RUN adduser ${USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chmod u+s /usr/bin/sudo

USER ${USER}
WORKDIR /home/${USER}

# Use the project environment and lock file for all Python dependencies.
COPY --from=ghcr.io/prefix-dev/pixi:0.82.1 /usr/local/bin/pixi /usr/local/bin/pixi
COPY --chown=omdao:omdao . /home/omdao/dymos
ENV PIXI_WORKSPACE_MANIFEST=/home/omdao/dymos/pyproject.toml
ENV PIXI_ENVIRONMENT_NAME=dev

RUN pixi install --locked -e dev --manifest-path "$PIXI_WORKSPACE_MANIFEST"

# Activate the same environment for interactive shells.
RUN echo 'eval "$(pixi shell-hook -e dev --manifest-path /home/omdao/dymos/pyproject.toml)"' >> ~/.bashrc

ENV PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
ENV OMPI_MCA_rmaps_base_oversubscribe=1
ENV OMPI_MCA_btl=^openib
ENV RDMAV_FORK_SAFE=true

# Set up a work directory that can be shared with the host operating system
WORKDIR /home/${USER}/work

ENTRYPOINT ["tail", "-f", "/dev/null"]
