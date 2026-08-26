#############################
######## Setup base #########
#############################

FROM containers.ligo.org/docker/base:conda AS os-deps
LABEL name="bilby CI testing" \
maintainer="Gregory Ashton <gregory.ashton@ligo.org>, Colm Talbot <colm.talbot@ligo.org>"

#############################
##### Download ROQ data #####
#############################

FROM debian:bookworm-slim AS data
WORKDIR /roq_basis
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*
# Add the ROQ data to the image
RUN  wget https://git.ligo.org/lscsoft/ROQ_data/raw/master/IMRPhenomPv2/4s/B_linear.npy \
    && wget https://git.ligo.org/lscsoft/ROQ_data/raw/master/IMRPhenomPv2/4s/B_quadratic.npy \
    && wget https://git.ligo.org/lscsoft/ROQ_data/raw/master/IMRPhenomPv2/4s/fnodes_linear.npy \
    && wget https://git.ligo.org/lscsoft/ROQ_data/raw/master/IMRPhenomPv2/4s/fnodes_quadratic.npy \
    && wget https://git.ligo.org/lscsoft/ROQ_data/raw/master/IMRPhenomPv2/4s/params.dat \
    && wget https://git.ligo.org/soichiro.morisaki/roq_basis/raw/main/IMRPhenomD/16s_nospins/basis_addcal.hdf5 \
    && wget https://git.ligo.org/soichiro.morisaki/roq_basis/raw/main/IMRPhenomD/16s_nospins/basis_multiband_addcal.hdf5

#############################
##### Build environment #####
#############################

FROM os-deps AS build-env
WORKDIR /bldenv
ARG ENV_FILE=environment.yml
COPY ${ENV_FILE} env.yml
ARG python_major_version=3
ARG python_minor_version=12

RUN echo "  - python=${python_major_version}.${python_minor_version}" >> env.yml
ENV conda_env=python${python_major_version}${python_minor_version}

RUN mamba env create -f env.yml -n ${conda_env}
RUN echo "source activate ${conda_env}" > ~/.bashrc
ENV PATH=/opt/conda/envs/${conda_env}/bin:$PATH
RUN /bin/bash -c "source activate ${conda_env}"
RUN cd / && rm -rf /bldenv
RUN mamba clean --all

#############################
### Finalize environment ####
#############################

FROM build-env AS final
COPY --from=data /roq_basis /roq_basis
# otherwise CI jobs don't like installing bilby
ENV SETUPTOOLS_SCM_IGNORE_DUBIOUS_OWNER=1
RUN mamba info
RUN python --version
