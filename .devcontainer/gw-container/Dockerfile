FROM continuumio/anaconda3

RUN conda install -c conda-forge bilby gwpy python-lalsimulation pymultinest


WORKDIR /workspace

COPY ../../. /workspace