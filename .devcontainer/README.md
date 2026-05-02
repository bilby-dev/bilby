# Bilby Development Containers

This directory contains [Dev Container](https://containers.dev/) configurations
for developing Bilby in a reproducible environment using VS Code or GitHub Codespaces.

## Available configurations

### 1. Core (default): `.devcontainer/devcontainer.json`

A lightweight Python 3.12 environment with bilby's core, sampler, and
optional requirements. Suitable for working on:

- Core likelihoods and priors
- Samplers (non-GW)
- Hyperparameter inference (`bilby.hyper`)
- Documentation and examples that don't require LALSuite

This is the default container that opens when you click **Reopen in Container**
or create a new Codespace.

### 2. GW (full): `.devcontainer/gw/devcontainer.json`

A heavier container that additionally installs the gravitational-wave
requirements: `lalsuite`, `astropy`, `gwpy`, `pyfftw`, and `scikit-learn`.
Also installs the underlying C libraries (`libfftw3-dev`, `libgsl-dev`,
`libhdf5-dev`) required to build LALSuite.

Use this when working on:

- `bilby.gw` (GW-specific priors, likelihoods, samplers)
- Waveform generation via LALSimulation
- GW data handling via GWpy

Note: this container is larger and takes longer to build due to LALSuite.
Requires at least 4 CPUs and 8 GB RAM.

## Usage

### VS Code (local)

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the bilby repository
3. Press `F1` → `Dev Containers: Reopen in Container`
4. To use the GW container instead: `F1` → `Dev Containers: Reopen in Container` → select `Bilby (with GW)`

### GitHub Codespaces

1. Navigate to the bilby repo on GitHub
2. Click the **Code** button → **Codespaces** tab → **Create codespace**
3. The default core container will be used. To select the GW container:
   - Use **Create codespace on main** → **Advanced options** → select configuration

## What's included

Both containers install:

- Python 3.12 with pip and build tools
- Git + GitHub CLI
- VS Code extensions: Python, Pylance, Ruff, Black, Jupyter, GitLens, autoDocstring
- Pytest configured for the `test/` directory

The GW container additionally installs:

- FFTW3, GSL, HDF5 system libraries
- LALSuite and gravitational-wave Python dependencies

## Contributing

If you find issues with the dev container setup or want to add more features
(e.g. pre-commit hooks, additional samplers), please open a PR or file an
issue referencing this README.
