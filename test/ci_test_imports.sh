#!/usr/bin/env bash

# test that the various imports within the package works and can be reused across workflows

set -euo pipefail

python -c "import bilby"
python -c "import bilby.bilby_mcmc"
python -c "import bilby.core"
python -c "import bilby.core.prior"
python -c "import bilby.core.sampler"
python -c "import bilby.core.utils"
python -c "import bilby.gw"
python -c "import bilby.gw.detector"
python -c "import bilby.gw.eos"
python -c "import bilby.gw.likelihood"
python -c "import bilby.gw.sampler"
python -c "import bilby.hyper"
python -c "import cli_bilby"
python test/import_test.py
