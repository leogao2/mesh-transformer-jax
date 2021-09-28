#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'
screen -d -m python3 -c 'import time; time.sleep(999999999)'

# initializes jax and installs ray on cloud TPUs
sudo pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install --upgrade ray[default]==1.5.1 fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout

curl https://raw.githubusercontent.com/leogao2/ray/4db2c807f34ba86de18a994ce51730cf24898fea/python/ray/_private/services.py | sudo tee /usr/local/lib/python3.8/dist-packages/ray/_private/services.py
