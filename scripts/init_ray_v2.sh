#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'
screen -d -m python3 -c 'import time; time.sleep(999999999)'

# initializes jax and installs ray on cloud TPUs
sudo pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install --upgrade ray[default]==1.5.1 fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout

sudo apt install curl gnupg
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel

sudo git clone https://github.com/leogao2/ray || echo already cloned
cd ray
sudo git checkout no_ver
cd python
sudo python3 setup.py install
