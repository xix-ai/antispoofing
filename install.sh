#!/bin/bash

# this script installs everything required for training Entry Antispoofing Model

python3 -m venv ~/venv
. ~/venv/bin/activate


# this part installs the main dependency - VISSL
# main reference: https://github.com/facebookresearch/vissl/blob/main/INSTALL.md
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu102_pyt181/download.html apex

# clone vissl repository
cd $HOME && git clone --recursive https://github.com/facebookresearch/vissl.git && cd $HOME/vissl/
# Optional, checkout stable v0.1.6 branch. While our docs are versioned, the tutorials
# use v0.1.6 and the docs are more likely to be up-to-date.
git checkout v0.1.6
git checkout -b v0.1.6
# install vissl dependencies
pip install --progress-bar off -r requirements.txt
pip install opencv-python
# update classy vision install to commit stable for vissl.
# Note: If building from vissl main, use classyvision main.
pip uninstall -y classy_vision
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
# update fairscale install to commit stable for vissl.
pip uninstall -y fairscale
pip install fairscale==0.4.6
# install vissl dev mode (e stands for editable)
pip install -e ".[dev]"
# verify installation
python -c 'import vissl, apex'
