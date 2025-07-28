#!/bin/bash

# Exit script if any command fails
set -e

# Update pip and install openmim
echo "Updating pip and installing openmim..."
pip install -U pip
pip install -U openmim

# Install packages using MIM
echo "Installing packages with MIM..."
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install git+ssh://git@github.com/open-mmlab/mmdetection.git@v3.3.0

# Install various Python packages using pip
echo "Installing various Python packages..."
pip install supervision
pip install transformers==4.38.2
pip install nltk
pip install h5py
pip install einops
pip install seaborn
pip install fairscale
pip install git+ssh://git@github.com/openai/CLIP.git
pip install git+ssh://git@github.com/siyuanliii/TrackEval.git
pip install git+ssh://git@github.com/SysCV/tet.git#subdirectory=teta
pip install git+ssh://git@github.com/scalabel/scalabel.git@scalabel-evalAPI
pip install git+ssh://git@github.com/TAO-Dataset/tao
pip install git+ssh://git@github.com/lvis-dataset/lvis-api.git

echo "All packages installed successfully!"