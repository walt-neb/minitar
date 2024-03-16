#!/bin/bash

# System-level dependencies (uncomment as necessary)
# sudo apt-get update
# sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
# sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
# sudo apt-get install -y libpq-dev  # Example for PostgreSQL

# Create a new virtual environment
deactivate #if a venv is already running..
python3.10 -m venv rl2
source rl2/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install TensorFlow, make sure to choose the correct version
pip install tensorflow==2.16.1

# Install TensorFlow Probability, compatible with the TensorFlow version
pip install tensorflow-probability

# Install TensorBoard
pip install tensorboard

# Install DM-Reverb, ensure compatibility with TensorFlow
pip install dm-reverb

# Install TF-Agents
pip install tf-agents[reverb]=0.18.0
pip install tf_keras==2.15.0
pip install pybullet


pip install 'imageio==2.4.0'
pip install ipython


#python3 -m pip install --upgrade tensorrt

# Other Python package installations (uncomment or add as necessary)
pip install numpy
pip install matplotlib
pip install gym
pip install pillow
pip install cloudpickle
pip install pygame
pip install gin-config

pip list

echo "Setup is complete. Activate your virtual environment with 'source rl-env/bin/activate'."
