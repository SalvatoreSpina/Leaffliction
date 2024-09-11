#!/bin/bash

# Step 1: Create the local library directory for Python packages
mkdir -p ./local_lib/python3.8/site-packages

# Step 2: Export the PYTHONPATH
export PYTHONPATH=$(pwd)/local_lib/python3.8/site-packages:$PYTHONPATH

# Step 3: Install required Python packages to the local library directory
pip install --target=$(pwd)/local_lib/python3.8/site-packages numpy \
    torch torchvision pandas matplotlib pillow \
    flake8 mizani==0.9.2 plantcv==3.14.3 \
    opencv-python-headless \
    tensorflow

# Step 4: Add the local library directory to the PYTHONPATH permanently
echo 'export PYTHONPATH=$(pwd)/local_lib/python3.8/site-packages:$PYTHONPATH' >> ~/.zshrc

# Step 5: Reload the shell configuration
source ~/.zshrc

alias norminette_python=flake8

echo "Setup complete. The environment has been configured to use locally installed Python packages."
