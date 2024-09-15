#!/bin/bash

# Step 1: Create the local library directory for Python packages
mkdir -p ./local_lib/python3.8/site-packages

# Step 2: Export the PYTHONPATH and set the PYTHONPYCACHEPREFIX to the local library
export PYTHONPATH=$(pwd)/local_lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPYCACHEPREFIX=$(pwd)/local_lib/__pycache__

# Step 3: Install required Python packages to the local library directory
pip install -r requirements.txt --target=$(pwd)/local_lib/python3.8/site-packages

# Step 4: Add the local library directory to the PYTHONPATH and PYTHONPYCACHEPREFIX permanently
echo 'export PYTHONPATH=$(pwd)/local_lib/python3.8/site-packages:$PYTHONPATH' >> ~/.zshrc
echo 'export PYTHONPYCACHEPREFIX=$(pwd)/local_lib/__pycache__' >> ~/.zshrc

# Step 5: Reload the shell configuration
source ~/.zshrc

# Alias for flake8
alias norminette_python=flake8

echo "Setup complete. The environment has been configured to use locally installed Python packages and cache redirected."
