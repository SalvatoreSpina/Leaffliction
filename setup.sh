#!/bin/bash

# Step 1: Create a directory for the local packages
mkdir -p ./local_python_packages

# Step 2: Install the required package (e.g., numpy) in the local directory
# Replace 'numpy' with the package you need to install
pip install --target=./local_python_packages -r requirements.txt

# Step 3: Inform the user to add the local packages directory to their PYTHONPATH
echo "The package has been installed in ./local_python_packages."
echo "To use this package, add the following line to your shell configuration or run it manually:"
echo "export PYTHONPATH=$(pwd)/local_python_packages:\$PYTHONPATH"
