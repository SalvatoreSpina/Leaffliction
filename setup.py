import subprocess
import sys
import os

def install_requirements(requirements_file, install_folder):
    # Check if virtualenv is installed, if not, install it
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'virtualenv'])
    except subprocess.CalledProcessError:
        print("Failed to install virtualenv.")
        sys.exit(1)

    # Create the virtual environment in the install folder
    venv_dir = os.path.join(install_folder, "venv")
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, '-m', 'virtualenv', venv_dir])

    # Activate the virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
    else:  # Unix/Linux/Mac
        activate_script = os.path.join(venv_dir, 'bin', 'activate')

    # Run pip install in the virtual environment
    install_command = f'source {activate_script} && pip install -r {requirements_file}'
    
    try:
        subprocess.check_call(install_command, shell=True, executable="/bin/bash")
        print("Requirements installed successfully in the isolated environment.")
    except subprocess.CalledProcessError:
        print("Failed to install requirements.")
        sys.exit(1)

if __name__ == '__main__':
    requirements_file = 'requirements.txt'  # Path to the requirements.txt file
    install_folder = 'local_packages'  # Folder where you want to install the environment

    if not os.path.exists(install_folder):
        os.makedirs(install_folder)

    install_requirements(requirements_file, install_folder)

    # Then, to activate the virtual environment and use the installed packages:
    # source local_packages/venv/bin/activate

