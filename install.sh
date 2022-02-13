#!/bin/bash

# 1. First check to see if the correct version of Python is installed on the local machine
echo "1. Checking Python version..."
REQ_PYTHON_V="390"

ACTUAL_PYTHON_V=$(python -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')
ACTUAL_PYTHON3_V=$(python3 -c 'import sys; version=sys.version_info[:3]; print("{0}{1}{2}".format(*version))')

if [[ $ACTUAL_PYTHON_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON_V == $REQ_PYTHON_V ]];  then
    PYTHON="python"
elif [[ $ACTUAL_PYTHON3_V > $REQ_PYTHON_V ]] || [[ $ACTUAL_PYTHON3_V == $REQ_PYTHON_V ]]; then
    PYTHON="python3"
else
    echo -e "\tPython 3.9 is not installed on this machine. Please install Python 3.9 before continuing."
    exit 1
fi

echo -e "\t--Python 3.9 is installed"

# 2. What OS are we running on?
PLATFORM=$($PYTHON -c 'import platform; print(platform.system())')

echo -e "2. Checking OS Platform..."
echo -e "\t--OS=Platform=$PLATFORM"

# 3. Create Virtual environment
echo -e "3. Creating new virtual environment..."

# Remove the env directory if it exists
if [[ -d gao-env ]]; then
    echo -e "\t--Virtual Environment already exists. Deleting old one now."
    rm -rf gao-env
fi

$PYTHON -m venv gao-env
if [[ ! -d gao-env ]]; then
    echo -e "\t--Could not create virtual environment...Please make sure venv is installed"
    exit 1
fi

# 4. Install Requirements

echo -e "4. Installing Requirements..."
if [[ ! -e "gao-g-requirements.txt" ]]; then
    echo -e "\t--Need gao-g-requirements.txt to install packages."
    exit 1
fi

source gao-env/bin/activate
pip install -r gao-g-requirements.txt