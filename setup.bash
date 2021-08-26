#!/bin/bash -e

SCRIPT_DIR=$(dirname "$0")

cd ${SCRIPT_DIR}

python3 -m venv python3_env
python3_env/bin/pip3 install -r requirements_3.txt

virtualenv python2_env
python2_env/bin/pip2 install -r requirements_2.txt

git submodule init
git submodule update
