#!/bin/bash
#sudo apt install python3-pip
sudo apt install python3-tk
sudo apt install graphviz
pip3 install pipenv
# fix this to use the actual python version
pypath=$(pipenv --venv)
pwd > $pypath/lib/python3.5/site-packages/pyenv.pth
pipenv shell
pipenv install
#pip3 install Cython
#pip3 install -r requirements.txt
