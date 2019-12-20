#!/bin/bash
sudo apt install python3-pip
sudo apt install python3-tk
sudo apt install graphviz
pip3 install pipenv
pypath=$(pipenv --venv)
pwd > $pypath/lib/python3.5/site-packages/pyenv.pth
pipenv install
pipenv shell
