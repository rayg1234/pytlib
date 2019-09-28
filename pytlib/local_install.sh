#!/bin/bash
sudo apt install python3-pip
sudo apt install python3-tk
sudo apt install graphviz
sudo pip3 install virtualenv
virtualenv pytenv
# fix this to use the actual python version
pwd > pytenv/lib/python3.5/site-packages/pyenv.pth
source pytenv/bin/activate
pip3 install Cython
pip3 install -r requirements.txt
