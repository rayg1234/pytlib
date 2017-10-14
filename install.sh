#!/bin/bash



sudo apt install python-pip
sudo apt install python-tk
sudo pip install virtualenv
virtualenv pytenv
pwd > pytenv/lib/python2.7/site-packages/pyenv.pth
source pytenv/bin/activate
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install -r requirements.txt
