#!/bin/bash



sudo apt install pip
sudo apt install python-tk
sudo pip install virtualenv
virtualenv pytenv
pwd > pytenv/lib/python2.7/site-packages/pyenv.pth
source pytenv/bin/activate
pip install -r requirements.txt

