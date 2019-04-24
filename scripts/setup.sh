#!/bin/bash

apt-get update
apt-get --yes --force-yes install python-pip
apt-get --yes --force-yes install python-tk
pip install -U matplotlib==2.0
pip install statsmodels
pip install pandas
pip install scipy
pip install numpy
pip install tensorflow
pip install keras