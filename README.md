# PM4Edu

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)



PM4Edu is a Process Mining for Educactino tool.

## Install / Configuration / Using

Ubuntu (20.4) & Python 3.8.10;


If you need change python version:

sudo apt install python3.8

sudo ln -sf /usr/bin/python3.8 /usr/bin/python3

and (just in case):

rm -rf venv

apt install python3.8-venv

python3.8 -m venv venv

So...

You can create a venv:

python3 -m venv venv

source venv/bin/activate

pip install --upgrade

pip install -r requirements.txt

apt-get install graphviz (required)


Optional:

export FLASK\_APP=app.py

export FLASK\_DEBUG=1

export FLASK\_ENV=development
