# PM4Edu

The presence of GraphViz is required on the system: apt-get install graphviz (for Debian/Ubuntu).

You can create a venv:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
apt-get install graphviz (maybe it's required)

Optional:
export FLASK_APP=app.py FLASK_DEBUG=1 FLASK_ENV=development
export FLASK_DEBUG=1
export FLASK_ENV=development
