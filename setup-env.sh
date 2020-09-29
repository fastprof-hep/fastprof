python3 -m venv env
source env/bin/activate
if [ ! -e build ]; then python setup.py install; fi
