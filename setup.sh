python3 -m venv env
source env/bin/activate
if [ ! -e build ]; then 
  pip install .; 
  cd doc; make doc; make html; cd ..
fi
disable -p '#'

