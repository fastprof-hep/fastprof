python3 -m venv env --prompt fastprof
source env/bin/activate
pip install --upgrade pip | grep -v 'already satisfied'
if [ ! -e build ]; then 
  pip install .; 
  if [[ ! -v NO_DOC ]]; then
    cd doc; make doc; make html; cd ..
  fi
fi

