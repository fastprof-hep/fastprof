rm -rf fastprof
mkdir -p fastprof/src fastprof/tests
cp -rf ../fastprof ../fastprof_utils ../fastprof_import fastprof/src
cp -f pyproject.toml fastprof
cp -f setup.cfg fastprof
cp -f setup.py fastprof
cp -f ../LICENSE fastprof
cp -f ../LICENSE fastprof
cp -f ../README.md fastprof

cd fastprof
python3 -m build

