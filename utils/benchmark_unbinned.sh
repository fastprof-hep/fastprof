name=$1
python3 -m cProfile -o unbinned_$name.prof fastprof/testing/test_unbinned_gen.py
pyprof2calltree -k -i unbinned_$name.prof
