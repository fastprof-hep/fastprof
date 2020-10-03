name=$1
python3 -m cProfile -o benchmarks/fast_$name.prof testing/test_sample_gen.py
pyprof2calltree -k -i benchmarks/fast_$name.prof
