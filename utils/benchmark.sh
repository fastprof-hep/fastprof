name=bounded
python3 -m cProfile -o fast_$name.prof fastprof/testing/test_sample_gen.py
pyprof2calltree -k -i fast_$name.prof
