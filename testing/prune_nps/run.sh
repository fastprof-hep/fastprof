prune_model.py -m examples/high_stats_two_bins.json \
               -o high_stats_two_bins_lessNPs.json \
               -p np_bkg1=52,np_eff=-1
sed -e 's/^/@ /' high_stats_two_bins_lessNPs.json
echo
poi_scan.py -m high_stats_two_bins_lessNPs.json \
            -d examples/high_stats_two_bins.json \
            -y xs1=0:20:20 -o test_prune
