reparam_model.py -m examples/high_stats_two_bins.json \
   -e "totalXX=linear_combination:xx1*2#xs2*1" \
   -n SR1:Signal:totalXX \
   -a xx1=0:0:20 -r xs1 \
   -o high_stats_two_bins_xx1.json
#poi_scan.py -m examples/high_stats_two_bins.json -y xs1=0:20:20 -o reparam_xs1 -x
sed -e 's/^/@ /' high_stats_two_bins_xx1.json
echo
poi_scan.py -m high_stats_two_bins_xx1.json -d examples/high_stats_two_bins.json -y xx1=0:20:20 -o reparam_xx1 -x

