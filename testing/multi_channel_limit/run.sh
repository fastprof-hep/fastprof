compute_fast_limits.py \
   -m examples/multi_channel.json \
   -y "xs_BSM=0.1|xs_BSM=0.5|xs_BSM=1|xs_BSM=1.5|xs_BSM=2|xs_BSM=4" \
   -n 2000 -s 131071 -o multi_channel_limit -b 2 --clsb --show-timing
