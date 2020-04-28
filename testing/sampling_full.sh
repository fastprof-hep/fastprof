./create_workspace.py -c ../../../../Stat/pyhf/fastprof/models/high_mass_gg.dat -o fastprof.root --poi mu --asimov

# Point at 1.3 GeV : Gaussian regime -- boost nominal signal from 1 to 100 events to ensure reasonable mus
./convert_ws.py -f fastprof.root --refit-asimov -b 1000:2000:20 --setval mX=1300,mu=1,nSignal0=100 -o high_mass_gg_1300.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1300,nSignal0=100,mu=0 -y 0.1,0.2,0.3,0.4,0.5,0.7,0.9 -o hypos_high_mass_gg_1300.json
python3 -i fastprof/examples/sampling_limits_simple.py

# Point at 1.7 GeV: Poisson regime
./convert_ws.py -f fastprof.root --refit-asimov -b 1000:2000:20 --setval mX=1700 -o high_mass_gg_1700.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1700,mu=0 -y 0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 -o hypos_high_mass_gg_1700.json
python3 -i fastprof/examples/sampling_limits_simple.py

