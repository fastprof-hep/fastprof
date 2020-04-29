./create_workspace.py -c ../../../../Stat/pyhf/fastprof/models/high_mass_gg.dat -o fastprof.root --poi mu --asimov

# Point at 1.3 GeV : Gaussian regime -- boost nominal signal from 1 to 100 events to ensure reasonable mus
./convert_ws.py -f fastprof.root -u --data-name obsData -b 1000:2000:20 --setval mX=1300,mu=1,nSignal0=100 -o high_mass_gg_1300.json
./fit_ws.py -f fastprof.root --data-name obsData --setval mX=1300,nSignal0=100,mu=0 -r " -3,20" -y 0.1,0.2,0.3,0.4,0.5,0.7,0.9 -o fits_high_mass_gg_1300.json
./fastprof/utils/check_model.py -m run/high_mass_gg_1300.json -f run/fits_high_mass_gg_1300.json
./fastprof/utils/compute_limits.py -m run/high_mass_gg_1300.json -f run/fits_high_mass_gg_1300.json -o samples/high_mass_gg_1300 -n 10000

# Point at 1.3 GeV with 100 bins (needed for fit precision) (same hypo file as above)
# -- import and check workspace data
./convert_ws.py -f fastprof.root  -u --data-name obsData -b 1000:2000:100 --setval mX=1300,mu=1,nSignal0=100 -o high_mass_gg_1300-100bins.json
./fastprof/utils/check_model.py -m run/high_mass_gg_1300-100bins.json -f run/fits_high_mass_gg_1300.json
# -- import and check Asimov data
./convert_ws.py -f fastprof.root -x --asimov -b 1000:2000:100 --setval mu=0 -o high_mass_gg_1300-100bins_Asimov.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1300,nSignal0=100,mu=0 -r " -3,20" -y 0.1,0.2,0.3,0.4,0.5,0.7,0.9 -o fits_high_mass_gg_1300_Asimov.json
./fastprof/utils/check_model.py -m run/high_mass_gg_1300-100bins.json -d run/high_mass_gg_1300-100bins_Asimov.json -f run/fits_high_mass_gg_1300_Asimov.json
./fastprof/utils/compute_limits.py -m run/high_mass_gg_1300-100bins.json -f run/fits_high_mass_gg_1300.json -o samples/high_mass_gg_1300-100bins -n 10000

# Point at 1.7 GeV: Poisson regime
# -- import and check workspace data
./convert_ws.py -f fastprof.root -u --data-name obsData -b 1000:2000:100 --setval mX=1700 -o high_mass_gg_1700-100bins.json
./fit_ws.py -f fastprof.root --data-name obsData --setval mX=1700,mu=0 -r "0,20" -y 0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 -o fits_high_mass_gg_1700.json
./fastprof/utils/check_model.py -m run/high_mass_gg_1700-100bins.json --asimov 0 -f run/fits_high_mass_gg_1700.json
# -- import and check Asimov data
./convert_ws.py -f fastprof.root -x --asimov -b 1000:2000:100 --setval mu=0 -o high_mass_gg_1700-100bins-Asimov.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1700,mu=0 -r "0,20" -y 0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 -o fits_high_mass_gg_1700_Asimov.json
./fastprof/utils/check_model.py -m run/high_mass_gg_1700-100bins.json -d run/high_mass_gg_1700-100bins-Asimov.json -f run/fits_high_mass_gg_1700_Asimov.json
./fastprof/utils/compute_limits.py -m run/high_mass_gg_1700-100bins.json -f run/fits_high_mass_gg_1700.json -o samples/high_mass_gg_1700-100bins -n 10000

