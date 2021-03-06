./create_workspace.py -c ../../../../Stat/pyhf/fastprof/models/high_mass_gg.dat -o fastprof.root --poi mu --asimov

# Point at 1.3 GeV : Gaussian regime -- boost nominal signal from 1 to 100 events to ensure reasonable mus
./convert_ws.py -f fastprof.root -u --data-name obsData -b 1000:2000:20 --setval mX=1300,mu=1,nSignal0=100 -o high_mass_gg_1300.json
./fit_ws.py -f fastprof.root --data-name obsData --setval mX=1300,nSignal0=100,mu=0 -r " -3,20" -y 0.1,0.2,0.3,0.4,0.5,0.7,0.9 -o fits_high_mass_gg_1300.json
./fastprof/utils/check_model.py -m run/fastprof/high_mass_gg_1300.json -f run/fastprof/fits_high_mass_gg_1300.json
./fastprof/utils/compute_limits.py -m run/fastprof/high_mass_gg_1300.json -f run/fastprof/fits_high_mass_gg_1300.json -o samples/high_mass_gg_1300 -n 10000

# Point at 1.3 GeV with 100 bins (needed for fit precision) (same hypo file as above)
# -- import and check workspace data
./convert_ws.py -f fastprof.root  -u --data-name obsData -b 1000:2000:100 --setval mX=1300,mu=1,nSignal0=100 -o high_mass_gg_1300-100bins.json
./fastprof/utils/check_model.py -m run/fastprof/high_mass_gg_1300-100bins.json -f run/fastprof/fits_high_mass_gg_1300.json
# -- import and check Asimov data
./convert_ws.py -f fastprof.root -x --asimov -b 1000:2000:100 --setval mu=0 -o high_mass_gg_1300-100bins_Asimov.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1300,nSignal0=100,mu=0 -r " -3,20" -y 0.1,0.2,0.3,0.4,0.5,0.7,0.9 -o fits_high_mass_gg_1300_Asimov.json
./fastprof/utils/check_model.py -m run/fastprof/high_mass_gg_1300-100bins.json -d run/fastprof/high_mass_gg_1300-100bins_Asimov.json -f run/fastprof/fits_high_mass_gg_1300_Asimov.json
./fastprof/utils/compute_limits.py -m run/fastprof/high_mass_gg_1300-100bins.json -f run/fastprof/fits_high_mass_gg_1300.json -o samples/high_mass_gg_1300-100bins -n 10000

# Point at 1.7 GeV: Poisson regime
# -- import and check workspace data
./convert_ws.py -f fastprof.root -u --data-name obsData -b 1000:2000:100 --setval mX=1700 -o high_mass_gg_1700-100bins.json
./fit_ws.py -f fastprof.root --data-name obsData --setval mX=1700,mu=0 -r "0,20" -y 0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 -o fits_high_mass_gg_1700.json
./fastprof/utils/check_model.py -m run/fastprof/high_mass_gg_1700-100bins.json --asimov 0 -f run/fastprof/fits_high_mass_gg_1700.json
# -- import and check Asimov data
./convert_ws.py -f fastprof.root -x --asimov -b 1000:2000:100 --setval mu=0 -o high_mass_gg_1700-100bins-Asimov.json
./fit_ws.py -f fastprof.root --asimov --setval mX=1700,mu=0 -r "0,20" -y 0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 -o fits_high_mass_gg_1700_Asimov.json
./fastprof/utils/check_model.py -m run/fastprof/high_mass_gg_1700-100bins.json -d run/fastprof/high_mass_gg_1700-100bins-Asimov.json -f run/fastprof/fits_high_mass_gg_1700_Asimov.json
./fastprof/utils/compute_limits.py -m run/fastprof/high_mass_gg_1700-100bins.json -f run/fastprof/fits_high_mass_gg_1700.json -o samples/high_mass_gg_1700-100bins -n 10000


./create_workspace.py -c ../datacards/LowHighMassRun2/hfitter_newResRun2_highMass_HiggsNW.dat -o HiggsNW.root --poi xs --asimov -i ..:../Hfitter/HfitterModels
./convert_ws.py -f HiggsNW.root -u --data-name obsData -b 150:2000:100:log  --setval mX=1700,xs=1 -o HiggsNW_1700-log100.json
./convert_ws.py -f HiggsNW.root -u --data-name obsData -b 150:2000:1000:log --setval mX=1700,xs=1 -o HiggsNW_1700-log1000.json
./convert_ws.py -f HiggsNW.root -u --data-name obsData -b 150:2000:2000:log --setval mX=1700,xs=1 -o HiggsNW_1700-log2000.json
./fit_ws.py -f HiggsNW.root --data-name obsData --setval mX=1700,xs=0 -r "0,20" -y 0.1,0.4,0.7,1,1.5,2,2.5,3,3.5,4 -o fits_HiggsNW_1700.json
# For the above, not in HGGfitter/run, otherwise class compilation fails somehow
./fastprof/utils/check_model.py -m run/fastprof/HiggsNW_1700-log100.json -f run/fastprof/fits_HiggsNW_1700.json
./fastprof/utils/check_model.py -m run/fastprof/HiggsNW_1700-log500.json -f run/fastprof/fits_HiggsNW_1700.json
./fastprof/utils/check_model.py -m run/fastprof/HiggsNW_1700-log1000.json -f run/fastprof/fits_HiggsNW_1700.json
./fastprof/utils/compute_limits.py -m run/fastprof/HiggsNW_1700-log100.json -f run/fastprof/fits_HiggsNW_1700.json -o samples/HiggsNW_1700-log100 -n 10000
./fastprof/utils/compute_limits.py -m run/fastprof/HiggsNW_1700-log500.json -f run/fastprof/fits_HiggsNW_1700.json -o samples/HiggsNW_1700-log500 -n 10000
./fastprof/utils/compute_limits.py -m run/fastprof/HiggsNW_1700-log1000.json -f run/fastprof/fits_HiggsNW_1700.json -o samples/HiggsNW_1700-log1000 -n 10000

./create_workspace.py -c ./fastprof/fidXsection_spin0_NW_withSS/hfitter_newResRun2_highMass_NW.dat -o fastprof/HighMass_NW.root --poi xs --asimov -i ..:../Hfitter/HfitterModels
# Use eps=4 to account for non-linear effect of dSig
./convert_ws.py -f HighMass_NW.root -u --asimov -b 150:2500:500:log --setval mX=1700,xs=1 -o HighMass_NW-1700-log500.json --eps 4
./fit_ws.py -f HighMass_NW.root --data-name obsData --setval mX=1700,xs=0 -r "0,1" -y 0.001,0.01,0.02,0.03,0.04,0.05,0.07,0.09,0.2 -o fits_HighMass_NW-1700.json              
# Use Asimov since the binning is comparable to the WS Asimov (but doesn't match, as it's log-scale) and this causes problems...
./fastprof/utils/check_model.py -m run/fastprof/HighMass_NW-1700-log500.json --asimov 0 -f run/fastprof/fits_HighMass_NW-1700.json         
./fastprof/utils/compute_limits.py -m  run/fastprof/HighMass_NW-1700-log500.json -f run/fastprof/fits_HighMass_NW-1700.json -o samples/HighMass_NW-1700-log500 -n 10000


./convert_ws.py -f HighMass_NW.root -u --asimov -b 150:2500:200:log --setval mX=1700,xs=1 --setconst dSig -o HighMass_NW-1700-log200-noDSig.json --validation-data HighMass_NW-1700-log200-noDSig-valid.json
./fit_ws.py -f HighMass_NW.root --data-name obsData --setval mX=1700,xs=0 --setconst dSig -r "0,1" -y 0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.2 -o fits_HighMass_NW-1700-noDSig.json
./fastprof/utils/check_model.py -m run/fastprof/HighMass_NW-1700-log200-noDSig.json --asimov 0 -f run/fastprof/fits_HighMass_NW-1700-noDSig.json
./fastprof/utils/compute_limits.py -m run/fastprof/HighMass_NW-1700-log200-noDSig.json -f run/fastprof/fits_HighMass_NW-1700-noDSig.json -o samples/HighMass_NW-1700-log200-noDSig -n 10000
./fastprof/utils/compute_limits.py -m run/fastprof/HighMass_NW-1700-log200-noDSig.json --asimov 0 -f run/fastprof/fits_HighMass_NW-1700-noDSig.json --regularize 3 -o samples/HighMass_NW-1700-log200-noDSig-r3 -n 10000
./fastprof/utils/compute_limits.py -m run/fastprof/HighMass_NW-1700-log200.json --asimov 0 -f run/fastprof/fits_HighMass_NW-1700-manual.json --regularize 5 -o samples/HighMass_NW-1700-log200-r5 -n 10000
./fastprof/utils/check_model.py -m run/fastprof/HighMass_NW-1700-log200.json --asimov 0 -f run/fastprof/fits_HighMass_NW-1700-manual.json -r 5

# in run/
./create_workspace.py -c fastprof/datacards/fidXsection_spin0_NW_withSS/hfitter_newResRun2_highMass_NW.dat --poi xs -f fastprof/ntup_data_all.root -i ..:../Hfitter/HfitterModels -o fastprof/highMass_NW.root
# in run/fastprof
# use python -u <cmd below> to write output in order (unbuffered) to a log file
python -u ./convert_ws.py -f highMass_NW.root --setval mX=$mass,xs=0 -r 0,10 -d obsData --refit 0 --binned --bkg dBkg -b 150:2500:200:log -o highMass_NW-$mass-log200.json
./convert_ws.py -f highMass_NW.root --bkg dBkg -b 150:2500:200:log -d obsData -x -o data_highMass-log200.json
./fit_ws.py -f highMass_NW.root -d obsData --binned --setval mX=1700 -y 13 -o fits_highMass_NW-1700.json
 # in fastprof/
./fastprof/utils/check_model.py -m run/fastprof/highMass_NW-1700-log200.json -d run/fastprof/data_highMass-log200.json -f run/fastprof/fits_highMass_NW-1700.json -r 5 
python -i ./fastprof/utils/plot_valid.py -m run/fastprof/highMass_NW-1700-log200.json -v run/fastprof/valid-highMass_NW-1700-log200.json -b 172
./fastprof/utils/compute_limits.py -m run/fastprof/highMass_NW-1700-log200.json -d run/fastprof/data_highMass-log200.json -f run/fastprof/fits_highMass_NW-1700.json -r 5 -o samples/highMass_NW-1700-log200-r5 -n 10000
python -i ./fastprof/utils/check_asymptotics.py -m run/fastprof/highMass_NW-$mass-log200.json -f run/fastprof/fits_highMass_NW-$mass.json -s samples/highMass_NW-$mass-log200-r1
python -i fastprof/utils/dump_debug.py samples/highMass_NW-2483-log200-r1_0.0671406_debug.csv -r --hypo run/fastprof/fits_highMass_NW-2483.json:11 -m run/fastprof/highMass_NW-2483-log200.json --log
split -d --lines 100 make_all_no_a0_17 make_all_no_a0_17_part

./convert_ws.py -f highMass_NW.root --bkg dBkg --refit 0 -d obsData --asimov -b 150:4000:250:log -x -o asimov_highMass-log250.json
./fastprof/utils/check_model.py -m models/highMass_NW-no-a0-17/model_highMass_NW-no-a0-17-log250-2939.json -d run/fastprof/data_highMass-log250.json -f fits/highMass_NW-no-a0-17/fits_highMass_NW-no-a0-17-2939.json -r 3 
fastprof/utils/iterate.py -p 150:2500:200:logint -c "source mk_inputs_no_a0_17 %"
fastprof/utils/iterate.py -p 150:2500:200:logint -c "sub_lim_no_a0_17 % 3" -o all_limits_no_a0_17
./fastprof/utils/compute_limits.py -m models/highMass_NW-no-a0-17/model_highMass_NW-no-a0-17-log250-2939.json -d run/fastprof/data_highMass-log250.json -f fits/highMass_NW-no-a0-17/fits_highMass_NW-no-a0-17-2939.json -r 3 -o samples/ref-2939/samples-2939 -n 10000 >&! samples/log-ref-2939
fastprof/utils/iterate.py -p 160:3000:500:logint -c "source mk_lyon_limit_no_a0_17 % 3" -o compute_lyon_limits-no_a0_17
python -i ./fastprof/utils/collect_results.py -p 160:3000:500:logint -i samples/highMass_NW-no-a0-17/highMass_NW-no-a0-17-obs_%-r3/samples/highMass_NW-no-a0-17-obs_%-r3_results.json -k limit_sampling_CLs -o lyon_limit_sampling_CLs.json --root-output lyon_limit_sampling_CLs.root

python -i ./fastprof/utils/compute_limits.py -m models/highMass_NW-no-a0-17/model_highMass_NW-no-a0-17-log250-2939.json -d run/fastprof/data_highMass-log250.json -f fits/highMass_NW-no-a0-17/fits_highMass_NW-no-a0-17-2939.json -o samples/highMass_NW-no-a0-17/highMass_NW-no-a0-17-obs_2939-r3/samples/highMass_NW-no-a0-17-obs_2939-r3 -n 10000 --truncate_dist 0.99 --bands 2

./convert_ws.py -x -f highMass_NW.root -d obsData -b 150:4000:250:log -o data_highMass-log250-new.json

python -i utils/plot_valid.py -m run/fastprof/models/highMass_NW-asym/model_highMass_NW-asym-log250-1164.json -v run/fastprof/models/highMass_NW-asym/val_highMass_NW-asym-log250-1164.json -b 156 -s Background

./iterate.py -p 160:3000:500:logint -c "./make_limits_batch.py %"
./collect_results.py -p 160:3000:500:logint -i batch/highMass_NW-prod1000_reg-obs_%/samples/highMass_NW-prod1000_reg-obs_%_results.json -k limit_asymptotics_CLs -o highMass_NW-prod1000-asymptotics_CLs.json --root-output highMass_NW-prod1000-asymptotics_CLs.root
 ./collect_results.py -p 160:3000:500:logint -i batch/highMass_NW-prod1000_reg-obs_%/samples/highMass_NW-prod1000_reg-obs_%_results.json -k limit_sampling_CLs -o highMass_NW-prod1000-sampling_CLs.json --root-output highMass_NW-prod1000-sampling_CLs.root
./collect_results.py -p 160:3000:500:logint -i batch/highMass_NW-prod1000_reg-obs_%/samples/highMass_NW-prod1000_reg-obs_%_results.json -k limit_sampling_CLs -o highMass_NW-prod1000-sampling_CLs_bands.json --root-output highMass_NW-prod1000-sampling_CLs_bands.root --bands 2
