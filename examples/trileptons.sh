# Create the base directory
mkdir trileptons
cd trileptons

# Install pyhf and fastprof
python3 -m venv pyhf
source pyhf/bin/activate
pip install pyhf
pip install fastprof

# Download an example pyhf model and unpack the archive
curl "https://www.hepdata.net/record/resource/2592329?view=true" -o trileptons.tar.gz
tar zxvf trileptons.tar.gz

# Patch the model to a specific signal model file
jsonpatch inclusive_bkgonly.json <(pyhf patchset extract inclusive_patchset.json --name "brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500") > brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json

# Convert the model to fastprof format
convert_pyhf.py brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json -o fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json

# Perform a likelihood scan over the mu_SIG POI
poi_scan.py -m fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json -y mu_SIG=-0.01:0.2:22+ -o mu_SIG -c 1 -v 1

# Plot the model -- first we need to reunite the 3 SRs, which are provided as separate bins in the pyhf model.
# The code below stitches together the different bins into a single range for each of the 3 SRs (SRFR, SR3l and SR4l)
merge_channels.py -m fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json -d fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500.json \
-o fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500_merged --obs-name mZl --obs-unit GeV -c "\
  SRFR=\
    SRFR_90_110_all_cuts:90:110,
    SRFR_110_130_all_cuts:110:130,
    SRFR_130_150_all_cuts:130:150,
    SRFR_150_170_all_cuts:150:170,
    SRFR_170_190_all_cuts:170:190,
    SRFR_190_210_all_cuts:190:210,
    SRFR_210_230_all_cuts:210:230,
    SRFR_230_250_all_cuts:230:250,
    SRFR_250_270_all_cuts:250:270,
    SRFR_270_300_all_cuts:270:300,
    SRFR_300_330_all_cuts:300:330,
    SRFR_330_360_all_cuts:330:360,
    SRFR_360_400_all_cuts:360:400,
    SRFR_400_440_all_cuts:400:440,
    SRFR_440_580_all_cuts:440:580,
    SRFR_580_inf_all_cuts:580:700~
  SR4l=\
    SR4l_90_110_all_cuts:90:110,
    SR4l_110_130_all_cuts:110:130,
    SR4l_130_150_all_cuts:130:150,
    SR4l_150_170_all_cuts:150:170,
    SR4l_170_190_all_cuts:170:190,
    SR4l_190_210_all_cuts:190:210,
    SR4l_210_230_all_cuts:210:230,
    SR4l_230_250_all_cuts:230:250,
    SR4l_250_270_all_cuts:250:270,
    SR4l_270_300_all_cuts:270:300,
    SR4l_300_330_all_cuts:300:330,
    SR4l_330_360_all_cuts:330:360,
    SR4l_360_400_all_cuts:360:400,
    SR4l_400_440_all_cuts:400:440,
    SR4l_440_580_all_cuts:440:580,
    SR4l_580_inf_all_cuts:580:700~
  SR3l=\
    SR3l_90_110_all_cuts:90:110,
    SR3l_110_130_all_cuts:110:130,
    SR3l_130_150_all_cuts:130:150,
    SR3l_150_170_all_cuts:150:170,
    SR3l_170_190_all_cuts:170:190,
    SR3l_190_210_all_cuts:190:210,
    SR3l_210_230_all_cuts:210:230,
    SR3l_230_250_all_cuts:230:250,
    SR3l_250_270_all_cuts:250:270,
    SR3l_270_300_all_cuts:270:300,
    SR3l_300_330_all_cuts:300:330,
    SR3l_330_360_all_cuts:330:360,
    SR3l_360_400_all_cuts:360:400,
    SR3l_400_440_all_cuts:400:440,
    SR3l_440_580_all_cuts:440:580,
    SR3l_580_inf_all_cuts:580:700"

# Now perform the actual plot
plot.py \
-m fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500_merged.json \
-d fastprof_brZ_60_brH_20_brW_20_bre_33_brm_33_brt_33_mass_500_merged_data.json \
--setval mu_SIG=0.05 --stack --log-scale --window 15x10 -o plot.pdf
    
