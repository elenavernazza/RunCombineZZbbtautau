# Non resonant 2018 split in 3 categories (res1B, etc) prod_240528 *with ZH combination*



#  ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,
#ZHKinFit_mass

############ ZbbHtt # 
python3 RunNonResLimits.py --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
    --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_500 --prd prod_240805 --grp datacard --num 9ZH # --move_eos --user_eos tcuisset



# Impacts
python3 ../RunNonResLimits.py --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12  \
  --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet   \
  --feat dnn_ZHbbtt_kl_500 --prd prod_240528_nonres --grp datacard --no_run_one --no_run_ch --no_run_cat --run_year --no_run_impacts --run_impacts_noMCStat

############## ZttHbb
python3 RunNonResLimits.py --ver ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
    --cat cat_ZttHbb_orthogonal_cut_90_resolved_1b,cat_ZttHbb_orthogonal_cut_90_resolved_2b,cat_ZttHbb_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_500 --prd prod_240805 --grp datacard --num 9ZH

# Impacts
python3 ../RunNonResLimits.py --ver ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12  \
  --cat cat_ZttHbb_orthogonal_cut_90_resolved_1b,cat_ZttHbb_orthogonal_cut_90_resolved_2b,cat_ZttHbb_orthogonal_cut_90_boosted_noPNet   \
  --feat dnn_ZHbbtt_kl_500 --prd prod_240528_nonres --grp datacard --no_run_one --no_run_ch --no_run_cat --run_zh_comb_year --run_year --no_run_impacts --run_impacts_noMCStat

############## Combination # ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12  , ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12
python3 RunNonResLimits.py --ver ul_2016_ALL_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12,ul_2016_ALL_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12  \
  --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet   \
  --feat dnn_ZHbbtt_kl_500 --prd prod_240805 --grp datacard --num 9ZH --no_run_cp --no_run_one --no_run_ch --no_run_cat --run_zh_comb_cat --no_run_year --run_zh_comb_year --no_comb_2016 --no_run_impacts --no_run_impacts_noMCStat

# Impacts
python3 ../RunNonResLimits.py --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12,ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12  \
  --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet   \
  --feat dnn_ZHbbtt_kl_1 --prd prod_240528_nonres --grp datacard --no_run_one --no_run_ch --no_run_cat --no_run_zh_comb_cat --no_run_year --run_zh_comb_year --no_run_impacts --run_impacts_noMCStat

#########################
####################### Move to EOS
# we use all categories (Zbb&Ztt), this will fail for some but it is expected
python3 ../RunNonResLimits.py --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12,ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12  \
    --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet,cat_ZttHbb_orthogonal_cut_90_resolved_1b,cat_ZttHbb_orthogonal_cut_90_resolved_2b,cat_ZttHbb_orthogonal_cut_90_boosted_noPNet  \
    --feat dnn_ZHbbtt_kl_1 --prd prod_240528_nonres --grp datacard --no_run --no_run_one --no_run_ch --no_run_cat --no_run_year --run_zh_comb_year --plot_only --move_eos --user_eos tcuisset


rsync -rltv /grid_mnt/data__data.polcms/cms/cuisset/ZHbbtautau/combine/RunCombineZZbbtautau/NonRes tcuisset@lxplus.cern.ch:/eos/user/t/tcuisset/www/zzbbtt/CombineLimits --delete --delete-excluded \
 --filter '- impacts/' --filter '+ */' \
 --filter '+ **.png' --filter '+ **.pdf' --filter '+ **/Significance.log' --filter '+ **/*_datacard_*.txt' --filter '- *' 



#########################
####################### Resonant limits
cd ../CMSSW_11_3_4/ && cmsenv && cd - && ulimit -s unlimited && cd ResLimits_v10e/

python3 ../RunAsymptoticLimits.py --ver ul_2016_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
    --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_1 --featureDependsOnMass --prd prod_240528 --grp datacard_res \
    --mass 600,800,1000,1200,1400,2000,3000,3500,4000 --no_run_year

# ZbbHtt
python3 ../RunAsymptoticLimits.py --ver ul_2016_HIPM_ZbbHtt_v12,ul_2016_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
    --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_1 --featureDependsOnMass --prd prod_240528 --grp datacard_res \
    --mass 600,800,1000,1200,1400,2000,3000,3500,4000 

# ZttHbb
python3 ../RunAsymptoticLimits.py --ver ul_2016_HIPM_ZttHbb_v12,ul_2016_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
    --cat cat_ZttHbb_orthogonal_cut_90_resolved_1b,cat_ZttHbb_orthogonal_cut_90_resolved_2b,cat_ZttHbb_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_1 --featureDependsOnMass --prd prod_240528 --grp datacard_res \
    --mass 600,800,1000,1200,1400,2000,3000,3500,4000  

# ZH combination
python3 ../RunAsymptoticLimits.py --ver ul_2018_ZbbHtt_v12,ul_2018_ZttHbb_v12,ul_2017_ZbbHtt_v12,ul_2017_ZttHbb_v12,ul_2016_ZbbHtt_v12,ul_2016_ZttHbb_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2016_HIPM_ZttHbb_v12 \
    --cat cat_ZbbHtt_orthogonal_cut_90_resolved_1b,cat_ZbbHtt_orthogonal_cut_90_resolved_2b,cat_ZbbHtt_orthogonal_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_1 --featureDependsOnMass --prd prod_240528 --grp datacard_res \
    --mass 600,800,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000 --no_run_one --no_run_ch --no_run_cat --run_zh_comb_cat --run_zh_comb_year

#     --move_eos --user_eos cuisset



# impacts
mkdir /grid_mnt/data__data.polcms/cms/cuisset/ZHbbtautau/combine/RunCombineZZbbtautau/ResLimits_v10d/Res/FullRun2_ZbbHtt/prod_240528/dnn_ZHbbtt_kl_1/Impacts
cd /grid_mnt/data__data.polcms/cms/cuisset/ZHbbtautau/combine/RunCombineZZbbtautau/ResLimits_v10d/Res/FullRun2_ZbbHtt/prod_240528/dnn_ZHbbtt_kl_1/Impacts
cp /grid_mnt/data__data.polcms/cms/cuisset/ZHbbtautau/combine/RunCombineZZbbtautau/ResLimits_v10d/Res/FullRun2_ZbbHtt/prod_240528/dnn_ZHbbtt_kl_1/M800/FullRun2_ZbbHtt_dnn_ZHbbtt_kl_1_os_iso.txt .

text2workspace.py FullRun2_*_os_iso.txt -o model.root
combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 --setParameterRanges r=0,2 --doInitialFit --robustFit 1 --exclude 'rgx{prop_bin.+}'
combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 --setParameterRanges r=0,2 --doFits --robustFit 1 --parallel 50 --exclude 'rgx{prop_bin.+}'
combineTool.py -M Impacts -d model.root -m 125 -o impacts.json
plotImpacts.py -i impacts.json -o impacts
mkdir impacts_root
mv higgsCombine_paramFit* higgsCombine_initialFit* impacts_root

