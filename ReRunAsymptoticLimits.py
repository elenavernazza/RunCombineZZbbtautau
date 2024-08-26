import os,json,concurrent.futures,itertools
import pdb, csv, subprocess
import numpy as np

import warnings, logging
warnings.filterwarnings("ignore", message=".*Type 3 font.*")
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def run_cmd(cmd, run=True, check=True):
    if run:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            if check:
                raise RuntimeError(f"Command {cmd} failed with exit code {e.returncode}. Working directory : {os.getcwd()}") from None
            else:
                print(f"## ERROR : Command {cmd} failed with exit code {e.returncode}. Working directory : {os.getcwd()}")

if __name__ == "__main__" :

    import matplotlib,argparse
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser("RunAsymptoticLimits")
    
    def makeFlag(arg_name, **kwargs):
        if arg_name.startswith("--"):
            arg_name = arg_name[2:]
        parser.add_argument(f"--{arg_name}", action='store_true', **kwargs)
        parser.add_argument(f"--no_{arg_name}", action='store_false', **kwargs)
    
    parser.add_argument("--ver",          dest="ver",                   default='')
    parser.add_argument("--cat",          dest="cat",                   default='')
    parser.add_argument("--prd",          dest="prd",                   default='')
    parser.add_argument("--feat",         dest="feat",                  default='dnn_ZZbbtt_kl_1')
    parser.add_argument("--grp",          dest="grp",                   default='datacard_zz')
    parser.add_argument("--channels",     dest="channels",              default="etau,mutau,tautau")
    parser.add_argument("--num",          dest="num",                   default='',               help='Assign number to output directory for versioning')
    parser.add_argument("--user_eos",     dest="user_eos",              default='evernazz',       help='User Name for lxplus account')
    parser.add_argument("--user_cmt",     dest="user_cmt",              default='vernazza',       help='User Name for cmt folder')
    parser.add_argument("--mass",         dest="mass",                  default='200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,3000')
    makeFlag("--run",                     dest="run",                   default=True,             help='Run commands or do a dry-run')
    makeFlag("--comb_2016",               dest="comb_2016",             default=True,             help='Combine 2016 and 2016_HIPM')
    makeFlag("--run_cp",                  dest="run_cp",                default=True,             help='Run copy of datacards')
    makeFlag("--run_one",                 dest="run_one",               default=True,             help='Run each channel or not')
    makeFlag("--run_ch",                  dest="run_ch",                default=True,             help='Combine channels or not')
    makeFlag("--run_cat",                 dest="run_cat",               default=True,             help='Combine categories or not')
    makeFlag("--run_zh_comb_cat",         dest="run_zh_comb_cat",       default=False,            help='Run ZbbHtt & ZttHbb combination for each year separately')
    makeFlag("--run_year",                dest="run_year",              default=True,             help='Combine years or not')
    makeFlag("--run_zh_comb_year",        dest="run_zh_comb_year",      default=False,            help='Run ZbbHtt & ZttHbb combination for Full Run2')
    makeFlag("--run_impacts",             dest="run_impacts",           default=True,             help='Make impact plots')
    makeFlag("--run_impacts_noMCStat",    dest="run_impacts_noMCStat",  default=False,            help='Make impact plots, but only without MC stat uncertainties (faster)')
    makeFlag("--plot_only",               dest="plot_only",             default=False,            help='Skip all combine commands and plot only')
    makeFlag("--singleThread",            dest="singleThread",          default=False,            help="Don't run in parallel, disable for debugging")
    makeFlag("--featureDependsOnMass",    dest='featureDependsOnMass',  default=False,            help="Add _$MASS to name of feature for each mass for parametrized DNN")
    makeFlag("--unblind",                 dest="unblind",               default=False,            help="Pick the unblinded datacards and run unblinded limits")
    options = parser.parse_args()

    comb_options = ' --rAbsAcc=0.05 --rMin 0 --rMax 100000000 '

    if ',' in options.ver:  versions = options.ver.split(',')
    else:                   versions = [options.ver]
    
    if ',' in options.cat:  categories = options.cat.split(',')
    else:                   categories = [options.cat]

    if ',' in options.feat: features = options.feat.split(',')
    else:                   features = [options.feat]

    if ',' in options.channels: channels = options.channels.split(',')
    else:                       channels = [options.channels]

    if ',' in options.mass:     mass_points = options.mass.split(',')
    else:                       mass_points = [options.mass]

    prd = options.prd
    grp = options.grp
    run = int(options.run) == 1
    run_one = options.run_one
    run_cp = options.run_cp
    run_ch = options.run_ch
    run_cat = options.run_cat
    run_year = options.run_year
    featureDependsOnMass = options.featureDependsOnMass
    comb_2016 = options.comb_2016
    unblind = options.unblind

    cmtdir = '/data_CMS/cms/' + options.user_cmt + '/cmt/CreateDatacards/'
    maindir = os.getcwd() + f'/Res{options.num}/'

    if "ZZ" in options.ver:       o_name = 'ZZbbtt'; process_tex = r"$X\rightarrow ZZ\rightarrow bb\tau\tau$";   x_axis = r"$m_{X}$ [GeV]"
    elif "ZbbHtt" in options.ver: o_name = 'ZbbHtt'; process_tex = r"$Z'\rightarrow ZH$";  x_axis = r"$m_{Z'}$ [GeV]" # \rightarrow bb\tau\tau
    elif "ZttHbb" in options.ver: o_name = 'ZttHbb'; process_tex = r"$Z'\rightarrow ZH$"; x_axis = r"$m_{Z'}$ [GeV]" # \rightarrow \tau\tau bb
    
    if comb_2016:

        if any("2016_Z" in s for s in versions) and any("2016_HIPM_Z" in s for s in versions):
        
            prefix = "ul_"
            suffix = "_Z" + versions[0].split("_Z")[1]
            v_2016 = prefix + "2016" + suffix
            v_2016_HIPM = prefix + "2016_HIPM" + suffix
            v_combined = prefix + "2016_ALL" + suffix
            versions.remove(v_2016) 
            versions.remove(v_2016_HIPM)
            versions.insert(0, v_combined)

    def rerun_single_limit(maindir, feature, version, prd, category, mass, channel, featureDependsOnMass):

        if featureDependsOnMass: feat_name = f'{feature}_{mass}'
        else:                    feat_name = f'{feature}'

        odir = maindir + f'/{version}/{prd}/{feature}/{category}/{channel}/M{mass}'
        with open(odir + '/limits.json', 'r') as f:
            data = json.load(f)

            if data[mass]["exp"] != -1:
                return True
            else:
                run_cmd('mkdir -p ' + odir)
                datafile = odir + f'/{version}_{category}_{feat_name}_{grp}_{channel}_os_iso.txt'
                if not os.path.isfile(datafile):
                    print("#### WARNING : could not find datacard " + datafile + ", ignoring")
                    return

                print(f" ### INFO: Re-running {odir}")
                cmd = f'cd {odir} && combine -M AsymptoticLimits {datafile} --run blind --noFitAsimov {comb_options} &> combine.log'
                run_cmd(cmd, run)

    if run_one:

            if options.singleThread:
                for feature in features:
                    for version in versions:
                        for category in categories:
                            for mass in mass_points:
                                for channel in channels:
                                    rerun_single_limit(maindir, feature, version, prd, category, mass, channel, featureDependsOnMass)

            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=50) as exc:
                    futures = []
                    for feature in features:
                        for version in versions:
                            for category in categories:
                                for mass in mass_points:
                                    for channel in channels:
                                        futures.append(exc.submit(rerun_single_limit, \
                                            maindir, feature, version, prd, category, mass, channel, featureDependsOnMass))
                    for res in concurrent.futures.as_completed(futures):
                        try:
                            res.result()
                        except RuntimeError as e:
                            print("#### ERROR : " + str(e))

    def rerun_comb_channels(maindir, cmtdir, feature, version, prd, category, mass, featureDependsOnMass):

        if featureDependsOnMass: feat_name = f'{feature}_{mass}'
        else:                    feat_name = f'{feature}'

        combdir = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/M{mass}'
        with open(combdir + '/limits.json', 'r') as f:
            data = json.load(f)

            if data[mass]["exp"] != -1:
                return True
            else:
                print(" ### INFO: Re-running ", combdir)
                run_cmd('mkdir -p ' + combdir, run)
                cmd = f'combine -M AsymptoticLimits {version}_{feature}_{category}_os_iso.txt --run blind --noFitAsimov {comb_options} &> combine.log'
                if run: os.chdir(combdir)
                run_cmd(cmd, run)

    if run_ch:

        ############################################################################
        print("\n ### INFO: Run Combination of channels \n")
        ############################################################################

        if options.singleThread:
            for feature in features:
                for version in versions:
                    for category in categories:
                        for mass in mass_points:
                            rerun_comb_channels(maindir, cmtdir, feature, version, prd, category, mass, featureDependsOnMass)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                futures = []
                for feature in features:
                    for version in versions:
                        for category in categories:
                            for mass in mass_points:
                                futures.append(exc.submit(rerun_comb_channels, maindir, cmtdir, feature, version, prd, category, mass, featureDependsOnMass))
                for res in concurrent.futures.as_completed(futures):
                    res.result()

    def rerun_comb_categories(maindir, cmtdir, feature, version, prd, mass, featureDependsOnMass):

        if featureDependsOnMass:    feat_name = f'{feature}_{mass}'
        else:                       feat_name = f'{feature}'

        combdir = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/M{mass}'
        with open(combdir + '/limits.json', 'r') as f:
            data = json.load(f)
            # breakpoint()

            if data[mass]["exp"] != -1:
                return True
            else:
                print(" ### INFO: Re-running ", combdir)
                run_cmd('mkdir -p ' + combdir, run)
                cmd = f'combine -M AsymptoticLimits {version}_{feature}_os_iso.txt --run blind --noFitAsimov {comb_options} &> combine.log'
                if run: os.chdir(combdir)
                run_cmd(cmd, run)

    if run_cat:

        ############################################################################
        print("\n ### INFO: Run Combination of categories \n")
        ############################################################################

        if options.singleThread:
            for feature in features:
                for version in versions:
                    for mass in mass_points:
                        rerun_comb_categories(maindir, cmtdir, feature, version, prd, mass, featureDependsOnMass)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                futures = []
                for feature in features:
                    for version in versions:
                        for mass in mass_points:
                            futures.append(exc.submit(rerun_comb_categories, maindir, cmtdir, feature, version, prd, mass, featureDependsOnMass))
                for res in concurrent.futures.as_completed(futures):
                    res.result()

    def rerun_comb_years(maindir, cmtdir, feature, prd, mass, featureDependsOnMass):

        if featureDependsOnMass:    feat_name = f'{feature}_{mass}'
        else:                       feat_name = f'{feature}'

        combdir = maindir + f'/FullRun2_{o_name}/{prd}/{feature}/M{mass}'
        with open(combdir + '/limits.json', 'r') as f:
            data = json.load(f)

            if data[mass]["exp"] != -1:
                return True
            else:
                print(" ### INFO: Re-running ", combdir)
                run_cmd('mkdir -p ' + combdir, run)
                cmd = f'combine -M AsymptoticLimits FullRun2_{o_name}_{feature}_os_iso.txt --run blind --noFitAsimov {comb_options} &> combine.log'
                if run: os.chdir(combdir)
                run_cmd(cmd, run)

    if run_year:

        ############################################################################
        print("\n ### INFO: Run Combination of years \n")
        ############################################################################

        if options.singleThread:
            for feature in features:
                for mass in mass_points:
                    rerun_comb_years(maindir, cmtdir, feature, prd, mass, featureDependsOnMass)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                futures = []
                for feature in features:
                    for mass in mass_points:
                        futures.append(exc.submit(rerun_comb_years, maindir, cmtdir, feature, prd, mass, featureDependsOnMass))
                for res in concurrent.futures.as_completed(futures):
                    res.result()

