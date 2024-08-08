import os,sys,pdb
import numpy as np
import concurrent.futures
import subprocess
import itertools

import uproot
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

import warnings, logging
warnings.filterwarnings("ignore", message=".*Type 3 font.*")
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# IF YOU ONLY NEED TO CHANGE THE PLOTTING STYLE, RUN WITH THE OPTION: --plot_only

'''
# Use --run_ch or --no_run_ch for example

ulimit -s unlimited

python3 RunNonResLimits.py --ver ul_2016_ZZ_v12,ul_2016_HIPM_ZZ_v12,ul_2017_ZZ_v12,ul_2018_ZZ_v12 \
    --cat cat_ZZ_elliptical_cut_90_resolved_1b,cat_ZZ_elliptical_cut_90_resolved_2b,cat_ZZ_elliptical_cut_90_boosted_noPNet \
    --feat dnn_ZZbbtt_kl_1 --prd prod_240523 --grp datacard_zz --move_eos --user_eos evernazz --num 0

python3 RunNonResLimits.py --ver ul_2016_HIPM_ZbbHtt_v12,ul_2016_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
    --cat cat_ZbbHtt_elliptical_cut_90_resolved_1b,cat_ZbbHtt_elliptical_cut_90_resolved_2b,cat_ZbbHtt_elliptical_cut_90_boosted_noPNet \
    --feat dnn_ZbbHtt_kl_1 --prd prod_... --grp datacard_zbbhtt \
    --move_eos --user_eos cuisset --user_cmt cuisset

python3 RunNonResLimits.py --ver ul_2016_HIPM_ZttHbb_v12,ul_2016_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
    --cat cat_ZttHbb_elliptical_cut_90_resolved_1b,cat_ZttHbb_elliptical_cut_90_resolved_2b,cat_ZttHbb_elliptical_cut_90_boosted_noPNet \
    --feat dnn_ZHbbtt_kl_1 --prd prod_... --grp datacard_ztthbb \
    --move_eos --user_eos cuisset --user_cmt cuisset
'''

def run_cmd(cmd, run=True, check=True):
    # print('\n', cmd)
    if run:
        try:
            subprocess.run(cmd, shell=True, check=check, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command {cmd} failed with exit code {e.returncode}. Working directory : {os.getcwd()}") from None

def GetCatShort(category):
    if 'resolved_1b' in category: cat = 'res1b'
    if 'resolved_2b' in category: cat = 'res2b'
    if 'boosted' in category: cat = 'boosted'
    return cat

def GetYear(version):
    return version.split("ul_")[1].split("_Z")[0]

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    import matplotlib,argparse
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser("RunNonResLimits")
    
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
    makeFlag("--only_cards",              dest="only_cards",            default=False,            help='Skip all combine commands and plot only')
    makeFlag("--move_eos",                dest="move_eos",              default=False,            help='Move results to eos')
    makeFlag("--singleThread",            dest="singleThread",          default=False,            help="Don't run in parallel, disable for debugging")
    makeFlag("--unblind",                 dest="unblind",               default=False,            help="Pick the unblinded datacards and run unblinded limits")
    options = parser.parse_args()

    if ',' in options.ver:  versions = options.ver.split(',')
    else:                   versions = [options.ver]
    
    if ',' in options.cat:  categories = options.cat.split(',')
    else:                   categories = [options.cat]

    if ',' in options.feat: features = options.feat.split(',')
    else:                   features = [options.feat]

    if ',' in options.channels: channels = options.channels.split(',')
    else:                       channels = [options.channels]

    prd = options.prd
    grp = options.grp
    run = int(options.run) == 1
    run_one = options.run_one
    run_cp = options.run_cp
    run_ch = options.run_ch
    run_cat = options.run_cat
    run_zh_comb_cat = options.run_zh_comb_cat 
    run_year = options.run_year
    comb_2016 = options.comb_2016
    unblind = options.unblind
    if not unblind: 
        run_blind = '-t -1 --expectSignal 1'
    else:
        run_blind = ''

    cmtdir = '/data_CMS/cms/' + options.user_cmt + '/cmt/CreateDatacards/'
    maindir = os.getcwd() + f'/NonRes{options.num}/'


    if "ZZ" in options.ver:
        o_name = 'ZZbbtt'; fancy_name = '$ZZ_{bb\\tau\\tau}$'
        def get_r_range(feature, comb_type, channel):
            return "--rMin 0 --rMax 2"
        def get_r_range_setPR(feature, comb_type, channel):
            return "--setParameterRanges r=0,2"
    else: 
        if "ZbbHtt" in options.ver:
            o_name = 'ZbbHtt'; fancy_name = '$Z_{bb}H_{\\tau\\tau}$'
        elif "ZttHbb" in options.ver:
            o_name = 'ZttHbb'; fancy_name = '$Z_{\\tau\\tau}H_{bb}$'
        d_DNN = {
            "single_re1b" : (-40., 45),
            "single" : (-20., 25),
            "channel_res1b" : (-30., 35),
            "channel" : (-10., 15.),
            "category" : (-1., 4),
            "year" : (-0.5, 3.)
        }
        d_KinFit = {
            "single" : (-100., 100.),
            None : (-50., 50.)
        }
        def get_r_range(feature, comb_type, category, setPR=False):
            if "KinFit" in feature:
                d = d_KinFit
            else:
                d = d_DNN
            try:
                t = d[comb_type + "_" + GetCatShort(category)]
            except (KeyError, TypeError):
                try:
                    t = d[comb_type]
                except KeyError:
                    t = d[None]
            if setPR:
                return f"--setParameterRanges r={t[0]},{t[1]}"
            else:
                return f"--rMin {t[0]} --rMax {t[1]}"


    dict_ch_name = {"etau": "$\\tau_{e}\\tau_{h}$", "mutau": "$\\tau_{\\mu}\\tau_{h}$", "tautau": "$\\tau_{h}\\tau_{h}$"}

    def SetStyle(fig, x, line1="", line2="", max=5):
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
        plt.axhline(y=3.84, color='gray', linestyle='--', linewidth=2)
        plt.text(x[0] + 0.05, 1 + 0.1, '68% C.L.', fontsize=18)
        plt.text(x[0] + 0.05, 3.84 + 0.1, '95% C.L.', fontsize=18)
        plt.text(0.03, 0.97, line1, ha="left", va="top", transform=plt.gca().transAxes, color="black", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        plt.text(0.03, 0.92, line2, ha="left", va="top", transform=plt.gca().transAxes, color="black", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        mplhep.cms.label(data=False)
        plt.xlabel('$\\mu$')
        plt.ylabel('-2 $\\Delta LL$')
        plt.title("")
        plt.xlim(x[0], x[-1])
        plt.ylim(0,max)
        plt.grid()

    def WriteResults (fig, x, y, x_stat, y_stat, sig_file, sig=True, round = 2):
        central = x[np.argmin(y)]
        interval_1sigma = x[np.where(y < 1)]
        min_1sigma = np.abs(min(interval_1sigma)-central)
        max_1sigma = np.abs(max(interval_1sigma)-central)
        interval_1sigma_stat = x_stat[np.where(y_stat < 1)]
        min_1sigma_stat = np.abs(min(interval_1sigma_stat)-central)
        max_1sigma_stat = np.abs(max(interval_1sigma_stat)-central)
        r = central
        up = max_1sigma
        down = min_1sigma
        up_stat = max_1sigma_stat
        down_stat = min_1sigma_stat
        up_syst = np.sqrt(max_1sigma**2 - max_1sigma_stat**2)
        down_syst = np.sqrt(min_1sigma**2 - min_1sigma_stat**2)
        sig = GetLimit(sig_file)

        if not unblind: mu = '1.00'
        else:           mu = f'{central:.{round}f}'
        text = fr"$\mu = {mu}^{{+{up_syst:.{round}f}}}_{{-{down_syst:.{round}f}}}(syst)^{{+{up_stat:.{round}f}}}_{{-{down_stat:.{round}f}}}(stat)$"
        if sig: text += f"\nSignificance = {sig:.{round}f}$\sigma$"
        plt.text(0.03, 0.91, text, ha='left', va='top', transform=plt.gca().transAxes, fontsize='small',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        sig = GetLimit(sig_file)
    
    def GetDeltaLL(LS_file):
        file = uproot.open(LS_file)
        limit = file["limit"]
        r = limit.array("r")
        deltaNLL = 2 * limit.array("deltaNLL")
        sorted_indices = np.argsort(r)
        r_sorted = r[sorted_indices]
        deltaNLL_sorted = deltaNLL[sorted_indices]
        return r_sorted[1:], deltaNLL_sorted[1:]

    def GetLimit(LS_file):
        file = uproot.open(LS_file)
        limit_tree = file["limit"]
        limit_value = limit_tree["limit"].array()[0]
        return limit_value

    if run_cp:

        ############################################################################
        print("\n ### INFO: Copy all datacards \n")
        ############################################################################

        for feature in features:
            for version in versions:
                for category in categories:
                    for channel in channels:
                        odir = maindir + f'/{version}/{prd}/{feature}/{category}/{channel}'
                        run_cmd('mkdir -p ' + odir)
                        if not unblind:
                            ch_file = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_{channel}_os_iso.txt'
                        else:
                            ch_file = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_{channel}_os_iso__unblind.txt'
                        run_cmd(f'cp {ch_file} {odir}/{version}_{category}_{feature}_{grp}_{channel}_os_iso.txt')

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    if comb_2016:

        if any("2016_Z" in s for s in versions) and any("2016_HIPM_Z" in s for s in versions):
        
            prefix = "ul_"
            suffix = "_Z" + versions[0].split("_Z")[1]
            v_2016 = prefix + "2016" + suffix
            v_2016_HIPM = prefix + "2016_HIPM" + suffix
            v_combined = prefix + "2016_ALL" + suffix

            def run_comb_2016(feature, category, channel):
            
                combdir = maindir + f'/{v_combined}/{prd}/{feature}/{category}/{channel}'
                print(" ### INFO: Saving combination in ", combdir)
                run_cmd('mkdir -p ' + combdir)

                cmd = f'combineCards.py'
                for version in [v_2016, v_2016_HIPM]:
                    year_file = maindir + f'/{version}/{prd}/{feature}/{category}/{channel}/{version}_{category}_{feature}_{grp}_{channel}_os_iso.txt'
                    year = version.split("ul_")[1].split("_Z")[0]
                    cmd += f' Y{year}={year_file}'
                    cmd += f' > {v_combined}_{category}_{feature}_{grp}_{channel}_os_iso.txt'
                if run: os.chdir(combdir)
                run_cmd(cmd, run)

            if not options.plot_only and (options.run_one or options.only_cards):
    
                ############################################################################
                print("\n ### INFO: Run Combination of 2016 and 2016_HIPM \n")
                ############################################################################

                if options.singleThread:
                    for feature in features:
                        for category in categories:
                            for channel in channels:
                                run_comb_2016(feature, category, channel)
                else:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                        futures = []
                        for feature in features:
                            for category in categories:
                                for channel in channels:
                                    run_comb_2016(feature, category, channel)
                        for res in concurrent.futures.as_completed(futures):
                            res.result()

            versions.remove(v_2016) 
            versions.remove(v_2016_HIPM)
            versions.insert(0, v_combined)

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_single_limit(feature, version, category, ch):

        if "boosted" in category:        cat_name = r"Boosted"
        elif "resolved_1b" in category:  cat_name = r"Res 1b"
        elif "resolved_2b" in category:  cat_name = r"Res 2b"
        else:                            cat_name = "category"

        r_range = get_r_range(feature, "single", category)

        ch_dir = maindir + f'/{version}/{prd}/{feature}/{category}/{ch}'
        run_cmd('mkdir -p ' + ch_dir)

        datafile = ch_dir + f'/{version}_{category}_{feature}_{grp}_{ch}_os_iso.txt'

        print(" ### INFO: Create workspace")
        cmd = f'cd {ch_dir} && text2workspace.py {datafile} -o {ch_dir}/model.root &>{ch_dir}/text2workspace.log'
        run_cmd(cmd, run)

        print(" ### INFO: Run Delta Log Likelihood Scan")
        cmd = f'cd {ch_dir} && combine -M MultiDimFit {ch_dir}/model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>{ch_dir}/multiDimFit.log'
        run_cmd(cmd, run)

        LS_file = f'{ch_dir}/higgsCombineTest.MultiDimFit.mH120.root'
        x, y = GetDeltaLL(LS_file)

        fig = plt.figure(figsize=(10, 10))
        plt.plot(x, y, label='Data', color='red', linewidth=3)
        SetStyle(fig, x, cat_name, dict_ch_name[ch])
        ver_short = version.split("ul_")[1].split("_Z")[0] ; cat_short = category.split("_cut_90_")[1]
        plt.savefig(f"{ch_dir}/DeltaNLL_{ver_short}_{cat_short}_{ch}.png")
        plt.savefig(f"{ch_dir}/DeltaNLL_{ver_short}_{cat_short}_{ch}.pdf")
        plt.close()

        if options.singleThread:

            # Creates problems when parallelizing, but it's only to print out the significance so we can drop it
            print(" ### INFO: Run significance extraction")
            cmd = f'cd {ch_dir} && combine -M Significance {datafile} {run_blind} --pvalue &> {ch_dir}/PValue.log'
            run_cmd(cmd, run)
            run_cmd(f'mv {ch_dir}/higgsCombineTest.Significance.mH120.root {ch_dir}/higgsCombineTest.Significance.mH120.pvalue.root')
            LS_file = f'{ch_dir}/higgsCombineTest.Significance.mH120.pvalue.root'
            a = GetLimit(LS_file)

            cmd = f'cd {ch_dir} && combine -M Significance {datafile} {run_blind} &> {ch_dir}/Significance_{ver_short}_{cat_short}_{ch}.log'
            run_cmd(cmd, run)
            run_cmd(f'mv {ch_dir}/higgsCombineTest.Significance.mH120.root {ch_dir}/higgsCombineTest.Significance.mH120.significance.root')
            LS_file = f'{ch_dir}/higgsCombineTest.Significance.mH120.significance.root'
            b = GetLimit(LS_file)

            print(" ### INFO: Results for", feature, version, category, ch)
            print(" ### p-value     = ", a)
            print(" ### significane = ", b, "\n")

        if run: os.chdir(ch_dir)

    if run_one and not options.only_cards:

        ############################################################################
        print("\n ### INFO: Run Single Limits\n")
        ############################################################################

        if options.singleThread:
            for feature in features:
                for version in versions:
                    for category in categories:
                        for channel in channels:
                            run_single_limit(feature, version, category, channel)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                futures = []
                for feature in features:
                    for version in versions:
                        for category in categories:
                            for channel in channels:
                                futures.append(exc.submit(run_single_limit, feature, version, category, channel))
                for res in concurrent.futures.as_completed(futures):
                    res.result()
        
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_comb_channels(feature, version, category):

        combdir = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch'
        print(" ### INFO: Saving combination in ", combdir)
        run_cmd('mkdir -p ' + combdir, run)

        cmd = f'combineCards.py'
        for ch in channels:
            ch_file = maindir + f'/{version}/{prd}/{feature}/{category}/{ch}/{version}_{category}_{feature}_{grp}_{ch}_os_iso.txt'
            cmd += f' {ch}={ch_file}'
        cmd += f' > {version}_{feature}_{category}_os_iso.txt'
        if run: os.chdir(combdir)
        run_cmd(cmd, run)

        if options.only_cards: return True

        r_range = get_r_range(feature, "channel", category)
        r_range_setPR = get_r_range(feature, "channel", category, setPR=True)
    
        cmd = f'text2workspace.py {version}_{feature}_{category}_os_iso.txt -o model.root &>{combdir}/text2workspace.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 {run_blind} &>multiDimFit_singles.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>multiDimFit_grid.log'
        run_cmd(cmd, run)
        cmd = f'combine -M Significance {version}_{feature}_{category}_os_iso.txt {run_blind} &> Significance_{version}_{feature}_{category}.log'
        run_cmd(cmd, run)

        cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            f' --preFitValue 1 {run_blind} &>MultiDimFit.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            f'--saveWorkspace --preFitValue 1 {run_blind} -n .scan.with_syst.statonly_correct --algo grid '\
            '--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>MultiDimFit_statOnly.log'
        run_cmd(cmd, run)

    if run_ch:

        if not options.plot_only:

            ############################################################################
            print("\n ### INFO: Run Combination of channels \n")
            ############################################################################

            if options.singleThread:
                for feature in features:
                    for version in versions:
                        for category in categories:
                            run_comb_channels(feature, version, category)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                    futures = []
                    for feature in features:
                        for version in versions:
                            for category in categories:
                                futures.append(exc.submit(run_comb_channels, feature, version, category))
                    for res in concurrent.futures.as_completed(futures):
                        res.result()

        if not options.only_cards:

            ############################################################################
            print("\n ### INFO: Plot Combination of channels \n")
            ############################################################################

            for feature in features:
                for version in versions:
                    for category in categories:

                        if "boosted" in category:        cat_name = r"Boosted"
                        elif "resolved_1b" in category:  cat_name = r"Res 1b"
                        elif "resolved_2b" in category:  cat_name = r"Res 2b"
                        else:                            cat_name = category

                        fig = plt.figure(figsize=(10, 10))
                        cmap = plt.get_cmap('tab10')
                        for i, ch in enumerate(channels):
                            LS_file = maindir + f'/{version}/{prd}/{feature}/{category}/{ch}/higgsCombineTest.MultiDimFit.mH120.root'
                            x, y = GetDeltaLL(LS_file)
                            plt.plot(x, y, label=dict_ch_name[ch], linewidth=3, color=cmap(i))
                        LS_file = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/higgsCombineTest.MultiDimFit.mH120.root'
                        x, y = GetDeltaLL(LS_file)
                        plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
                        LS_file = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
                        x_stat, y_stat = GetDeltaLL(LS_file)
                        plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
                        plt.legend(loc='upper right', fontsize=18, frameon=True)
                        SetStyle(fig, x, cat_name, "", 8)
                        WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/higgsCombineTest.Significance.mH120.root')
                        ver_short = version.split("ul_")[1].split("_Z")[0] ; cat_short = category.split("_cut_90_")[1]
                        plt.savefig(maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/DeltaNLL_{ver_short}_{cat_short}.png')
                        plt.savefig(maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/DeltaNLL_{ver_short}_{cat_short}.pdf')
                        plt.close()

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_comb_categories(feature, version):

        combdir = maindir + f'/{version}/{prd}/{feature}/Combination_Cat'
        print(" ### INFO: Saving combination in ", combdir)
        run_cmd('mkdir -p ' + combdir)

        cmd = f'combineCards.py'
        for category in categories:
            cat_file = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/{version}_{feature}_{category}_os_iso.txt'
            cmd += f' {GetCatShort(category)}={cat_file}'
        cmd += f' > {combdir}/{version}_{feature}_os_iso.txt'
        if run: os.chdir(combdir)
        run_cmd(cmd, run)

        if options.only_cards: return True

        r_range = get_r_range(feature, "category", None)
        r_range_setPR = get_r_range(feature, "category", None, setPR=True)
    
        cmd = f'cd {combdir} && text2workspace.py {combdir}/{version}_{feature}_os_iso.txt --channel-masks -o {combdir}/model.root &>{combdir}/text2workspace.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit {combdir}/model.root --algo=singles {r_range} --preFitValue 1 {run_blind} &>{combdir}/multiDimFit_singles.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>multiDimFit_grid.log'
        run_cmd(cmd, run)
        cmd = f'combine -M Significance {combdir}/{version}_{feature}_os_iso.txt {run_blind} &> {combdir}/Significance_{version}_{feature}.log'
        run_cmd(cmd, run)

        cmd = f'combine -M MultiDimFit {combdir}/model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            f' --preFitValue 1 {run_blind} &>{combdir}/MultiDimFit.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit {combdir}/higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            f'--saveWorkspace --preFitValue 1 {run_blind} -n .scan.with_syst.statonly_correct --algo grid '\
            f'--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>{combdir}/MultiDimFit_statOnly.log'
        run_cmd(cmd, run)

    if run_cat:

        if not options.plot_only:

            ############################################################################
            print("\n ### INFO: Run Combination of categories \n")
            ############################################################################

            if options.singleThread:
                for feature in features:
                    for version in versions:
                        run_comb_categories(feature, version)

            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                    futures = []
                    for feature in features:
                        for version in versions:
                            futures.append(exc.submit(run_comb_categories, feature, version))
                    for res in concurrent.futures.as_completed(futures):
                        res.result()

        if not options.only_cards:

            ############################################################################
            print("\n ### INFO: Plot Combination of categories \n")
            ############################################################################

            for feature in features:
                for version in versions:
                    fig = plt.figure(figsize=(10, 10))
                    cmap = plt.get_cmap('tab10')
                    for i, category in enumerate(categories):
                        LS_file = maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/higgsCombineTest.MultiDimFit.mH120.root'
                        x, y = GetDeltaLL(LS_file)
                        if "boosted" in category:        cat_name = r"Boosted"
                        elif "resolved_1b" in category:  cat_name = r"Res 1b"
                        elif "resolved_2b" in category:  cat_name = r"Res 2b"
                        plt.plot(x, y, label=cat_name, linewidth=3, color=cmap(i))
                    LS_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombineTest.MultiDimFit.mH120.root'
                    x, y = GetDeltaLL(LS_file)
                    plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
                    LS_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
                    x_stat, y_stat = GetDeltaLL(LS_file)
                    plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
                    plt.legend(loc='upper right', fontsize=18, frameon=True)
                    year = version.split("ul_")[1].split("_Z")[0]
                    SetStyle(fig, x, year, "", 8)
                    WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombineTest.Significance.mH120.root')
                    ver_short = version.split("ul_")[1].split("_Z")[0]
                    plt.savefig(maindir + f'/{version}/{prd}/{feature}/Combination_Cat/DeltaNLL_{ver_short}.png')
                    plt.savefig(maindir + f'/{version}/{prd}/{feature}/Combination_Cat/DeltaNLL_{ver_short}.pdf')
                    plt.close()

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_comb_ZH(feature, version_ZbbHtt, version_ZttHbb):
        """ Combination of ZbbHtt & ZttHbb for a single year """
        version_comb = version_ZbbHtt.replace("ZbbHtt", "ZHComb")
        combdir = maindir + f'/{version_comb}/{prd}/{feature}/'
        print(" ### INFO: Saving ZH combination in ", combdir)
        run_cmd('mkdir -p ' + combdir)

        cmd = f'combineCards.py'
        for version, short_name in [(version_ZbbHtt, "ZbbHtt"), (version_ZttHbb, "ZttHbb")]:
            cat_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/{version}_{feature}_os_iso.txt'
            cmd += f' {short_name}={cat_file}'
        cmd += f' > {version_comb}_{feature}_os_iso.txt'
        if run: os.chdir(combdir)
        run_cmd(cmd, run)

        r_range = get_r_range(feature, "category", None)
        r_range_setPR = get_r_range(feature, "category", None, setPR=True)
    
        cmd = f'text2workspace.py {version_comb}_{feature}_os_iso.txt -o model.root &>text2workspace.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 {run_blind} &>multiDimFit_singles.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>multiDimFit_grid.log'
        run_cmd(cmd, run)
        cmd = f'combine -M Significance {version_comb}_{feature}_os_iso.txt {run_blind} &> Significance_{version_comb}_{feature}.log'
        run_cmd(cmd, run)

        cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            f' --preFitValue 1 {run_blind} &>MultiDimFit.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            f'--saveWorkspace --preFitValue 1 {run_blind} -n .scan.with_syst.statonly_correct --algo grid '\
            f'--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>MultiDimFit_statOnly.log'
        run_cmd(cmd, run)

    def split_versions_ZH():
        """ Split the input versions into ZbbHtt versions and ZttHbb versions """
        versions_ZbbHtt = []
        versions_ZttHbb = []
        for version in versions:
            if "ZbbHtt" in version:
                ver_ZttHbb = version.replace("ZbbHtt", "ZttHbb")
                if ver_ZttHbb not in versions:
                    raise RuntimeError(f"Whilst running ZH combination : {version} was found but not corresponding {ver_ZttHbb}")
                versions_ZbbHtt.append(version)
                versions_ZttHbb.append(ver_ZttHbb)
        return versions_ZbbHtt, versions_ZttHbb

    if run_zh_comb_cat: # Combine ZH for every year
        versions_ZbbHtt, versions_ZttHbb = split_versions_ZH()

        if not options.plot_only:

            ############################################################################
            print("\n ### INFO: Run Combination of ZH \n")
            ############################################################################

            if options.singleThread:
                for feature in features:
                    for ver_ZbbHtt, ver_ZttHbb in zip(versions_ZbbHtt, versions_ZttHbb):
                        run_comb_ZH(feature, ver_ZbbHtt, ver_ZttHbb)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                    futures = []
                    for feature in features:
                        for ver_ZbbHtt, ver_ZttHbb in zip(versions_ZbbHtt, versions_ZttHbb):
                            futures.append(exc.submit(run_comb_ZH, feature, ver_ZbbHtt, ver_ZttHbb))
                    for res in concurrent.futures.as_completed(futures):
                        res.result()
        
        if not options.only_cards:

            ############################################################################
            print("\n ### INFO: Plot Combination of ZH \n")
            ############################################################################

            for feature in features:
                for version_ZbbHtt, version_ZttHbb in zip(versions_ZbbHtt, versions_ZttHbb):
                    version_comb = version_ZbbHtt.replace("ZbbHtt", "ZHComb")
                    fig = plt.figure(figsize=(10, 10))
                    cmap = plt.get_cmap('tab10')
                    for i, (version, short_name) in enumerate([(version_ZbbHtt, "ZbbHtt"), (version_ZttHbb, "ZttHbb")]):
                        cat_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/{version}_{feature}_os_iso.txt'
                        LS_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombineTest.MultiDimFit.mH120.root'
                        x, y = GetDeltaLL(LS_file)
                        plt.plot(x, y, label=short_name, linewidth=3, color=cmap(i))
                    LS_file = maindir + f'/{version_comb}/{prd}/{feature}/higgsCombineTest.MultiDimFit.mH120.root'
                    x, y = GetDeltaLL(LS_file)
                    plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
                    LS_file = maindir + f'/{version_comb}/{prd}/{feature}/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
                    x_stat, y_stat = GetDeltaLL(LS_file)
                    plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
                    plt.legend(loc='upper right', fontsize=18, frameon=True)
                    SetStyle(fig, x, "ZH Combination", "", 8)
                    WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/{version_comb}/{prd}/{feature}/higgsCombineTest.Significance.mH120.root')
                    ver_short = version.split("ul_")[1].split("_Z")[0]
                    plt.savefig(maindir + f'/{version_comb}/{prd}/{feature}/DeltaNLL_{ver_short}.png')
                    plt.savefig(maindir + f'/{version_comb}/{prd}/{feature}/DeltaNLL_{ver_short}.pdf')
                    plt.close()

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_comb_years(feature):

        combdir = maindir + f'/FullRun2_{o_name}/{prd}/{feature}'
        print(" ### INFO: Saving combination in ", combdir)
        run_cmd('mkdir -p ' + combdir)

        cmd = f'combineCards.py'
        for version in versions:
            ver_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/{version}_{feature}_os_iso.txt'
            if os.path.exists(ver_file):
                cmd += f' Y{GetYear(version)}={ver_file}'
        cmd += f' > FullRun2_{o_name}_{feature}_os_iso.txt'
        if run: os.chdir(combdir)
        print('\n',cmd,'\n')
        run_cmd(cmd, run)

        if options.only_cards: return True

        r_range = get_r_range(feature, "year", None)
        r_range_setPR = get_r_range(feature, "year", None, setPR=True)
    
        cmd = f'text2workspace.py FullRun2_{o_name}_{feature}_os_iso.txt -o model.root &>text2workspace.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 {run_blind} &>multiDimFit_singles.log'
        run_cmd(cmd, run)
        if True or options.singleThread:
            prefix_cmd = "combine "
        else: # weird things happen
            prefix_cmd = "combineTool.py --split-points 5 --job-mode=interactive --parallel=20 "
        cmd = prefix_cmd + f'-M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>multiDimFit_grid.log'
        run_cmd(cmd, run)
        cmd = f'combine -M Significance FullRun2_{o_name}_{feature}_os_iso.txt {run_blind} &> Significance_{feature}.log'
        run_cmd(cmd, run)

        cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            f' --preFitValue 1 {run_blind} &>MultiDimFit.log'
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            f'--saveWorkspace --preFitValue 1 {run_blind} -n .scan.with_syst.statonly_correct --algo grid '\
            f'--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>MultiDimFit_statOnly.log'
        run_cmd(cmd, run)

        if options.run_impacts or options.run_impacts_noMCStat:
            print(" ### INFO: Produce Full Run 2 Impact Plots")

            if not options.run_impacts_noMCStat:
                cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' 
                run_cmd(cmd, run)
                cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doFits --robustFit 1 --parallel 50'
                run_cmd(cmd, run)
                cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts.json --parallel 50'
                run_cmd(cmd, run)
                cmd = f'plotImpacts.py -i impacts.json -o Impacts_{o_name}_{feature}'
                run_cmd(cmd, run)
                if run: run_cmd('mkdir -p impacts')
                if run: run_cmd('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts')

            cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' +  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doFits --robustFit 1 --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts_noMCstats.json --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = f'plotImpacts.py -i impacts_noMCstats.json -o Impacts_{o_name}_{feature}_NoMCstats'
            run_cmd(cmd, run)
            if run: run_cmd('mkdir -p impacts_noMCstats')
            if run: run_cmd('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts_noMCstats')

    if run_year:

        if not options.plot_only:

            ############################################################################
            print("\n ### INFO: Run Combination of years \n")
            ############################################################################

            if options.singleThread:
                for feature in features:
                    run_comb_years(feature)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                    futures = []
                    for feature in features:
                        futures.append(exc.submit(run_comb_years, feature))
                    for res in concurrent.futures.as_completed(futures):
                        res.result()

        if not options.only_cards:

            ############################################################################
            print("\n ### INFO: Plot Combination of years \n")
            ############################################################################

            for feature in features:
                fig = plt.figure(figsize=(10, 10))
                cmap = plt.get_cmap('tab10')
                for i, version in enumerate(versions):
                    LS_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombineTest.MultiDimFit.mH120.root'
                    x, y = GetDeltaLL(LS_file)
                    plt.plot(x, y, label=version.split("ul_")[1].split("_Z")[0], linewidth=3, color=cmap(i))
                LS_file = maindir + f'/FullRun2_{o_name}/{prd}/{feature}/higgsCombineTest.MultiDimFit.mH120.root'
                x, y = GetDeltaLL(LS_file)
                plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
                LS_file = maindir + f'/FullRun2_{o_name}/{prd}/{feature}/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
                x_stat, y_stat = GetDeltaLL(LS_file)
                plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
                plt.legend(loc='upper right', fontsize=18, frameon=True)
                SetStyle(fig, x, fancy_name, "", 8)
                WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/FullRun2_{o_name}/{prd}/{feature}/higgsCombineTest.Significance.mH120.root')
                plt.savefig(maindir + f'/FullRun2_{o_name}/{prd}/{feature}/DeltaNLL_FullRun2_{o_name}.png')
                plt.savefig(maindir + f'/FullRun2_{o_name}/{prd}/{feature}/DeltaNLL_FullRun2_{o_name}.pdf')
                plt.close()


    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def run_comb_years_ZH(feature):

        combdir = maindir + f'/FullRun2_ZHComb/{prd}/{feature}'
        print(" ### INFO: Saving ZH FullRun2 combination in ", combdir)
        run_cmd('mkdir -p ' + combdir)

        cmd = f'combineCards.py'
        versions_ZbbHtt, versions_ZttHbb = split_versions_ZH()
        for version in versions_ZbbHtt:
            year = version.split("ul_")[1].split("_Z")[0]
            ver_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/{version}_{feature}_os_iso.txt'
            if os.path.exists(ver_file):
                cmd += f' Y{year}_ZbbHtt={ver_file}'
        for version in versions_ZttHbb:
            year = version.split("ul_")[1].split("_Z")[0]
            ver_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/{version}_{feature}_os_iso.txt'
            if os.path.exists(ver_file):
                cmd += f' Y{year}_ZttHbb={ver_file}'
        cmd += f' > FullRun2_ZHComb_{o_name}_{feature}_os_iso.txt'
        if run: os.chdir(combdir)
        print(cmd)
        run_cmd(cmd, run)

        r_range = get_r_range(feature, "year", None)
        r_range_setPR = get_r_range(feature, "year", None, setPR=True)
    
        cmd = f'text2workspace.py FullRun2_ZHComb_{o_name}_{feature}_os_iso.txt -o model.root &>text2workspace.log'
        print(cmd)
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 {run_blind} &>multiDimFit_singles.log'
        print(cmd)
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 {run_blind} &>multiDimFit_grid.log'
        print(cmd)
        run_cmd(cmd, run)
        cmd = f'combine -M Significance FullRun2_ZHComb_{o_name}_{feature}_os_iso.txt {run_blind} &> Significance_{feature}.log'
        print(cmd)
        run_cmd(cmd, run)

        cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            f' --preFitValue 1 {run_blind} &>MultiDimFit.log'
        print(cmd)
        run_cmd(cmd, run)
        cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            f'--saveWorkspace --preFitValue 1 {run_blind} -n .scan.with_syst.statonly_correct --algo grid '\
            f'--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>MultiDimFit_statOnly.log'
        print(cmd)
        run_cmd(cmd, run)

        if options.run_impacts:
            print(" ### INFO: Produce Full Run 2 ZHComb Impact Plots")

            if not options.run_impacts_noMCStat:
                cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' 
                run_cmd(cmd, run)
                cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doFits --robustFit 1 --parallel 50'
                run_cmd(cmd, run)
                cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts.json --parallel 50'
                run_cmd(cmd, run)
                cmd = f'plotImpacts.py -i impacts.json -o Impacts_{o_name}_{feature}'
                run_cmd(cmd, run)
                run_cmd('mkdir -p impacts')
                run_cmd('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts')

            cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' +  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = f'combineTool.py -M Impacts -d model.root -m 125 --preFitValue 1 {run_blind} {r_range_setPR} --doFits --robustFit 1 --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts_noMCstats.json --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
            run_cmd(cmd, run)
            cmd = f'plotImpacts.py -i impacts_noMCstats.json -o Impacts_{o_name}_{feature}_NoMCstats'
            run_cmd(cmd, run)
            run_cmd('mkdir -p impacts_noMCstats')
            run_cmd('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts_noMCstats')

    if options.run_zh_comb_year:

        if not options.plot_only:

            ############################################################################
            print("\n ### INFO: Run Combination of years & ZH \n")
            ############################################################################

            if options.singleThread:
                for feature in features:
                    run_comb_years_ZH(feature)
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=15) as exc:
                    futures = []
                    for feature in features:
                        futures.append(exc.submit(run_comb_years_ZH, feature))
                    for res in concurrent.futures.as_completed(futures):
                        res.result()

        ############################################################################
        print("\n ### INFO: Plot Combination of years & ZH \n")
        ############################################################################

        for feature in features:
            # Plot with all years and ZbbHtt/ZttHbb
            fig = plt.figure(figsize=(10, 10))
            cmap = plt.get_cmap('tab10')
            for i, version in enumerate(versions):
                LS_file = maindir + f'/{version}/{prd}/{feature}/Combination_Cat/higgsCombineTest.MultiDimFit.mH120.root'
                x, y = GetDeltaLL(LS_file)
                plt.plot(x, y, label=version.split("ul_")[1].replace("_", " "), linewidth=3, color=cmap(i))
            LS_file = maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombineTest.MultiDimFit.mH120.root'
            x, y = GetDeltaLL(LS_file)
            plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
            LS_file = maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
            x_stat, y_stat = GetDeltaLL(LS_file)
            plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
            plt.legend(loc='upper right', fontsize=18, frameon=True)
            SetStyle(fig, x, fancy_name, "", 8)
            WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombineTest.Significance.mH120.root')
            plt.savefig(maindir + f'/FullRun2_ZHComb/{prd}/{feature}/DeltaNLL_FullRun2_ZHComb_allYears.png')
            plt.savefig(maindir + f'/FullRun2_ZHComb/{prd}/{feature}/DeltaNLL_FullRun2_ZHComb_allYears.pdf')
            plt.close()

            # Plot only ZbbHtt / ZttHbb
            fig = plt.figure(figsize=(10, 10))
            cmap = plt.get_cmap('tab10')
            for i, version in enumerate(["ZbbHtt", "ZttHbb"]):
                LS_file = maindir + f'/FullRun2_{version}/{prd}/{feature}/higgsCombineTest.MultiDimFit.mH120.root'
                x, y = GetDeltaLL(LS_file)
                plt.plot(x, y, label=version, linewidth=3, color=cmap(i))
            LS_file = maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombineTest.MultiDimFit.mH120.root'
            x, y = GetDeltaLL(LS_file)
            plt.plot(x, y, label='Combination', linewidth=3, color=cmap(i+1))
            LS_file = maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
            x_stat, y_stat = GetDeltaLL(LS_file)
            plt.plot(x_stat, y_stat, label='Stat-only', linewidth=3, linestyle='--', color=cmap(i+1))
            plt.legend(loc='upper right', fontsize=18, frameon=True)
            SetStyle(fig, x, fancy_name, "", 8)
            WriteResults(fig, x, y, x_stat, y_stat, maindir + f'/FullRun2_ZHComb/{prd}/{feature}/higgsCombineTest.Significance.mH120.root')
            plt.savefig(maindir + f'/FullRun2_ZHComb/{prd}/{feature}/DeltaNLL_FullRun2_ZHComb_summary.png')
            plt.savefig(maindir + f'/FullRun2_ZHComb/{prd}/{feature}/DeltaNLL_FullRun2_ZHComb_summary.pdf')
            plt.close()

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    if options.move_eos:

        eos_dir = f'/eos/user/e/evernazz/www/ZZbbtautau/B2GPlots/2024_06_14/{o_name}/Limits/NonRes'
        user = options.user_eos
        print(f" ### INFO: Copy results to {user}@lxplus.cern.ch")
        print(f"           Inside directory {eos_dir}\n")

        # [FIXME] Work-around for mkdir on eos
        run_cmd(f'mkdir -p TMP_RESULTS_NONRES && cp index.php TMP_RESULTS_NONRES')
        for feature in features:
            run_cmd(f'cp ' + maindir + f'/FullRun2_{o_name}/{prd}/{feature}/DeltaNLL_*.p* TMP_RESULTS_NONRES')
            run_cmd(f'cp ' + maindir + f'/FullRun2_{o_name}/{prd}/{feature}/*_os_iso.txt TMP_RESULTS_NONRES')
            run_cmd(f'cp ' + maindir + f'/FullRun2_{o_name}/{prd}/{feature}/Impacts* TMP_RESULTS_NONRES')
            for version in versions:
                ver_short = version.split("ul_")[1].split("_Z")[0]
                run_cmd(f'mkdir -p TMP_RESULTS_NONRES/{ver_short} && cp index.php TMP_RESULTS_NONRES/{ver_short}')
                run_cmd(f'cp ' + maindir + f'/{version}/{prd}/{feature}/Combination_Cat/DeltaNLL*.p* TMP_RESULTS_NONRES/{ver_short}')
                for category in categories:
                    run_cmd(f'cp ' + maindir + f'/{version}/{prd}/{feature}/{category}/Combination_Ch/DeltaNLL*.p* TMP_RESULTS_NONRES/{ver_short}', check=False)
        run_cmd(f'rsync -rltv TMP_RESULTS_NONRES/* {user}@lxplus.cern.ch:{eos_dir}')
        run_cmd(f'rm -r TMP_RESULTS_NONRES')


        if options.run_zh_comb_year:
            eos_dir = f'/eos/user/e/evernazz/www/ZZbbtautau/B2GPlots/2024_06_14/ZHComb/Limits/NonRes'
            run_cmd(f'mkdir -p TMP_RESULTS_NONRES && cp index.php TMP_RESULTS_NONRES')
            run_cmd(f'cp ' + maindir + f'/FullRun2_ZHComb/{prd}/{feature}/DeltaNLL_*.p* TMP_RESULTS_NONRES')
            run_cmd(f'cp ' + maindir + f'/FullRun2_ZHComb/{prd}/{feature}/*_os_iso.txt TMP_RESULTS_NONRES')
            run_cmd(f'cp ' + maindir + f'/FullRun2_ZHComb/{prd}/{feature}/Impacts* TMP_RESULTS_NONRES')

            versions_ZbbHtt, versions_ZttHbb = split_versions_ZH()
            for version in versions_ZbbHtt:
                version_comb = version_ZbbHtt.replace("ZbbHtt", "ZHComb")
                ver_short = version.split("ul_")[1].split("_Z")[0]
                run_cmd(f'cp ' + maindir + f'/{version}/{prd}/{feature}/DeltaNLL*.p* TMP_RESULTS_NONRES/{ver_short}')
            run_cmd(f'rsync -rltv TMP_RESULTS_NONRES/* {user}@lxplus.cern.ch:{eos_dir}')
            run_cmd(f'rm -r TMP_RESULTS_NONRES')

