import os,sys,pdb
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep
plt.style.use(mplhep.style.CMS)
ROOT.gROOT.SetBatch(True)

def GetLegend(x, y, x_stat, y_stat, round=3):
    central = x[np.argmin(y)]
    interval_1sigma = x[np.where(y < 1)]
    min_1sigma = np.abs(min(interval_1sigma)-central)
    max_1sigma = np.abs(max(interval_1sigma)-central)
    interval_1sigma_stat = x_stat[np.where(y_stat < 1)]
    min_1sigma_stat = np.abs(min(interval_1sigma_stat)-central)
    max_1sigma_stat = np.abs(max(interval_1sigma_stat)-central)
    r = np.round(central, round)
    up = np.round(max_1sigma, round)
    down = np.round(min_1sigma, round)
    up_stat = np.round(max_1sigma_stat, round)
    down_stat = np.round(min_1sigma_stat, round)
    up_syst = np.round(np.sqrt(max_1sigma**2 - max_1sigma_stat**2), round)
    down_syst = np.round(np.sqrt(min_1sigma**2 - min_1sigma_stat**2), round)
    return r, up, down, up_stat, down_stat, up_syst, down_syst

'''
python3 PlotNonResLimits.py --ver ul_2018_ZZ_v12 \
    --cat cat_ZZ_elliptical_cut_90_resolved_1b,cat_ZZ_elliptical_cut_90_resolved_2b,cat_ZZ_elliptical_cut_90_boosted_noPNet \
    --feat dnn_ZZbbtt_kl_1 --prd prod_240523 --grp datacard_zz
'''

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",     dest="run",      default=True)
    parser.add_option("--ver",     dest="ver",      default='')
    parser.add_option("--cat",     dest="cat",      default='')
    parser.add_option("--prd",     dest="prd",      default='')
    parser.add_option("--feat",    dest="feat",     default='dnn_ZZbbtt_kl_1')
    parser.add_option("--grp",     dest="grp",      default='datacard_zz')
    parser.add_option("--channels", dest="channels", default="etau,mutau,tautau")
    (options, args) = parser.parse_args()

    if ',' in options.ver:
        versions = options.ver.split(',')
        if "ZZ" in options.ver:
            o_name = 'ZZ_FullRun2'
        elif "ZbbHtt" in options.ver:
            o_name = 'ZbbHtt_FullRun2'
        elif "ZttHbb" in options.ver:
            o_name = 'ZttHbb_FullRun2'
    else:
        versions = [options.ver]
        o_name = options.ver
    
    if ',' in options.cat:
        categories = options.cat.split(',')
    else:
        categories = [options.cat]

    if ',' in options.feat:
        features = options.feat.split(',')
    else:
        features = [options.feat]
    
    if ',' in options.channels:
        channels = options.channels.split(',')
    else:
        channels = [options.channels]

    prd = options.prd
    grp = options.grp
    run = int(options.run) == 1

    if os.environ["USER"] == 'evernazza':
        cmtdir = '/data_CMS/cms/' + os.environ["USER"][1:] + '/cmt/CreateDatacards/'
    else:
        cmtdir = '/data_CMS/cms/' + os.environ["USER"] + '/cmt/CreateDatacards/'
        
    maindir = os.getcwd()

    for feature in features:


        years_list = []

        for version in versions:
            if "2016_HIPM" in version:
                years_list.append("2016_HIPM")
            else:
                years_list.append(version[3:7])

            datacard_list = []
            root_list = []
            LS_file_comb_list = []

            for category in categories:

                combdir = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/Combination'
                print(" ### INFO: Saving combination in ", combdir)
                if run: os.system('mkdir -p ' + combdir)
                etau_file = ''; mutau_file = ''; tautau_file = ''
                if 'etau' in channels: 
                    etau_file = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_etau_os_iso.txt'
                    etau_root = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_etau_os_iso.root'
                if 'mutau' in channels:
                    mutau_file = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_mutau_os_iso.txt'
                    mutau_root = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_mutau_os_iso.root'
                if 'tautau' in channels:
                    tautau_file = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_tautau_os_iso.txt'
                    tautau_root = cmtdir + f'/{version}/{category}/{prd}/{feature}_{grp}_tautau_os_iso.root'

                cmd = 'combineCards.py'
                n_comb = 1
                if os.path.exists(etau_file):
                    os.system('cp {} {} {}'.format(etau_file, etau_root, combdir))
                    cmd += f' etau={feature}_{grp}_etau_os_iso.txt'
                    n_comb += 1
                if os.path.exists(mutau_file):
                    os.system('cp {} {} {}'.format(mutau_file, mutau_root, combdir))
                    cmd += f' mutau={feature}_{grp}_mutau_os_iso.txt'
                    n_comb += 1
                if os.path.exists(tautau_file):
                    os.system('cp {} {} {}'.format(tautau_file, tautau_root, combdir))
                    cmd += f' tautau={feature}_{grp}_tautau_os_iso.txt'
                    n_comb += 1
                if n_comb == 1:
                    raise RuntimeError(f"Did not find any datacard, looked for \n{etau_file}")
                cmd += f' > {version}_{feature}_os_iso.txt'
                print(cmd)
                if run: os.chdir(combdir)
                if run: os.system(cmd)

                datacard_list.append(f'{combdir}/{version}_{feature}_os_iso.txt')
                root_list.append([etau_root, mutau_root, tautau_root])

                if "ZZ" in version:
                    r_range = "--rMin 0 --rMax 2"
                    r_range_setPR = "--setParameterRanges r=0,2"
                elif "ZbbHtt" in version or "ZttHbb" in version:
                    if "ZHKinFit_mass" in feature:
                        r_range = "--rMin -50 --rMax 50"
                        r_range_setPR = "--setParameterRanges r=-50,50"
                    else:
                        r_range = "--rMin -10 --rMax 15"
                        r_range_setPR = "--setParameterRanges r=-10,15"
                else:
                    raise ValueError("COuld not determine ZZ or ZH analysis")
            
                cmd = f'text2workspace.py {version}_{feature}_os_iso.txt -o model.root'
                if run: os.system(cmd)
                cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 --expectSignal 1 -t -1'
                if run: os.system(cmd)
                cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 --expectSignal 1 -t -1'
                if run: os.system(cmd)
                cmd = f'combine -M Significance {version}_{feature}_os_iso.txt -t -1 --expectSignal=1 &> Significance.log'
                if run: os.system(cmd)

                cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
                    ' --preFitValue 1 --expectSignal 1 -t -1 &>MultiDimFit.log'
                if run: os.system(cmd)
                cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
                    '--saveWorkspace --preFitValue 1 --expectSignal 1 -t -1 -n .scan.with_syst.statonly_correct --algo grid '\
                    '--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances &>MultiDimFit_statOnly.log'
                if run: os.system(cmd)

                #######################################################################
                #######################################################################
                #######################################################################

                LS_file_etau = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/etau/higgsCombineTest.MultiDimFit.mH120.root'
                if os.path.exists(LS_file_etau):
                    f_etau = ROOT.TFile(LS_file_etau)
                    limit_etau = f_etau.Get("limit")
                    to_draw_etau = ROOT.TString("2*deltaNLL:r")
                    n_etau = limit_etau.Draw( to_draw_etau.Data(), "", "l")

                    x_etau = np.array(np.ndarray((n_etau), 'd', limit_etau.GetV2())[1:])
                    y_etau = np.array(np.ndarray((n_etau), 'd', limit_etau.GetV1())[1:])

                    LS_file = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/etau/higgsCombineTest.Significance.mH120.significance.root'
                    f = ROOT.TFile(LS_file)
                    limit = f.Get("limit")
                    limit.GetEntry(0)
                    sig_etau = limit.limit

                else:
                    x_etau, y_etau = np.array([]), np.array([])
                    sig_etau = None
                    print("File not found:", LS_file_etau)

                LS_file_mutau = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/mutau/higgsCombineTest.MultiDimFit.mH120.root'
                if os.path.exists(LS_file_mutau):
                    f_mutau = ROOT.TFile(LS_file_mutau)
                    limit_mutau = f_mutau.Get("limit")
                    to_draw_mutau = ROOT.TString("2*deltaNLL:r")
                    n_mutau = limit_mutau.Draw( to_draw_mutau.Data(), "", "l")

                    x_mutau = np.array(np.ndarray((n_mutau), 'd', limit_mutau.GetV2())[1:])
                    y_mutau = np.array(np.ndarray((n_mutau), 'd', limit_mutau.GetV1())[1:])

                    LS_file = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/mutau/higgsCombineTest.Significance.mH120.significance.root'
                    f = ROOT.TFile(LS_file)
                    limit = f.Get("limit")
                    limit.GetEntry(0)
                    sig_mutau = limit.limit

                else:
                    x_mutau, y_mutau = np.array([]), np.array([])
                    sig_mutau = None
                    print("File not found:", LS_file_mutau)

                LS_file_tautau = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/tautau/higgsCombineTest.MultiDimFit.mH120.root'
                if os.path.exists(LS_file_tautau):
                    f_tautau = ROOT.TFile(LS_file_tautau)
                    limit_tautau = f_tautau.Get("limit")
                    to_draw_tautau = ROOT.TString("2*deltaNLL:r")
                    n_tautau = limit_tautau.Draw( to_draw_tautau.Data(), "", "l")

                    x_tautau = np.array(np.ndarray((n_tautau), 'd', limit_tautau.GetV2())[1:])
                    y_tautau = np.array(np.ndarray((n_tautau), 'd', limit_tautau.GetV1())[1:])

                    LS_file = maindir + f'/NonRes/{version}/{prd}/{feature}/{category}/tautau/higgsCombineTest.Significance.mH120.significance.root'
                    f = ROOT.TFile(LS_file)
                    limit = f.Get("limit")
                    limit.GetEntry(0)
                    sig_tautau = limit.limit

                else:
                    x_tautau, y_tautau = np.array([]), np.array([])
                    sig_tautau = None
                    print("File not found:", LS_file_tautau)

                LS_file_comb = combdir + f'/higgsCombineTest.MultiDimFit.mH120.root'
                f_comb= ROOT.TFile(LS_file_comb)
                limit_comb = f_comb.Get("limit")
                to_draw_comb = ROOT.TString("2*deltaNLL:r")
                n_comb = limit_comb.Draw( to_draw_comb.Data(), "", "l")

                x_comb = np.array(np.ndarray((n_comb), 'd', limit_comb.GetV2())[1:])
                y_comb = np.array(np.ndarray((n_comb), 'd', limit_comb.GetV1())[1:])

                LS_file_comb_stat = combdir + f'/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
                f_comb_stat= ROOT.TFile(LS_file_comb_stat)
                limit_comb_stat = f_comb_stat.Get("limit")
                to_draw_comb_stat = ROOT.TString("2*deltaNLL:r")
                n_comb_stat = limit_comb_stat.Draw( to_draw_comb_stat.Data(), "", "l")

                x_comb_stat = np.array(np.ndarray((n_comb_stat), 'd', limit_comb_stat.GetV2())[1:])
                y_comb_stat = np.array(np.ndarray((n_comb_stat), 'd', limit_comb_stat.GetV1())[1:])

                LS_file_comb_list.append(LS_file_comb)

                #######################################################################
                #######################################################################
                #######################################################################

                f, ax = plt.subplots(figsize = [10,10])
                plt.plot(x_etau, y_etau, linewidth=2, label=r'$\tau_{e}\tau_{h}$')
                plt.plot(x_mutau, y_mutau, linewidth=2, label=r'$\tau_{\mu}\tau_{h}$')
                plt.plot(x_tautau, y_tautau, linewidth=2, label=r'$\tau_{h}\tau_{h}$')
                plt.plot(x_comb, y_comb, linewidth=2, color='firebrick', label='Combination')
                plt.plot(x_comb_stat, y_comb_stat, linewidth=2, color='firebrick', linestyle='--', label='Stat-only')
                # central = round(x_comb[np.argmin(y_comb)], 3)
                r, up, down, up_stat, down_stat, up_syst, down_syst = GetLegend(x_comb, y_comb, x_comb_stat, y_comb_stat)
                text1 = fr"$\mu = 1.00^{{+{up}}}_{{-{down}}}$"
                plt.text(.02, .98, text1, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                text2 = fr"$\mu = 1.00^{{+{up_syst}}}_{{-{down_syst}}}(syst)^{{+{up_stat}}}_{{-{down_stat}}}(stat)$"
                plt.text(.02, .92, text2, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                plt.xlabel(r'$\mu$')
                plt.ylabel(r'-2$\Delta$ LL')
                plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
                plt.axhline(y=3.84, color='black', linestyle='--', alpha=0.5)
                plt.text(x=x_mutau[0], y=1.1, s="68% C.L.", fontsize="x-small")
                plt.text(x=x_mutau[0], y=3.94, s="95% C.L.", fontsize="x-small")
                plt.legend(fontsize=20, loc='upper right', frameon=True)
                if "ZZ" in version:
                    plt.xlim(0,2)
                    plt.ylim(-0.05, 1.1*np.max(y_comb))
                elif "ZbbHtt" in version or "ZttHbb" in version:
                    if "ZHKinFit_mass" in feature:
                        plt.xlim(-50, 50)
                    else: # DNN
                        plt.xlim(-10, 15)
                    plt.ylim(-0.05, 6)
                # mplhep.cms.label(data=False, rlabel='2018, (13.6 TeV) 59.7 $fb^{-1}$', fontsize=20)
                mplhep.cms.label(data=False, rlabel='(13 TeV)', fontsize=20)
                plt.grid()
                savefile = combdir + '/Combination_Mu'
                plt.savefig(savefile+'.png')
                plt.savefig(savefile+'.pdf')
                print(savefile+'.png')
                plt.close() 

                f, ax = plt.subplots(figsize = [10,10])
                xs=5.52*0.033658*0.1512
                plt.plot(xs*x_etau, y_etau, linewidth=2, label=r'$\tau_{e}\tau_{h}$')
                plt.plot(xs*x_mutau, y_mutau, linewidth=2, label=r'$\tau_{\mu}\tau_{h}$')
                plt.plot(xs*x_tautau, y_tautau, linewidth=2, label=r'$\tau_{h}\tau_{h}$')
                plt.plot(xs*x_comb, y_comb, linewidth=2, color='firebrick', label='Combination')
                plt.plot(xs*x_comb_stat, y_comb_stat, linewidth=2, color='firebrick', linestyle='--', label='Stat-only')
                r, up, down, up_stat, down_stat, up_syst, down_syst = GetLegend(xs*x_comb, y_comb, xs*x_comb_stat, y_comb_stat)
                text1 = fr"$\sigma = {{{r}}}^{{+{up}}}_{{-{down}}}$ pb"
                plt.text(.02, .98, text1, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                text2 = fr"$\sigma = {{{r}}}^{{+{up_syst}}}_{{-{down_syst}}}(syst)^{{+{up_stat}}}_{{-{down_stat}}}(stat)$ pb"
                plt.text(.02, .92, text2, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                if "ZZ" in version:
                    plt.xlabel(r'$\sigma\;(ZZ \rightarrow bb\tau\tau)$ [pb]')
                elif "ZbbHtt" in version:
                    plt.xlabel(r'$\sigma\;(ZH \rightarrow bb\tau\tau)$ [pb]')
                elif "ZttHbb" in version:
                    plt.xlabel(r'$\sigma\;(ZH \rightarrow \tau\tau bb)$ [pb]')
                plt.ylabel(r'-2$\Delta$ LL')
                plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
                plt.axhline(y=3.84, color='black', linestyle='--', alpha=0.5)
                plt.text(x=xs*x_mutau[0], y=1.1, s="68% C.L.", fontsize="x-small")
                plt.text(x=xs*x_mutau[0], y=3.94, s="95% C.L.", fontsize="x-small")
                plt.legend(fontsize=20, loc='upper right', frameon=True)
                if "ZZ" in version:
                    plt.ylim(-0.05, 1.1*np.max(y_comb))
                elif "ZbbHtt" in version or "ZttHbb" in version:
                    plt.ylim(-0.05, 6)
                
                # mplhep.cms.label(data=False, rlabel='2018, (13.6 TeV) 59.7 $fb^{-1}$', fontsize=20)
                mplhep.cms.label(data=False, rlabel='(13 TeV)', fontsize=20)
                plt.grid()
                savefile = combdir + '/Combination_Sigma'
                plt.savefig(savefile+'.png')
                plt.savefig(savefile+'.pdf')
                print(savefile+'.png')
                plt.close() 

                LS_file = f'{combdir}/higgsCombineTest.Significance.mH120.root'
                f = ROOT.TFile(LS_file)
                limit = f.Get("limit")
                limit.GetEntry(0)
                b = limit.limit

                # print(" ### INFO: Produce impact plots")
                # cmd = 'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 --setParameterRanges r=0,2 --doInitialFit --robustFit 1'
                # if run: os.system(cmd)
                # cmd = 'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 --setParameterRanges r=0,2 --doFits --robustFit 1'
                # if run: os.system(cmd)
                # cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts.json'
                # if run: os.system(cmd)
                # cmd = 'plotImpacts.py -i impacts.json -o impacts'
                # if run: os.system(cmd)
                # if run: os.system('mkdir -p impacts')
                # if run: os.system('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts')

                print(" ### INFO: Results for combination")
                print(" ### significance etau = ", sig_etau)
                print(" ### significance mutau = ", sig_mutau)
                print(" ### significance tautau = ", sig_tautau)
                print(" ### significance combination = ", b)

                # print(GetLegend(x_etau, y_etau, x_etau, y_etau, round=5))
                # print(GetLegend(x_mutau, y_mutau, x_mutau, y_mutau, round=5))
                # print(GetLegend(x_tautau, y_tautau, x_tautau, y_tautau, round=5))

            # Combine all categories for same year
            combdir_year = maindir + f'/NonRes/{version}/{prd}/{feature}/Combination'
            os.system(f'mkdir -p {combdir_year}')
                
            cmd = 'combineCards.py'
            n_comb = 1
            for datacard, root_file, category in zip(datacard_list, root_list, categories):
                cmd += f' {category}={datacard}'
                n_comb += 1
            cmd += f' > CombCategories_{feature}_os_iso.txt'
            print(cmd)
            if run: os.chdir(combdir_year)
            if run: os.system(cmd)

            if "ZZ" in version:
                r_range = "--rMin 0 --rMax 2"
                r_range_setPR = "--setParameterRanges r=0,2"
            elif "ZbbHtt" in version or "ZttHbb" in version:
                if "ZHKinFit_mass" in feature:
                    r_range = "--rMin -50 --rMax 50"
                    r_range_setPR = "--setParameterRanges r=-50,50"
                else: # DNN
                    r_range = "--rMin -10 --rMax 15"
                    r_range_setPR = "--setParameterRanges r=-10,15"
                

            cmd = f'text2workspace.py CombCategories_{feature}_os_iso.txt -o model.root'
            if run: os.system(cmd)
            cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 --expectSignal 1 -t -1'
            if run: os.system(cmd)
            cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 --expectSignal 1 -t -1'
            if run: os.system(cmd)
            cmd = f'combine -M Significance CombCategories_{feature}_os_iso.txt -t -1 --expectSignal=1 &> Significance.log'
            if run: os.system(cmd)

            cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
                ' --preFitValue 1 --expectSignal 1 -t -1'
            if run: os.system(cmd)
            cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
                '--saveWorkspace --preFitValue 1 --expectSignal 1 -t -1 -n .scan.with_syst.statonly_correct --algo grid '\
                '--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances'
            if run: os.system(cmd)
            


        continue

        #######################################################################
        #######################################################################
        #######################################################################
        
        # combine all years for the same feature
            
        combdir_run2 = maindir + f'/NonRes/FullRun2/{grp}/{prd}/{feature}/Combination'
        os.system(f'mkdir -p {combdir_run2}')
            
        cmd = 'combineCards.py'
        n_comb = 1
        for datacard, root_file, year in zip(datacard_list, root_list, years_list):
            cmd += f' Year{year}={datacard}'
            n_comb += 1
        cmd += f' > FullRun2_{feature}_os_iso.txt'
        print(cmd)
        if run: os.chdir(combdir_run2)
        if run: os.system(cmd)

        if "ZZ" in version:
            r_range = "--rMin 0 --rMax 2"
            r_range_setPR = "--setParameterRanges r=0,2"
        elif "ZbbHtt" in version or "ZttHbb" in version:
            if "ZHKinFit_mass" in feature:
                r_range = "--rMin -50 --rMax 50"
                r_range_setPR = "--setParameterRanges r=-50,50"
            else: # DNN
                r_range = "--rMin -10 --rMax 15"
                r_range_setPR = "--setParameterRanges r=-10,15"
            

        cmd = f'text2workspace.py FullRun2_{feature}_os_iso.txt -o model.root'
        if run: os.system(cmd)
        cmd = f'combine -M MultiDimFit model.root --algo=singles {r_range} --preFitValue 1 --expectSignal 1 -t -1'
        if run: os.system(cmd)
        cmd = f'combine -M MultiDimFit model.root --algo=grid --points 100 {r_range} --preFitValue 1 --expectSignal 1 -t -1'
        if run: os.system(cmd)
        cmd = f'combine -M Significance FullRun2_{feature}_os_iso.txt -t -1 --expectSignal=1 &> Significance.log'
        if run: os.system(cmd)

        cmd = f'combine -M MultiDimFit model.root -m 125 -n .bestfit.with_syst {r_range_setPR} --saveWorkspace'\
            ' --preFitValue 1 --expectSignal 1 -t -1'
        if run: os.system(cmd)
        cmd = f'combine -M MultiDimFit higgsCombine.bestfit.with_syst.MultiDimFit.mH125.root {r_range_setPR} '\
            '--saveWorkspace --preFitValue 1 --expectSignal 1 -t -1 -n .scan.with_syst.statonly_correct --algo grid '\
            '--points 100 --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances'
        if run: os.system(cmd)

        # pdb.set_trace()

        LS_file_1 = LS_file_comb_list[0] #combdir + f'/higgsCombineTest.MultiDimFit.mH120.root'
        f_1= ROOT.TFile(LS_file_1)
        limit_1 = f_1.Get("limit")
        to_draw_1 = ROOT.TString("2*deltaNLL:r")
        n_1 = limit_1.Draw( to_draw_1.Data(), "", "l")
        x_1 = np.array(np.ndarray((n_1), 'd', limit_1.GetV2())[1:])
        y_1 = np.array(np.ndarray((n_1), 'd', limit_1.GetV1())[1:])
        LS_file_2 = LS_file_comb_list[1]
        f_2= ROOT.TFile(LS_file_2)
        limit_2 = f_2.Get("limit")
        to_draw_2 = ROOT.TString("2*deltaNLL:r")
        n_2 = limit_2.Draw( to_draw_2.Data(), "", "l")
        x_2 = np.array(np.ndarray((n_2), 'd', limit_2.GetV2())[1:])
        y_2 = np.array(np.ndarray((n_2), 'd', limit_2.GetV1())[1:])
        LS_file_3 = LS_file_comb_list[2]
        f_3= ROOT.TFile(LS_file_3)
        limit_3 = f_3.Get("limit")
        to_draw_3 = ROOT.TString("2*deltaNLL:r")
        n_3 = limit_3.Draw( to_draw_3.Data(), "", "l")
        x_3 = np.array(np.ndarray((n_3), 'd', limit_3.GetV2())[1:])
        y_3 = np.array(np.ndarray((n_3), 'd', limit_3.GetV1())[1:])
        LS_file_4 = LS_file_comb_list[3]
        f_4= ROOT.TFile(LS_file_4)
        limit_4 = f_4.Get("limit")
        to_draw_4 = ROOT.TString("2*deltaNLL:r")
        n_4 = limit_4.Draw( to_draw_4.Data(), "", "l")
        x_4 = np.array(np.ndarray((n_4), 'd', limit_4.GetV2())[1:])
        y_4 = np.array(np.ndarray((n_4), 'd', limit_4.GetV1())[1:])

        LS_file_run2 = combdir_run2 + f'/higgsCombineTest.MultiDimFit.mH120.root'
        f_run2= ROOT.TFile(LS_file_run2)
        limit_run2 = f_run2.Get("limit")
        to_draw_run2 = ROOT.TString("2*deltaNLL:r")
        n_run2 = limit_run2.Draw( to_draw_run2.Data(), "", "l")

        x_run2 = np.array(np.ndarray((n_run2), 'd', limit_run2.GetV2())[1:])
        y_run2 = np.array(np.ndarray((n_run2), 'd', limit_run2.GetV1())[1:])

        LS_file_run2_stat = combdir_run2 + f'/higgsCombine.scan.with_syst.statonly_correct.MultiDimFit.mH120.root'
        f_run2_stat= ROOT.TFile(LS_file_run2_stat)
        limit_run2_stat = f_run2_stat.Get("limit")
        to_draw_run2_stat = ROOT.TString("2*deltaNLL:r")
        n_run2_stat = limit_run2_stat.Draw( to_draw_run2_stat.Data(), "", "l")

        x_run2_stat = np.array(np.ndarray((n_run2_stat), 'd', limit_run2_stat.GetV2())[1:])
        y_run2_stat = np.array(np.ndarray((n_run2_stat), 'd', limit_run2_stat.GetV1())[1:])

        f, ax = plt.subplots(figsize = [10,10])
        plt.plot(x_1, y_1, linewidth=2, label=r'2016')
        plt.plot(x_2, y_2, linewidth=2, label=r'2016 HIPM')
        plt.plot(x_3, y_3, linewidth=2, label=r'2017')
        plt.plot(x_4, y_4, linewidth=2, label=r'2018', color='Purple')
        plt.plot(x_run2, y_run2, linewidth=2, color='firebrick', label='Full Run 2')
        plt.plot(x_run2_stat, y_run2_stat, linewidth=2, color='firebrick', linestyle='--', label='Stat-only')
        # central = round(x_comb[np.argmin(y_comb)], 3)
        r, up, down, up_stat, down_stat, up_syst, down_syst = GetLegend(x_run2, y_run2, x_run2_stat, y_run2_stat)
        text1 = fr"$\mu = 1.00^{{+{up}}}_{{-{down}}}$"
        plt.text(.02, .98, text1, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        text2 = fr"$\mu = 1.00^{{+{up_syst}}}_{{-{down_syst}}}(syst)^{{+{up_stat}}}_{{-{down_stat}}}(stat)$"
        plt.text(.02, .92, text2, ha='left', va='top', transform=ax.transAxes, fontsize='small', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'-2$\Delta$ LL')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=3.84, color='black', linestyle='--', alpha=0.5)
        plt.text(x=x_1[0], y=1.1, s="68% C.L.", fontsize="x-small")
        plt.text(x=x_1[0], y=3.94, s="95% C.L.", fontsize="x-small")
        plt.legend(fontsize=20, loc='upper right', frameon=True)
        if "ZZ" in version:
            plt.xlim(0,2)
            plt.ylim(-0.05, 1.1*np.max(y_comb))
        elif "ZbbHtt" in version or "ZttHbb" in version:
            if "ZHKinFit_mass" in feature:
                plt.xlim(-50, 50)
            else: # DNN
                plt.xlim(-10, 15)
            plt.ylim(-0.05, 6)
        # mplhep.cms.label(data=False, rlabel='2018, (13.6 TeV) 59.7 $fb^{-1}$', fontsize=20)
        mplhep.cms.label(data=False, rlabel='(13 TeV)', fontsize=20)
        plt.grid()
        savefile = combdir_run2 + '/Combination_Mu'
        plt.savefig(savefile+'.png')
        plt.savefig(savefile+'.pdf')
        print(savefile+'.png')
        plt.close() 

        print(" ### INFO: Produce Full Run 2 impact plots")
        cmd = f'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' 
        if run: os.system(cmd)
        cmd = f'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 {r_range_setPR} --doFits --robustFit 1 --parallel 50'
        if run: os.system(cmd)
        cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts.json --parallel 50'
        if run: os.system(cmd)
        cmd = 'plotImpacts.py -i impacts.json -o impacts'
        if run: os.system(cmd)
        if run: os.system('mkdir -p impacts')
        if run: os.system('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts')

        cmd = f'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 {r_range_setPR} --doInitialFit --robustFit 1 --parallel 50 ' +  r" --exclude 'rgx{prop_bin.+}'"
        if run: os.system(cmd)
        cmd = f'combineTool.py -M Impacts -d model.root -m 125 --expectSignal 1 -t -1 --preFitValue 1 {r_range_setPR} --doFits --robustFit 1 --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
        if run: os.system(cmd)
        cmd = 'combineTool.py -M Impacts -d model.root -m 125 -o impacts_noMCstats.json --parallel 50'+  r" --exclude 'rgx{prop_bin.+}'"
        if run: os.system(cmd)
        cmd = 'plotImpacts.py -i impacts_noMCstats.json -o impacts_noMCstats'
        if run: os.system(cmd)
        if run: os.system('mkdir -p impacts_noMCstats')
        if run: os.system('mv higgsCombine_paramFit* higgsCombine_initialFit* impacts_noMCstats')