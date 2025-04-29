import os,json
import ROOT
import numpy as np
ROOT.gROOT.SetBatch(True)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--config",   help="Configuration name (ex : ul_2018_ZZ_v12)")
    parser.add_option("--mass",     dest="mass",     default='200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,3000')
    parser.add_option("--feat",     dest="feat",     default='ZZKinFit_mass')
    parser.add_option("--ver",      dest="ver",      default='prod_231129')
    parser.add_option("--ch",       dest="ch",       default='combination')
    parser.add_option("--featureDependsOnMass", help="Add _$MASS to name of feature for each mass -> for parametrized DNN", action="store_true", default=False)
    (options, args) = parser.parse_args()

    if ',' in options.mass:
        mass_points = options.mass.split(',')
    else:
        mass_points = [options.mass]
    
    feat = options.feat
    ver = options.ver
    ch = options.ch

    dirs = [f'{ver}_M{mass}' for mass in mass_points]

    maindir = os.getcwd() + '/ResLimits'
    os.system('mkdir -p ' + maindir)

    mass_dict = []
    
    mass  = []
    exp   = []
    m1s_t = []
    p1s_t = []
    m2s_t = []
    p2s_t = []

    for dir, mass_fromDir in zip(dirs, mass_points):
        feat_ver = f'{feat}_{mass_fromDir}' if options.featureDependsOnMass else feat
        limit_file = maindir + f'/{ch}/{feat_ver}/limits.json'
        with open(limit_file, 'r') as json_file:
            mass_dict = json.load(json_file)
        m = list(mass_dict.keys())[0].split("_M")[1]
        first_key = list(mass_dict.keys())[0]
        mass.append(m)
        exp.append(mass_dict[first_key]['exp'])
        m1s_t.append(mass_dict[first_key]['m1s_t'])
        p1s_t.append(mass_dict[first_key]['p1s_t'])
        m2s_t.append(mass_dict[first_key]['m2s_t'])
        p2s_t.append(mass_dict[first_key]['p2s_t'])

    mass  = np.array(mass, dtype=float)
    exp   = np.array(exp, dtype=float)
    m1s_t = np.array(m1s_t, dtype=float)
    p1s_t = np.array(p1s_t, dtype=float)
    m2s_t = np.array(m2s_t, dtype=float)
    p2s_t = np.array(p2s_t, dtype=float)  
        
    G_exp = ROOT.TGraph()
    G_sig1 = ROOT.TGraphAsymmErrors()
    G_sig2 = ROOT.TGraphAsymmErrors()

    ipt = 0
    for m, e, m1_t, p1_t, m2_t, p2_t in zip(mass, exp, m1s_t, p1s_t, m2s_t, p2s_t):

        p2 = p2_t - e
        p1 = p1_t - e
        m2 = e - m2_t
        m1 = e - m1_t

        G_exp.SetPoint(ipt, m, e)
        G_sig1.SetPoint(ipt, m, e)
        G_sig1.SetPointError(ipt, 0, 0, m1, p1)
        G_sig2.SetPoint(ipt, m, e)
        G_sig2.SetPointError(ipt, 0, 0, m2, p2)

        ipt += 1

    G_exp.SetMarkerStyle(24)
    G_exp.SetMarkerColor(4)
    G_exp.SetMarkerSize(0.8)
    G_exp.SetLineColor(ROOT.kBlack)
    G_exp.SetLineWidth(3)
    G_exp.SetLineStyle(2)
    G_exp.SetFillColor(0)

    G_sig1.SetMarkerStyle(0)
    G_sig1.SetMarkerColor(3)
    G_sig1.SetFillColor(ROOT.kGreen+1)
    G_sig1.SetLineColor(ROOT.kGreen+1)
    G_sig1.SetFillStyle(1001)
    
    G_sig2.SetMarkerStyle(0)
    G_sig2.SetMarkerColor(5)
    G_sig2.SetFillColor(ROOT.kOrange)
    G_sig2.SetLineColor(ROOT.kOrange)
    G_sig2.SetFillStyle(1001)

    canvas = ROOT.TCanvas('canvas', 'canvas', 650, 500)
    canvas.SetFrameLineWidth(3)
    canvas.SetBottomMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetLeftMargin(0.15)
    canvas.SetGridx()
    canvas.SetGridy()

    # Outside frame
    x_min = np.min(mass) - 30
    x_max = np.max(mass) + 30
    frame_bounds = x_min, x_max
    hframe = ROOT.TH1F('hframe', '',
                       100, frame_bounds[0], frame_bounds[1])
    hframe.SetMinimum(0.1)
    hframe.GetYaxis().SetRangeUser(0.005,5)

    hframe.GetYaxis().SetTitleSize(0.047)
    hframe.GetXaxis().SetTitleSize(0.055)
    hframe.GetYaxis().SetLabelSize(0.045)
    hframe.GetXaxis().SetLabelSize(0.045)
    hframe.GetXaxis().SetLabelOffset(0.012)
    hframe.GetYaxis().SetTitleOffset(1.2)
    hframe.GetXaxis().SetTitleOffset(1.1)
    if "ZZ" in options.config:
        process_tex = "X#rightarrowZZ#rightarrow bb#tau#tau"
        hframe.GetXaxis().SetTitle("m_{X} [GeV]")
    elif "ZbbHtt" in options.config:
        process_tex = "Z'#rightarrowZH#rightarrow bb#tau#tau"
        hframe.GetXaxis().SetTitle("m_{Z'} [GeV]")
    elif "ZttHbb" in options.config:
        process_tex = "Z'#rightarrowZH#rightarrow #tau#tau bb"
        hframe.GetXaxis().SetTitle("m_{Z'} [GeV]")
    else:
        process_tex = "unknown process"
    hframe.GetYaxis().SetTitle("95% CL on #sigma #times #bf{#it{#Beta}}(" + process_tex + ") [pb]")
    
    hframe.SetStats(0)
    ROOT.gPad.SetTicky()
    hframe.Draw()
    hframe.GetYaxis().SetRangeUser(0.005,5)

    ptext1 = ROOT.TPaveText(0.1663218-0.02, 0.886316, 0.3045977-0.02, 0.978947, 'brNDC')
    ptext1.SetBorderSize(0)
    ptext1.SetTextAlign(12)
    ptext1.SetTextFont(62)
    ptext1.SetTextSize(0.05)
    ptext1.SetFillColor(0)
    ptext1.SetFillStyle(0)
    ptext1.AddText('CMS #font[52]{Internal}')
    ptext1.Draw()

    ptext2 = ROOT.TPaveText(0.74, 0.91, 0.85, 0.95, 'brNDC')
    ptext2.SetBorderSize(0)
    ptext2.SetFillColor(0)
    ptext2.SetTextSize(0.040)
    ptext2.SetTextFont(42)
    ptext2.SetFillStyle(0)
    ptext2.AddText('2018 - 59.7 fb^{-1} (13 TeV)')
    ptext2.Draw()

    legend = ROOT.TLegend(0.7,0.7,0.9,0.89)
    legend.SetBorderSize(0)
    legend.AddEntry(G_sig1, '68% exp.', 'f')
    legend.AddEntry(G_sig2, '95% exp.', 'f')
    legend.AddEntry(G_exp, 'Expected', 'l')
    G_sig2.Draw('3same')
    G_sig1.Draw('3same')
    G_exp.Draw('Lsame')
    
    canvas.Update()
    canvas.SetLogy()

    legend.Draw()

    save_name = maindir + f'/Limits_'+feat+'_'+ver+'_'+ch
    canvas.SaveAs(save_name+'.png')
    canvas.SaveAs(save_name+'.pdf')
