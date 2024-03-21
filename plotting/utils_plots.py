200, 0, 1#import uproot
#import awkward as ak
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D, TH2D, TH2F, TH1F, TCanvas, TGraph, TMultiGraph, TLegend, TColor, TLatex
import math
import pandas as pd
from root_numpy import fill_hist  as fh
from root_numpy import array2hist as a2h
from root_numpy import hist2array as h2a
from root_numpy import tree2array
from scipy.stats import entropy
from rootplotting import ap
from rootplotting.tools import *
from scipy.stats import pearsonr
from operator import itemgetter
MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

def convert_graph_to_hist(graph, bins=100, x_min=None, x_max=None):
    # Create a histogram with specified number of bins and range
    if x_min is None:
        x_min = graph.GetXaxis().GetXmin()
    if x_max is None:
        x_max = graph.GetXaxis().GetXmax()

    hist = ROOT.TH1F("hist", "Histogram from TGraph", bins, x_min, x_max)

    # Get data points from the TGraph and fill the histogram
    for i in range(graph.GetN()):
        x = ROOT.Double(0)
        y = ROOT.Double(0)
        graph.GetPoint(i, x, y)
        hist.Fill(x)

    return hist

def get_xy_values_from_tgraph(graph):
    num_points = graph.GetN()
    x_values = []
    y_values = []

    for i in range(num_points):
        x = ROOT.Double(0)
        y = ROOT.Double(0)
        graph.GetPoint(i, x, y)
        x_values.append(x)
        y_values.append(y)
    matrix = np.array([x_values, y_values]).T
    matrix = np.array(sorted(matrix, key=itemgetter(0)))
    return matrix[:,0], matrix[:,1]

def kl_divergence(histogram1, histogram2):
    kl_div = 0.0
    for i in range(1, histogram1.GetNbinsX()+1):
        p = histogram1.GetBinContent(i)
        q = histogram2.GetBinContent(i)

        # Avoid division by zero
        if p != 0.0 and q != 0.0:
            kl_div += p * math.log(p / q)

    return kl_div

def jsd_divergence(histogram1, histogram2):
    # Calculate the average distribution (M)
    avg_histogram = histogram1.Clone()
    avg_histogram.Add(histogram2)
    avg_histogram.Scale(0.5)

    # Calculate the KL Divergences
    kl_div_p_m = kl_divergence(histogram1, avg_histogram)
    kl_div_q_m = kl_divergence(histogram2, avg_histogram)

    # Calculate the JSD
    jsd_value = 0.5 * kl_div_p_m + 0.5 * kl_div_q_m

    return jsd_value

# def MassCutLow50(x):
#     return  (69.8199410326) + (0.0150081803314)*pow(x,1) + (-6.17224460098e-05)*pow(x,2) + (7.86154778779e-08)*pow(x,3) + (-4.97257435999e-11)*pow(x,4) + (1.50129847267e-14)*pow(x,5) + (-1.72903388714e-18)*pow(x,6);
#
# def MassCutHigh50(x):
#     return  (165.048442698) + (-0.364378464009)*pow(x,1) + (0.000723923415283)*pow(x,2) + (-7.12212365612e-07)*pow(x,3) + (3.66502330413e-10)*pow(x,4) + (-9.36551583203e-14)*pow(x,5) + (9.37467300706e-18)*pow(x,6);

def MassCutLow50(x):
    return (70.7028111112)+(-0.0067474504758)*pow(x,1)+(1.46396493452e-05)*pow(x,2)+(-1.36289152607e-08)*pow(x,3)+(9.47767607532e-13)*pow(x,4)+(1.94842183915e-15)*pow(x ,5)+(-4.4112166542e-19)*pow(x ,6)
    # return (87.7801)+(-0.115962)*pow(x,1)+(0.000268137)*pow(x,2)+(-2.97262e-07)*pow(x,3)+(1.63808e-10)*pow(x,4)+(-4.402e-14)*pow(x,5)+(4.59902e-18)*pow(x,6)
    # return (69.8199410326) + (0.0150081803314)*pow(x,1) + (-6.17224460098e-05)*pow(x,2) + (7.86154778779e-08)*pow(x,3) + (-4.97257435999e-11)*pow(x,4) + (1.50129847267e-14)*pow(x,5) + (-1.72903388714e-18)*pow(x,6)
def MassCutHigh50(x):
    return (66.7831332301)+(0.0458904153319)*pow(x,1)+(-4.63040529804e-05)*pow(x,2)+(2.21486704013e-08)*pow(x,3)+(-3.42485672162e-12)*pow(x,4)+(8023.93536723)*pow(x ,-1)
    # return (139.702)+(-0.206293)*pow(x,1)+(0.000365374)*pow(x,2)+(-3.31522e-07)*pow(x,3)+(1.61432e-10)*pow(x,4)+(-3.93562e-14)*pow(x,5)+(3.76831e-18)*pow(x,6)
    # return (165.048442698) + (-0.364378464009)*pow(x,1) + (0.000723923415283)*pow(x,2) + (-7.12212365612e-07)*pow(x,3) + (3.66502330413e-10)*pow(x,4) + (-9.36551583203e-14)*pow(x,5) + (9.37467300706e-18)*pow(x,6)
def MassCutLow80(x):
    return (79.5500134182)+(-0.0734677572022)*pow(x,1)+(0.000107937560498)*pow(x,2)+(-7.43444281905e-08)*pow(x,3)+(2.24232950741e-11)*pow(x,4)+(-2.47470101128e-15)*pow(x,5)
def MassCutHigh80(x):
    return (157.846131378)+(-0.164760797665)*pow(x,1)+(0.000179183802128)*pow(x,2)+(-8.80761966242e-08)*pow(x,3)+(2.0622297396e-11)*pow(x,4)+(-1.81688990961e-15)*pow(x,5)

def roc_from_histos_DAVIDE(h_sig, h_bkg, h_sigCut, h_bkgCut, h_sig_NOWEIGHTS, h_bkg_NOWEIGHTS, h_sigCut_NOWEIGHTS, h_bkgCut_NOWEIGHTS,  wpcut=None, massCut=False):
    tprs = []
    fprs = []
    num_bins = h_sig.GetNbinsX()
    n_sig = h_sig.Integral(1,num_bins+1)
    n_bkg = h_bkg.Integral(1,num_bins+1)

    if wpcut:
        i_wpbin = h_sig.GetXaxis().FindBin(wpcut)

    tpr_wp = 0
    fpr_wp = 0

    n_tp_total = h_sigCut.Integral(1, num_bins+1)
    n_fp_total = h_bkgCut.Integral(1, num_bins+1)

    n_tp_total_NOWEIGHTS = h_sig_NOWEIGHTS.Integral(1, num_bins+1)
    n_fp_total_NOWEIGHTS = h_bkg_NOWEIGHTS.Integral(1, num_bins+1)

    n_tpCUT_total_NOWEIGHTS = h_sigCut_NOWEIGHTS.Integral(1, num_bins+1)
    n_fpCUT_total_NOWEIGHTS = h_bkgCut_NOWEIGHTS.Integral(1, num_bins+1)


    for i_bin in range(1, num_bins+2):

        if massCut==False:
            n_tp = h_sig.Integral(i_bin, num_bins+1)
            n_fp = h_bkg.Integral(i_bin, num_bins+1)

            tpr = n_tp / n_sig
            fpr = n_fp / n_bkg

        else:
            n_tp = h_sigCut.Integral(i_bin, num_bins+1)
            n_fp = h_bkgCut.Integral(i_bin, num_bins+1)

            ## the expresion Davide try to use is : (NORMAL ROC CURVE) * EFF_DUE_TO_MASS_CUTS
            #tpr = (n_tp/n_tp_total ) * ( n_tp_total/n_sig) == n_tp / n_sig   # this is the same we have

            ## HOWEVER, in ( n_fp_total/n_sig) he didn't include weights so the expresion is:

            tpr = (n_tp/n_tp_total ) * (n_tpCUT_total_NOWEIGHTS / n_tp_total_NOWEIGHTS)
            fpr = (n_fp / n_fp_total ) * (n_fpCUT_total_NOWEIGHTS / n_fp_total_NOWEIGHTS)


        tprs.append(tpr)
        fprs.append(fpr)

        if wpcut and i_bin == i_wpbin:
            tpr_wp = tpr
            fpr_wp = fpr

    a_tprs = np.array(tprs)
    a_fprs = np.array(fprs)

    #get area under curve
    auc = np.abs( np.trapz(a_tprs, a_fprs) )

    #flip curve if auc is negative
    if auc < 0.5 and massCut==False:
        a_tprs = 1 - a_tprs
        a_fprs = 1 - a_fprs
        auc = 1 - auc

        if wpcut:
            tpr_wp = 1 - tpr_wp
            fpr_wp = 1 - fpr_wp

    if wpcut:
        return a_tprs, a_fprs, auc, tpr_wp, fpr_wp

    else:
        return a_tprs, a_fprs, auc



def GetBinominalErrors(numerator_hist, denominator_hist):
    rejection_hist = numerator_hist.Clone("background_rejection_hist")
    # Loop over the bins
    for bin in range(1, numerator_hist.GetNbinsX() + 1):
        # Get the bin content for numerator and denominator histograms
        numerator = numerator_hist.GetBinContent(bin)
        denominator = denominator_hist.GetBinContent(bin)

        # Calculate the background rejection and the binomial error
        rejection = 0.0
        error = 0.0
        if denominator != 0:
            rejection = numerator / denominator
            error = rejection * ROOT.TMath.Sqrt((1 - rejection) / denominator)

        # Set the rejection and the error in the rejection histogram
        rejection_hist.SetBinContent(bin, rejection)
        rejection_hist.SetBinError(bin, error)
    return rejection_hist

def roc_from_histos(h_sig, h_bkg, h_sigCut, h_bkgCut, wpcut=None, massCut=False):
    """
    Compute a ROC curve given two histograms.
    """

    tprs = []
    fprs = []

    c = ROOT.TCanvas('', '', 500, 500)
    c.SetLogy()
    h_sig.Draw('colz')
    c.SaveAs('h_sig.png')
    h_bkg.Draw('colz')
    c.SaveAs('h_bkg.png')
    #num_bins = h_sig.GetXaxis().GetNbins()
    num_bins = h_sig.GetNbinsX()

    n_sig = h_sig.Integral(1,num_bins+1)
    n_bkg = h_bkg.Integral(1,num_bins+1)

    if wpcut:
        i_wpbin = h_sig.GetXaxis().FindBin(wpcut)

    tpr_wp = 0
    fpr_wp = 0

    for i_bin in range(1, num_bins+2):

        if massCut==False:
            n_tp = h_sig.Integral(i_bin, num_bins+1)
            n_fp = h_bkg.Integral(i_bin, num_bins+1)
        else:
            n_tp = h_sigCut.Integral(i_bin, num_bins+1)
            n_fp = h_bkgCut.Integral(i_bin, num_bins+1)

        tpr = n_tp / n_sig
        fpr = n_fp / n_bkg

        tprs.append(tpr)
        fprs.append(fpr)

        if wpcut and i_bin == i_wpbin:
            tpr_wp = tpr
            fpr_wp = fpr

    a_tprs = np.array(tprs)
    a_fprs = np.array(fprs)

    #get area under curve
    auc = np.abs( np.trapz(a_tprs, a_fprs) )

    #flip curve if auc is negative
    if auc < 0.5 and massCut==False:
        a_tprs = 1 - a_tprs
        a_fprs = 1 - a_fprs
        auc = 1 - auc

        if wpcut:
            tpr_wp = 1 - tpr_wp
            fpr_wp = 1 - fpr_wp

    if wpcut:
        return a_tprs, a_fprs, auc, tpr_wp, fpr_wp

    else:
        return a_tprs, a_fprs, auc


dijet_xsweights_dict = {
    361022:   811423.536 *    1.0,
    361023:   8453.64024 *    1.0,
    361024:   134.9920945 *   1.0,
    361025:   4.19814486 *    1.0,
    361026:   0.241941709 *   1.0,
    361027:   0.006358874 *   1.0,
    361028:   0.006354782 *   1.0,
    361029:   0.000236819 *   1.0,
    361030:   7.054e-06 *     1.0,
    361031:   1.13e-07 *      1.0,
    361032:   4.405975e-10 *  1.0,

    364702: 2433000 * 0.0098631 / 1110002,
    364703: 26450 * 0.011658 / 1671907,
    364704: 254.61 * 0.013366 / 1839956,
    364705: 4.5529 * 0.014526 / 1435042,  # This is the only thing that changed due to extra root file for full
    364706: 0.25754 * 0.0094734 / 773626, # declustering
    364707: 0.016215 * 0.011097 / 960798,
    364708: 0.00062506 * 0.010156 / 1315619,
    364709: 1.9639E-05 * 0.012056 / 1082543,
    364710: 1.1962E-06 * 0.0058933 / 201761,
    364711: 4.2263E-08 * 0.002673 / 247582,
    364712: 1.0367E-09 * 0.00042889 / 288782,

    #ttbar
    426347:   1.0,
    426345:   1.0,
    -1: 1.0,
}


def assign_weights(mcid, mcweight):
    return dijet_xsweights_dict[mcid]*mcweight

def make_rocs(taggers, prefix=''):
    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    c1 = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if taggers[t].name == "3var":
            continue
        if taggers[t].name == "HerwigAngular" or taggers[t].name == "HerwigDipole" or taggers[t].name == "SherpaCluster" :
            continue
        tprs, fprs, auc = taggers[t].get_roc()

        ## Fill random guessing
        if count==0:
            h = TGraph(len(tprs), np.linspace(0.0001, 1, len(tprs)), np.linspace(0.0001, 1, len(tprs)))
            c1.graph(h, linestyle=2, linecolor=ROOT.kBlack, option="AL", label="Random")

        h = TGraph(len(tprs), fprs, tprs)
        c1.graph(h, linestyle=1, linecolor=colours[count], markercolor=colours[count], option="L", label=taggers[t].name)
        count+=1

    c1.xlabel('False positive rate')
    c1.ylabel('True positive rate')
    c1.xlim(0, 1) ## c1.xlim(0.2, 1)
    #c1.ylim(0, 1.2)

    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
            "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
            ], qualifier='Simulation Preliminary')
    c1.legend()
    c1.save("{}/ROCcurves.png".format(prefix))


def make_efficiencies_all(taggers, prefix=''):

    l=TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextSize(0.042)
    s=TLatex()
    s.SetNDC();
    s.SetTextFont(42)
    s.SetTextSize(0.04)
    legend=ROOT.TLegend(0.18,0.16,0.65,0.33)
    legend.SetNColumns(2)
    legend.Clear()
    legend.SetFillStyle(0)

    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982'), ROOT.kRed]

    count = 0
    c1 = ap.canvas(num_pads=1, batch=True)

    mg = TMultiGraph()

    ## LundNetNN
    tprs, fprs, auc = taggers["LundNet_class"].get_roc()
    inv = 1/fprs
    inv[inv == np.inf] = 1e10
    label = "LundNet^{NN}"
    h = TGraph(len(np.array(tprs, dtype=np.float64)), np.array(tprs, dtype=np.float64), np.array(inv, dtype=np.float64))
    print(len(np.array(tprs, dtype=np.float64)))
    h = c1.graph(h, linestyle=1,linewidth=2, linecolor=colours[0], markercolor=colours[0], markerstyle=1, label=label)
    mg.Add(h, "AL")
    # h.Draw()

    legend.AddEntry(h, "LundNet^{NN}", 'l')

    # file = ROOT.TFile.Open("/home/jmsardain/LJPTagger/ljptagger/Plotting/FromShudong/ROCs.root")
    # # Particle transformer
    # ROC_pt_full_Wtag_ParT = file.Get("ROC_pt_full_Wtag_ParT")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_ParT)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::1000], dtype=np.float64)), np.array(x_values[::1000], dtype=np.float64), np.array(y_values[::1000], dtype=np.float64))
    # h = c1.graph(h, linestyle=2,linewidth=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, label="Particle transfomer")
    # h.Draw('SAME')
    # legend.AddEntry(h, "ParT", 'l')
    #
    # # ParticleNet
    # ROC_pt_full_Wtag_PN = file.Get("ROC_pt_full_Wtag_PN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_PN)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::10000], dtype=np.float64)), np.array(x_values[::10000], dtype=np.float64), np.array(y_values[::10000], dtype=np.float64))
    # h = c1.graph(h, linestyle=3,linewidth=2, linecolor=colours[2], markercolor=colours[2], markerstyle=1,  label="ParticleNet")
    # h.Draw('SAME')
    # legend.AddEntry(h, "ParticleNet", 'l')
    #
    # ## PFN
    # ROC_pt_full_Wtag_PFN = file.Get("ROC_pt_full_Wtag_PFN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_PFN)
    # h = TGraph(len(np.array(x_values[::10000], dtype=np.float64)), np.array(x_values[::10000], dtype=np.float64), np.array(y_values[::10000], dtype=np.float64))
    # h = c1.graph(h, linestyle=4,linewidth=2, linecolor=colours[3], markercolor=colours[3], markerstyle=1,  label="PFN")
    # h.Draw('SAME')
    # legend.AddEntry(h, "PFN", 'l')
    #
    # ## EFN
    # ROC_pt_full_Wtag_EFN = file.Get("ROC_pt_full_Wtag_EFN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_EFN)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::20000], dtype=np.float64)), np.array(x_values[::20000], dtype=np.float64), np.array(y_values[::20000], dtype=np.float64))
    # h = c1.graph(h, linestyle=5,linewidth=2, linecolor=colours[4], markercolor=colours[4], markerstyle=1,  label="EFN")
    # h.Draw('SAME')
    # legend.AddEntry(h, "EFN", 'l')

    # ## DNN
    # h = TGraph(len(getANNROCresults("x", "NN")), getANNROCresults("x", "NN"), getANNROCresults("y", "NN"))
    # h = c1.graph(h, linestyle=6,linewidth=2, linecolor=colours[5], markercolor=colours[5], markerstyle=1,  label="z_{NN}")
    # h.Draw('SAME')
    # legend.AddEntry(h, "z_{NN}", 'l')
    mg.Draw()

    c1.xlabel('Signal efficiency (#varepsilon^{rel}_{sig})')
    c1.ylabel('Background rejection (1/#varepsilon^{rel}_{bkg})')
    c1.xlim(0.2,  1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 2e16)

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #         "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
    #         ], qualifier='Simulation Preliminary')

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #              "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
    #              "#scale[0.85]{Large-#it{R} jet #it{p}_{T} > 200 GeV}"
    #         ], qualifier='Simulation Preliminary')

    # c1.legend(xmin=0.7, xmax=0.9)
    c1.log()
    l.DrawLatex(0.18, 0.89,        "ATLAS")
    s.DrawLatex(0.18+(0.14), 0.89, " Simulation Preliminary")
    s.DrawLatex(0.18, 0.84,        "#sqrt{s} = 13 TeV, #it{top} tagging")
    s.DrawLatex(0.18, 0.79,        "anti-#it{k_{t}} #it{R}=1.0 UFO Soft-Drop CS+SK jets")
    s.DrawLatex(0.18, 0.74, "p_{T} > 200 GeV, |#eta| < 2.0")

    legend.Draw()
    c1.save("{}/fig_02a.png".format(prefix))
    c1.save("{}/fig_02a.pdf".format(prefix))
    c1.save("{}/fig_02a.eps".format(prefix))


def make_efficiencies_3var(taggers, prefix=''):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    count = 0
    c1 = ap.canvas(num_pads=1, batch=True)

    mg = TMultiGraph()

    ## LundNetNN
    tprs, fprs, auc = taggers["LundNet_class"].get_roc()
    inv = 1/fprs
    inv[inv == np.inf] = 1e10
    label = "LundNet^{NN}"
    h = TGraph(len(np.array(tprs, dtype=np.float64)), np.array(tprs, dtype=np.float64), np.array(inv, dtype=np.float64))
    # h = TGraph(len(np.array(tprs[::10], dtype=np.float64)), np.array(tprs[::10], dtype=np.float64), np.array(inv[::10], dtype=np.float64))
    # h = TGraph(len(np.array(tprs[::10], dtype=np.float64)), np.array(tprs[::10], dtype=np.float64), np.array(inv[::10], dtype=np.float64))
    # h = c1.graph(h, linestyle=1,linewidth=1, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="L", label=label)
    h = c1.graph(h, linestyle=1,linewidth=1, linecolor=colours[1], markercolor=colours[1], markerstyle=1, label=label)
    # h.SetLineStyle(1)
    # h.SetLineWidth(1)
    # h.SetLineColor(colours[1])
    # h.SetMarkerColor(colours[1])
    # h.SetMarkerStyle(1)
    a = h.Eval(0.5)
    print("hLundNetNN ")
    print(a)
    mg.Add(h, "L")

    # ## LundNetANN
    # tprs, fprs, auc = taggers["LundNet"].get_roc()
    # inv = 1/fprs
    # inv[inv == np.inf] = 1e10
    # label = "LundNet^{ANN}"
    # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
    # #h = TGraph(len(tprs), tprs, inv)
    # h = c1.graph(h, linestyle=2, linewidth=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="L", label=label)
    # # h.SetLineStyle(2)
    # # h.SetLineWidth(2)
    # # h.SetLineColor(colours[0])
    # # h.SetMarkerColor(colours[0])
    # # h.SetMarkerStyle(1)
    # a = h.Eval(0.5)
    # print("hLundNetANN ")
    # print(a)
    # mg.Add(h, "L")
    #
    # ## DNN
    # h = TGraph(len(getANNROCresults("x", "NN")), getANNROCresults("x", "NN"), getANNROCresults("y", "NN"))
    # h = c1.graph(h, linestyle=3,linewidth=1, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
    # # h.SetLineStyle(3)
    # # h.SetLineWidth(1)
    # # h.SetLineColor(ROOT.kRed)
    # # h.SetMarkerColor(ROOT.kRed)
    # # h.SetMarkerStyle(1)
    # a = h.Eval(0.5)
    # print("DNN ")
    # print(a)
    # mg.Add(h, "L")
    #
    # ## ANN
    # h = TGraph(len(getANNROCresults("x", "ANN")), getANNROCresults("x", "ANN"), getANNROCresults("y", "ANN"))
    # h = c1.graph(h, linestyle=4, linewidth=1, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
    # # h.SetLineStyle(4)
    # # h.SetLineStyle(1)
    # # h.SetLineColor(ROOT.kBlue)
    # # h.SetMarkerColor(ROOT.kBlue)
    # # h.SetMarkerStyle(1)
    # a = h.Eval(0.5)
    # print("ANN ")
    # print(a)
    # mg.Add(h, "L")



    mg.Draw()
    c1.xlabel('Signal efficiency')
    c1.ylabel('Background rejection')
    c1.xlim(0.2, 1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 1e5)

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #         "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
    #         ], qualifier='Simulation Preliminary')

    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 # "#scale[0.85]{Cut on m_{J} from 3-var tagger}"
            ], qualifier='Simulation Preliminary')
    c1.log()
    c1.legend(xmin=0.7, xmax=0.9)
    c1.save("{}/Efficiencies_taggers.png".format(prefix))
    c1.save("{}/Efficiencies_taggers.pdf".format(prefix))
    c1.save("{}/Efficiencies_taggers.eps".format(prefix))



def make_efficiencies_3var_massCut(taggers, prefix=''):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    count = 0
    c1 = ap.canvas(num_pads=1, batch=True)

    mg = TMultiGraph()

    ## LundNetNN
    tprs, fprs, auc = taggers["LundNet_class"].get_roc_mass()
    label = "LundNet^{NN}"
    ## protect for inf
    inv = 1/fprs
    inv[inv == np.inf] = 1e10

    ## create TGraph for each tagger
    hLundNetNN = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
    hLundNetNN = c1.graph(hLundNetNN, linestyle=1, linewidth=1, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
    mg.Add(hLundNetNN, "L")
    a = hLundNetNN.Eval(0.5)
    print("hLundNetNN ")
    print(a)
    ## LundNetANN
    tprs, fprs, auc = taggers["LundNet"].get_roc_mass()
    label = "LundNet^{ANN}"
    ## protect for inf
    inv = 1/fprs
    inv[inv == np.inf] = 1e10
    ## create TGraph for each tagger
    hLundNetANN = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
    hLundNetANN = c1.graph(hLundNetANN, linestyle=2, linewidth=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option=" L", label=label)
    mg.Add(hLundNetANN, "L")
    a = hLundNetANN.Eval(0.5)
    print("hLundNetANN ")
    print(a)
    ## Fill DNN (Davide)
    hDNN = TGraph(len(getANNROCresults_mass("x", "NN")), getANNROCresults_mass("x", "NN"), getANNROCresults_mass("y", "NN"))
    hDNN = c1.graph(hDNN, linestyle=3, linewidth=1, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option=" L", label="z_{NN}")
    mg.Add(hDNN, "L")
    a = hDNN.Eval(0.5)
    print("hDNN ")
    print(a)
    ## Fill ANN (Davide)
    hANN = TGraph(len(getANNROCresults_mass("x", "ANN")), getANNROCresults_mass("x", "ANN"), getANNROCresults_mass("y", "ANN"))
    hANN = c1.graph(hANN, linestyle=4, linewidth=1, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
    mg.Add(hANN, "L")
    a = hANN.Eval(0.5)
    print("hANN ")
    print(a)
    # Fill 3-var tagger
    hPoint = TGraph()
    # hPoint.SetPoint(0, 0.5072431, 40.86972)
    hPoint.SetPoint(0, 0.5072431, 64.8601)
    hPoint = c1.graph(hPoint, linecolor=ROOT.kWhite, markercolor=ROOT.kRed, markerstyle=20, option="P", label="3-var tagger")
    mg.Add(hPoint, "P")

    h = TGraph(len(tprs), np.linspace(0.00001, 1, len(tprs)), np.linspace(0.0001, 0.0001, len(tprs)))
    c1.graph(h, linestyle=2, linecolor=ROOT.kBlack, option="L")

    mg.Draw()


    c1.xlabel('Signal efficiency')
    c1.ylabel('Background rejection')
    c1.xlim(0.1, 1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 1e7)

    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 "#scale[0.85]{Cut on m_{J} from 3-var tagger}"
            ], qualifier='Simulation Preliminary')
    c1.log()
    c1.legend(xmin=0.7, xmax=0.9)
    c1.save("{}/Efficiencies_taggers_massCut.png".format(prefix))



def get_eff_score(pt_vs_score,wp):
    scores_projection = pt_vs_score.ProjectionX()
    pt_value = []
    tag_score = []
    for ptbin in range(1,pt_vs_score.GetNbinsX()+1):

        #print("pt bin->",scores_projection.GetBinCenter(ptbin), "pt_vs_score.ProjectionX().GetBinContent(ptbin)", pt_vs_score.ProjectionX().GetBinContent(ptbin)   )

        curcont = 0
        #pt_value.append(scores_projection.GetBinCenter(ptbin))
        if scores_projection.GetBinContent(ptbin)==0:
            tag_score.append(0)
            pt_value.append(scores_projection.GetBinCenter(ptbin))
            continue
        #for scorebin in range(1, pt_vs_score.GetNbinsX()):



        #for scorebin in range(1, pt_vs_score.GetNbinsY()+1):
        for scorebin in range( pt_vs_score.GetNbinsY(),0,-1 ):
            curcont += pt_vs_score.GetBinContent(ptbin, scorebin)
            if curcont/scores_projection.GetBinContent(ptbin) >= wp:
                tag_score.append(scorebin/400.)
                pt_value.append(scores_projection.GetBinCenter(ptbin))
                break
    return pt_value, tag_score


def get_eff_score_mass(pt_vs_score_mass, pt_vs_score_total, wp):
    scores_projection = pt_vs_score_mass.ProjectionX()
    scores_projection_total = pt_vs_score_total.ProjectionX()
    pt_value = []
    tag_score = []
    for ptbin in range(1, pt_vs_score_mass.GetNbinsX()+1):
        curcont = 0


        if scores_projection.GetBinContent(ptbin)==0:
            tag_score.append(0)
            pt_value.append(scores_projection.GetBinCenter(ptbin))
            continue


        #for scorebin in range(1, pt_vs_score_mass.GetNbinsY()+1): ## events NO pass mass cut = total events - events that pass mass cut
        for scorebin in range( pt_vs_score_mass.GetNbinsY(),0,-1 ): ##
            curcont += pt_vs_score_mass.GetBinContent(ptbin, scorebin)
            if curcont/( scores_projection_total.GetBinContent(ptbin) ) >= wp:
                if ptbin==1:
                    break
                    #tag_score.append(scorebin/100.)
                    #pt_value.append(scores_projection.GetBinCenter(ptbin))

                #tag_score.append(0.)
                tag_score.append(scorebin/100.)
                pt_value.append(scores_projection.GetBinCenter(ptbin))
                break

    return pt_value, tag_score



def wp50_cut(p,pt):
    return p[0]+p[1]/(p[2]+math.exp(p[3]*(pt+p[4])))


def get_wp_tag(tagger, wp, prefix=''):
    ROOT.gStyle.SetPalette(ROOT.kBird)
    h_pt_nn   = TH2D( "h_pt_nn{}".format(tagger.name), "h_pt_nn{}".format(tagger.name), 100, 0., 3000, 400,0,1 )
    h_pt_nn_mass50   = TH2D( "h_pt_nn_mass50{}".format(tagger.name), "h_pt_nn_mass50{}".format(tagger.name), 100, 0., 3000, 400,0,1 )
    for pt,nn,weight in zip(tagger.signal["fjet_pt"],tagger.signal["fjet_nnscore"],tagger.signal["fjet_weight_pt"]):
        h_pt_nn.Fill(pt,nn,weight)
    pts, scores = get_eff_score(h_pt_nn,wp)
    scores = scores[11:]
    print("Normal scores cuts->",scores)
    pts = pts [11:]
    gra = TGraph(len(pts), np.array(pts).astype("float"), np.array(scores).astype("float"))
    fitfunc = ROOT.TF1("fit", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 350, 2900) #exponential sigmoid fit (best so far)
    #fitfunc = root.TF1("fit", "pol10", 200, 2700) #12th order polynomial fit
    gra.Fit(fitfunc,"R,S")
    c = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)
    h_pt_nn.Draw('colz')
    c.SetRightMargin(0.2)
    c.SetLogz()
    c.SaveAs('2dplot.png')
    gra.Draw()

    p = fitfunc.GetParameters()
    tagger.scores["tag_cut"] = np.vectorize(lambda x:p[0]+p[1]/(p[2]+math.exp(p[3]*(x+p[4]))))(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[(tagger.bg.fjet_nnscore < tagger.bg.tag_cut) & (tagger.bg.fjet_nnscore>=0)]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]


    if wp ==0.5:

        for pt,nn,weight in zip(tagger.signalmass50["fjet_pt"], tagger.signalmass50["fjet_nnscore"], tagger.signalmass50["fjet_weight_pt"]):
            h_pt_nn_mass50.Fill(pt,nn,weight)

        ptsmass50, scoresmass50 = get_eff_score_mass(h_pt_nn_mass50, h_pt_nn, wp)
        print("my scoresmass50->", scoresmass50, "len(scoresmass50)->",len(scoresmass50) )
        #ptsmass50, scoresmass50 = get_eff_score(h_pt_nn_mass50,wp)

        scoresmass50 = scoresmass50[6:]
        ptsmass50 = ptsmass50 [6:]
        gramass50 = TGraph(len(ptsmass50), np.array(ptsmass50).astype("float"), np.array(scoresmass50).astype("float"))
        fitfuncmass50 = ROOT.TF1("fitfuncmass50", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700)

        gramass50.Fit(fitfuncmass50,"R,S")
        cmass50 = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)

        gramass50.Draw()

        pmass50 = fitfuncmass50.GetParameters()
        tagger.signalmass50["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.signalmass50.fjet_pt)
        tagger.bgmass50["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.bgmass50.fjet_pt)

        aa= np.array([200,500,1000,2000,3000])
        vfunc = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))
        print("tag_score[:100]", vfunc(aa) )

        #tagger.signalmass50 = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
        #tagger.bgmass50 = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

        tagger.bgmass50_tagged = tagger.bgmass50[tagger.bgmass50.fjet_nnscore > tagger.bgmass50.tag_cut_mass]
        #tagger.bgmass50_untagged = tagger.bgmass50[(tagger.bgmass50.fjet_nnscore < tagger.bgmass50.tag_cut_mass) & (tagger.bgmass50.fjet_nnscore>=0)]
        tagger.bgmass50_untagged = tagger.bg_untagged
        tagger.signalmass50_tagged = tagger.signalmass50[tagger.signalmass50.fjet_nnscore > tagger.signalmass50.tag_cut_mass]

        print(len(tagger.bgmass50_tagged))
        print(len(tagger.signalmass50_tagged))
        print(len(tagger.bg))


        ############################################### 80% case ######################################################

    if wp ==0.8:

        for pt,nn,weight in zip(tagger.signalmass80["fjet_pt"], tagger.signalmass80["fjet_nnscore"],tagger.signalmass80["fjet_weight_pt"]):
            h_pt_nn_mass50.Fill(pt,nn,weight)

        ptsmass50, scoresmass50 = get_eff_score_mass(h_pt_nn_mass50, h_pt_nn, wp)
        print("my scoresmass50->", scoresmass50, "len(scoresmass50)->",len(scoresmass50) )

        scoresmass50 = scoresmass50[6:]
        ptsmass50 = ptsmass50 [6:]
        gramass50 = TGraph(len(ptsmass50), np.array(ptsmass50).astype("float"), np.array(scoresmass50).astype("float"))
        fitfuncmass50 = ROOT.TF1("fitfuncmass50", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700)

        gramass50.Fit(fitfuncmass50,"R,S")
        cmass50 = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)

        gramass50.Draw()

        pmass50 = fitfuncmass50.GetParameters()
        tagger.signalmass80["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.signalmass80.fjet_pt)
        tagger.bgmass80["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.bgmass80.fjet_pt)

        aa= np.array([200,500,1000,2000,3000])
        vfunc = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))
        print("tag_score[:100]", vfunc(aa) )

        tagger.bgmass80_tagged = tagger.bgmass80[tagger.bgmass80.fjet_nnscore > tagger.bgmass80.tag_cut_mass]

        tagger.bgmass80_untagged = tagger.bg_untagged
        tagger.signalmass80_tagged = tagger.signalmass80[tagger.signalmass80.fjet_nnscore > tagger.signalmass80.tag_cut_mass]

        # print(len(tagger.bgmass50_tagged))
        # print(len(tagger.signalmass80_tagged))
        # print(len(tagger.bg))


def get_wp_th1(tagger,wp, prefix=''):
    #print("I am here in get_wp_th1 ")
    print (tagger.name)
    h_pt_nn     = TH2D( "h_pt_nn{}".format(tagger.name), "h_pt_nn{}".format(tagger.name), 100, 0., 3000,100,0,1 )
    h_pt_nn.SetDirectory(0)
    for pt,nn in zip(tagger.signal["fjet_pt"].values,tagger.signal["fjet_nnscore"].values):
        h_pt_nn.Fill(pt,nn)

    c = TCanvas("", "", 500, 500)
    c.SetRightMargin(0.2)
    h_pt_nn.Draw("colz")
    c.SaveAs("{}/pt_vs_score.png".format(prefix))

    pts, scores = get_eff_score(h_pt_nn,wp)
    # #print(len(pts))
    # #print(len(scores))
    # #print(pts)
    # #print(scores)

    h_pt_nn_h   = TH1D("h_pt_nn_histo{}".format(tagger.name), "h_pt_nn_histo{}".format(tagger.name), len(pts), 0., 3000)
    ##print(pts)
    ##print(scores)

    a2h(scores,h_pt_nn_h)

    def score_cut(pt):
        return h_pt_nn_h.GetBinContent(h_pt_nn_h.FindBin(pt))
    tagger.scores["tag_cut"] = np.vectorize(score_cut)(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]

    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[tagger.bg.fjet_nnscore < tagger.bg.tag_cut]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]


def get_flat_weight(pt,dsid):

    inFile = TFile.Open( "flat_weights.root" ," READ ")
    flat_bg = inFile.Get("bg_inv")
    flat_sig = inFile.Get("h_sig_inv")

    if dsid > 370000:
        return flat_sig.GetBinContent(flat_sig.FindBin(pt))
    else:
        return flat_bg.GetBinContent(flat_bg.FindBin(pt))

class tagger_scores():
    def __init__(self, name, score_file,working_point):
        intreename = "FlatSubstructureJetTree"
        self.name = name
        self.score_file = score_file

        f = TFile.Open(score_file, 'READ')
        tree = f.Get(intreename)

        #self.events = uproot.open(score_file+":"+intreename)
        #self.scores = self.events.arrays( library="pd")
        branches = []
        mycopy = tree.GetListOfBranches()
        for i in mycopy:
            branches.append(i.GetName())
        #print(str(branches))
        arr = tree2array(tree, branches = branches, include_weight = False)
        self.scores = pd.DataFrame(arr)

        # #print(self.scores.head())
        # #print(self.scores["EventInfo_mcChannelNumber"].values)

        self.scores["no_weight"]   = np.ones_like(self.scores.fjet_pt.values)
        # #print("I am here in utils")
        try:
            self.scores["chris_weight"] = (self.scores["fjet_weight_pt_dR"].values)
        except:
            self.scores["chris_weight"] = (self.scores["fjet_weight_pt"].values)

        self.scores      = self.scores[  (self.scores["chris_weight"] < 150) | (self.scores.EventInfo_mcChannelNumber<370000) ]
        
        ## Make the pt spectrum smooth (Chris Delitzsch advice)
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364702][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364703][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364704][self.scores.fjet_pt > 2000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]

        self.signal        = self.scores[self.scores.EventInfo_mcChannelNumber>370000]
        # self.signal =  self.signal[ self.signal["ungroomedtruthjet_m"]>50000 ]
        # self.signal  =self.signal[ self.signal["EventInfo_NBHadrons"] == 0 ]
        # self.signal =  self.signal[ self.signal["ungroomedtruthjet_split12"]/1000 > 55.25*np.exp( (-2.34/1000.) * (self.signal["ungroomedtruthjet_pt"]/1000) )  ]
        self.signal_tagged = self.signal[self.signal.fjet_nnscore > working_point]

        self.bg          = self.scores[self.scores.EventInfo_mcChannelNumber<370000]
        self.bg_tagged   = self.bg[self.bg.fjet_nnscore > working_point]
        self.bg_untagged = self.bg[self.bg.fjet_nnscore < working_point]


        self.signal_pt_300_650      = self.signal[  (self.signal["fjet_pt"] < 650) & ((self.signal["fjet_pt"] > 300))]
        self.bg_pt_300_650          = self.bg[ (self.bg["fjet_pt"] < 650) & ((self.bg["fjet_pt"] > 300))]

        self.signal_pt_650_1000      = self.signal[  (self.signal["fjet_pt"] < 1000) & ((self.signal["fjet_pt"] > 650))]
        self.bg_pt_650_1000          = self.bg[ (self.bg["fjet_pt"] < 1000) & ((self.bg["fjet_pt"] > 650))]

        self.signal_pt_1000_2000      = self.signal[  (self.signal["fjet_pt"] < 2000) & ((self.signal["fjet_pt"] > 1000))]
        self.bg_pt_1000_2000          = self.bg[ (self.bg["fjet_pt"] < 2000) & ((self.bg["fjet_pt"] > 1000))]

        self.signal_pt_2000_3000      = self.signal[  (self.signal["fjet_pt"] < 3000) & ((self.signal["fjet_pt"] > 2000))]
        self.bg_pt_2000_3000          = self.bg[ (self.bg["fjet_pt"] < 3000) & ((self.bg["fjet_pt"] > 2000))]


        # self.signalmass50      = self.signal[ (self.signal["fjet_m"] < MassCutHigh50(self.signal["truthjet_pt"])) & (self.signal["fjet_m"] > MassCutLow50(self.signal["truthjet_pt"]))]
        # self.bgmass50          = self.bg[ (self.bg["fjet_m"] < MassCutHigh50(self.bg["truthjet_pt"])) & (self.bg["fjet_m"] > MassCutLow50(self.bg["truthjet_pt"]))]

        # mask = lambda data: (data['fjet_m'] > ((69.8199410326) + (0.0150081803314)*pow(data['truthjet_pt'],1) + (-6.17224460098e-05)*pow(data['truthjet_pt'],2) + (7.86154778779e-08)*pow(data['truthjet_pt'],3) + (-4.97257435999e-11)*pow(data['truthjet_pt'],4) + (1.50129847267e-14)*pow(data['truthjet_pt'],5) + (-1.72903388714e-18)*pow(data['truthjet_pt'],6)) ) & (data['fjet_m'] < ((165.048442698) + (-0.364378464009)*pow(data['truthjet_pt'],1) + (0.000723923415283)*pow(data['truthjet_pt'],2) + (-7.12212365612e-07)*pow(data['truthjet_pt'],3) + (3.66502330413e-10)*pow(data['truthjet_pt'],4) + (-9.36551583203e-14)*pow(data['truthjet_pt'],5) + (9.37467300706e-18)*pow(data['truthjet_pt'],6)) )
        # self.signalmass50      = self.signal[ (self.signal["fjet_m"] < MassCutHigh50(self.signal["fjet_pt"])) & (self.signal["fjet_m"] > MassCutLow50(self.signal["fjet_pt"]))]
        # self.bgmass50          = self.bg[ (self.bg["fjet_m"] < MassCutHigh50(self.bg["fjet_pt"])) & (self.bg["fjet_m"] > MassCutLow50(self.bg["fjet_pt"]))]
        # self.signalmass50      = self.signal[(self.signal["fjet_pt"] < 1000) & ((self.signal["fjet_pt"] > 200))]
        # self.bgmass50          = self.bg[(self.bg["fjet_pt"] < 1000) & ((self.bg["fjet_pt"] > 200))]


        self.signalmass50      = self.signal[ (self.signal["fjet_m"] < MassCutHigh50(self.signal["fjet_pt"])) & (self.signal["fjet_m"] > MassCutLow50(self.signal["fjet_pt"]))]
        self.bgmass50          = self.bg[ (self.bg["fjet_m"] < MassCutHigh50(self.bg["fjet_pt"])) & (self.bg["fjet_m"] > MassCutLow50(self.bg["fjet_pt"]))]


        # self.signalmass50      = self.signalmass50[(self.signalmass50["fjet_pt"] < 3000) & ((self.signalmass50["fjet_pt"] > 200))]
        # self.bgmass50          = self.bgmass50[(self.bgmass50["fjet_pt"] < 3000) & ((self.bgmass50["fjet_pt"] > 200))]

        # self.signalmass50      = self.signal[ (self.signal["fjet_m"] < MassCutHigh80(self.signal["fjet_pt"])) & (self.signal["fjet_m"] > MassCutLow80(self.signal["fjet_pt"]))]
        # self.bgmass50          = self.bg[ (self.bg["fjet_m"] < MassCutHigh80(self.bg["fjet_pt"])) & (self.bg["fjet_m"] > MassCutLow80(self.bg["fjet_pt"]))]

        ## Rafael fix
        self.signalmass80      = self.signal[ (self.signal["fjet_m"] < MassCutHigh80(self.signal["fjet_pt"])) & (self.signal["fjet_m"] > MassCutLow80(self.signal["fjet_pt"]))]
        self.bgmass80          = self.bg[ (self.bg["fjet_m"] < MassCutHigh80(self.bg["fjet_pt"])) & (self.bg["fjet_m"] > MassCutLow80(self.bg["fjet_pt"]))]


        # self.signalmass50      = self.signalmass50[ (self.signalmass50["fjet_pt"] < 3000) & ((self.signalmass50["fjet_pt"] > 200))]
        # self.bgmass50          = self.bgmass50[ (self.bgmass50["fjet_pt"] < 3000) & ((self.bgmass50["fjet_pt"] > 200))]

        # self.signalmass50      = self.signalmass50[ (self.signalmass50["fjet_m"] > 40)]
        # self.bgmass50          = self.bgmass50[ (self.bgmass50["fjet_m"] > 40) ]


        self.signalmass50_pt_300_650      = self.signalmass50[  (self.signalmass50["fjet_pt"] < 650) & ((self.signalmass50["fjet_pt"] > 300))]
        self.bgmass50_pt_300_650          = self.bgmass50[ (self.bgmass50["fjet_pt"] < 650) & ((self.bgmass50["fjet_pt"] > 300))]

        self.signalmass50_pt_650_1000      = self.signalmass50[  (self.signalmass50["fjet_pt"] < 1000) & ((self.signalmass50["fjet_pt"] > 650))]
        self.bgmass50_pt_650_1000          = self.bgmass50[ (self.bgmass50["fjet_pt"] < 1000) & ((self.bgmass50["fjet_pt"] > 650))]

        self.signalmass50_pt_1000_2000      = self.signalmass50[  (self.signalmass50["fjet_pt"] < 2000) & ((self.signalmass50["fjet_pt"] > 1000))]
        self.bgmass50_pt_1000_2000          = self.bgmass50[ (self.bgmass50["fjet_pt"] < 2000) & ((self.bgmass50["fjet_pt"] > 1000))]

        self.signalmass50_pt_2000_3000      = self.signalmass50[  (self.signalmass50["fjet_pt"] < 3000) & ((self.signalmass50["fjet_pt"] > 2000))]
        self.bgmass50_pt_2000_3000          = self.bgmass50[ (self.bgmass50["fjet_pt"] < 3000) & ((self.bgmass50["fjet_pt"] > 2000))]


        # print(self.signalmass50)
        self.signal = self.signal.dropna()
        self.bg = self.bg.dropna()
        self.signal_300_650 = self.signal_pt_300_650.dropna()
        self.bg_300_650 = self.bg_pt_300_650.dropna()
        self.signal_650_1000 = self.signal_pt_650_1000.dropna()
        self.bg_650_1000 = self.bg_pt_650_1000.dropna()
        self.signal_1000_2000 = self.signal_pt_1000_2000.dropna()
        self.bg_1000_2000 = self.bg_pt_1000_2000.dropna()
        self.signal_2000_3000 = self.signal_pt_2000_3000.dropna()
        self.bg_2000_3000 = self.bg_pt_2000_3000.dropna()
        self.signalmass50 = self.signalmass50.dropna()
        self.bgmass50 = self.bgmass50.dropna()
        self.signalmass50_pt_300_650 = self.signalmass50_pt_300_650.dropna()
        self.bgmass50_pt_300_650    = self.bgmass50_pt_300_650.dropna()
        self.signalmass50_pt_650_1000   = self.signalmass50_pt_650_1000.dropna()
        self.bgmass50_pt_650_1000      = self.bgmass50_pt_650_1000.dropna()
        self.signalmass50_pt_1000_2000  = self.signalmass50_pt_1000_2000.dropna()
        self.bgmass50_pt_1000_2000      = self.bgmass50_pt_1000_2000.dropna()
        self.signalmass50_pt_2000_3000  = self.signalmass50_pt_2000_3000.dropna()
        self.bgmass50_pt_2000_3000      = self.bgmass50_pt_2000_3000.dropna()

        self.signal_taggedmass50   = self.signalmass50[self.signalmass50.fjet_nnscore > working_point]
        self.bg_taggedmass50   = self.bgmass50[self.bgmass50.fjet_nnscore > working_point]
        self.signal_taggedmass50 = self.signal_taggedmass50.dropna()
        self.bg_taggedmass50 = self.bg_taggedmass50.dropna()

        print (self.name)

        #print ("signal ratio:",len(self.signal_tagged.values)/len(self.signal.values))
        #print ("bg ratio:",        len(self.bg_tagged.values)/len(self.bg.values))

        # #print(self.signal["fjet_nnscore"].values)

        self.h_signal = TH1D( "signal{}".format(self.name), "signal{}".format(self.name), 1200, 0, 1)
        self.h_bg     = TH1D(     "bg{}".format(self.name),     "bg{}".format(self.name), 1200, 0, 1)
        self.h_signal_300_650 = TH1D( "signal_300_650{}".format(self.name), "signal_300_650{}".format(self.name), 200, 0, 1)
        self.h_bg_300_650     = TH1D(     "bg_300_650{}".format(self.name),     "bg_300_650{}".format(self.name), 200, 0, 1)
        self.h_signal_650_1000 = TH1D( "signal_650_1000{}".format(self.name), "signal_650_1000{}".format(self.name), 200, 0, 1)
        self.h_bg_650_1000     = TH1D(     "bg_650_1000{}".format(self.name),     "bg_650_1000{}".format(self.name), 200, 0, 1)
        self.h_signal_1000_2000 = TH1D( "signal_1000_2000{}".format(self.name), "signal_1000_2000{}".format(self.name), 200, 0, 1)
        self.h_bg_1000_2000     = TH1D(     "bg_1000_2000{}".format(self.name),     "bg_1000_2000{}".format(self.name), 200, 0, 1)
        self.h_signal_2000_3000 = TH1D( "signal_2000_3000{}".format(self.name), "signal_2000_3000{}".format(self.name), 200, 0, 1)
        self.h_bg_2000_3000     = TH1D(     "bg_2000_3000{}".format(self.name),     "bg_2000_3000{}".format(self.name), 200, 0, 1)

        self.h_signalmass50 = TH1D( "signalmass50{}".format(self.name), "signalmass50{}".format(self.name), 200, 0, 1)
        self.h_bgmass50     = TH1D(     "bgmass50{}".format(self.name),     "bgmass50{}".format(self.name), 200, 0, 1)
        self.h_signalmass50_pt_300_650 = TH1D( "signalmass50_pt_300_650{}".format(self.name), "signalmass50_pt_300_650{}".format(self.name), 200, 0, 1)
        self.h_bgmass50_pt_300_650     = TH1D(     "bgmass50_pt_300_650{}".format(self.name),     "bgmass50_pt_300_650{}".format(self.name), 200, 0, 1)
        self.h_signalmass50_pt_650_1000 = TH1D( "signalmass50_pt_650_1000{}".format(self.name), "signalmass50_pt_650_1000{}".format(self.name), 200, 0, 1)
        self.h_bgmass50_pt_650_1000     = TH1D(     "bgmass50_pt_650_1000{}".format(self.name),     "bgmass50_pt_650_1000{}".format(self.name), 200, 0, 1)
        self.h_signalmass50_pt_1000_2000 = TH1D( "signalmass50_pt_1000_2000{}".format(self.name), "signalmass50_pt_1000_2000{}".format(self.name), 200, 0, 1)
        self.h_bgmass50_pt_1000_2000     = TH1D(     "bgmass50_pt_1000_2000{}".format(self.name),     "bgmass50_pt_1000_2000{}".format(self.name), 200, 0, 1)
        self.h_signalmass50_pt_2000_3000 = TH1D( "signalmass50_pt_2000_3000{}".format(self.name), "signalmass50_pt_2000_3000{}".format(self.name), 200, 0, 1)
        self.h_bgmass50_pt_2000_3000     = TH1D(     "bgmass50_pt_2000_3000{}".format(self.name),     "bgmass50_pt_2000_3000{}".format(self.name), 200, 0, 1)

        self.h_signal.SetDirectory(0)
        self.h_bg.SetDirectory(0)
        self.h_signal_300_650.SetDirectory(0)
        self.h_bg_300_650.SetDirectory(0)
        self.h_signal_650_1000.SetDirectory(0)
        self.h_bg_650_1000.SetDirectory(0)
        self.h_signal_1000_2000.SetDirectory(0)
        self.h_bg_1000_2000.SetDirectory(0)
        self.h_signal_2000_3000.SetDirectory(0)
        self.h_bg_2000_3000.SetDirectory(0)

        self.h_signalmass50.SetDirectory(0)
        self.h_bgmass50.SetDirectory(0)
        self.h_signalmass50_pt_300_650.SetDirectory(0)
        self.h_bgmass50_pt_300_650.SetDirectory(0)
        self.h_signalmass50_pt_650_1000.SetDirectory(0)
        self.h_bgmass50_pt_650_1000.SetDirectory(0)
        self.h_signalmass50_pt_1000_2000.SetDirectory(0)
        self.h_bgmass50_pt_1000_2000.SetDirectory(0)
        self.h_signalmass50_pt_2000_3000.SetDirectory(0)
        self.h_bgmass50_pt_2000_3000.SetDirectory(0)


        fh(self.h_signal, self.signal["fjet_nnscore"].values, self.signal["fjet_weight_pt"].values)
        fh(self.h_bg,         self.bg["fjet_nnscore"].values,     self.bg["fjet_weight_pt"].values)
        #fh(self.h_signal, self.signal["fjet_nnscore"].values, self.signal["EventInfo_mcEventWeight"].values)
        #fh(self.h_bg,         self.bg["fjet_nnscore"].values,     self.bg["EventInfo_mcEventWeight"].values)

        fh(self.h_signal_300_650, self.signal_300_650["fjet_nnscore"].values, self.signal_300_650["fjet_weight_pt"].values)
        fh(self.h_bg_300_650,         self.bg_300_650["fjet_nnscore"].values,     self.bg_300_650["fjet_weight_pt"].values)
        fh(self.h_signal_650_1000, self.signal_650_1000["fjet_nnscore"].values, self.signal_650_1000["fjet_weight_pt"].values)
        fh(self.h_bg_650_1000,         self.bg_650_1000["fjet_nnscore"].values,     self.bg_650_1000["fjet_weight_pt"].values)
        fh(self.h_signal_1000_2000, self.signal_1000_2000["fjet_nnscore"].values, self.signal_1000_2000["fjet_weight_pt"].values)
        fh(self.h_bg_1000_2000,         self.bg_1000_2000["fjet_nnscore"].values,     self.bg_1000_2000["fjet_weight_pt"].values)
        fh(self.h_signal_2000_3000, self.signal_2000_3000["fjet_nnscore"].values, self.signal_2000_3000["fjet_weight_pt"].values)
        fh(self.h_bg_2000_3000,         self.bg_2000_3000["fjet_nnscore"].values,     self.bg_2000_3000["fjet_weight_pt"].values)

        fh(self.h_signalmass50, self.signalmass50["fjet_nnscore"].values, self.signalmass50["fjet_weight_pt"].values)
        fh(self.h_bgmass50,         self.bgmass50["fjet_nnscore"].values,     self.bgmass50["fjet_weight_pt"].values)


        fh(self.h_signalmass50_pt_300_650, self.signalmass50_pt_300_650["fjet_nnscore"].values, self.signalmass50_pt_300_650["fjet_weight_pt"].values)
        fh(self.h_bgmass50_pt_300_650,         self.bgmass50_pt_300_650["fjet_nnscore"].values,     self.bgmass50_pt_300_650["fjet_weight_pt"].values)

        fh(self.h_signalmass50_pt_650_1000, self.signalmass50_pt_650_1000["fjet_nnscore"].values, self.signalmass50_pt_650_1000["fjet_weight_pt"].values)
        fh(self.h_bgmass50_pt_650_1000,         self.bgmass50_pt_650_1000["fjet_nnscore"].values,     self.bgmass50_pt_650_1000["fjet_weight_pt"].values)

        fh(self.h_signalmass50_pt_1000_2000, self.signalmass50_pt_1000_2000["fjet_nnscore"].values, self.signalmass50_pt_1000_2000["fjet_weight_pt"].values)
        fh(self.h_bgmass50_pt_1000_2000,         self.bgmass50_pt_1000_2000["fjet_nnscore"].values,     self.bgmass50_pt_1000_2000["fjet_weight_pt"].values)

        fh(self.h_signalmass50_pt_2000_3000, self.signalmass50_pt_2000_3000["fjet_nnscore"].values, self.signalmass50_pt_2000_3000["fjet_weight_pt"].values)
        fh(self.h_bgmass50_pt_2000_3000,         self.bgmass50_pt_2000_3000["fjet_nnscore"].values,     self.bgmass50_pt_2000_3000["fjet_weight_pt"].values)


        # ##### tests without weights  _NOWEIGHTS  #######################################################################################################
        #
        # self.h_signal_NOWEIGHTS = TH1D( "signal{}".format(self.name), "signal{}".format(self.name), 500, 0, 1)
        # self.h_bg_NOWEIGHTS     = TH1D(     "bg{}".format(self.name),     "bg{}".format(self.name), 500, 0, 1)
        # self.h_signal_300_650_NOWEIGHTS = TH1D( "signal_300_650{}".format(self.name), "signal_300_650{}".format(self.name), 500, 0, 1)
        # self.h_bg_300_650_NOWEIGHTS     = TH1D(     "bg_300_650{}".format(self.name),     "bg_300_650{}".format(self.name), 500, 0, 1)
        # self.h_signal_650_1000_NOWEIGHTS = TH1D( "signal_650_1000{}".format(self.name), "signal_650_1000{}".format(self.name), 500, 0, 1)
        # self.h_bg_650_1000_NOWEIGHTS     = TH1D(     "bg_650_1000{}".format(self.name),     "bg_650_1000{}".format(self.name), 500, 0, 1)
        # self.h_signal_1000_2000_NOWEIGHTS = TH1D( "signal_1000_2000{}".format(self.name), "signal_1000_2000{}".format(self.name), 500, 0, 1)
        # self.h_bg_1000_2000_NOWEIGHTS     = TH1D(     "bg_1000_2000{}".format(self.name),     "bg_1000_2000{}".format(self.name), 500, 0, 1)
        # self.h_signal_2000_3000_NOWEIGHTS = TH1D( "signal_2000_3000{}".format(self.name), "signal_2000_3000{}".format(self.name), 500, 0, 1)
        # self.h_bg_2000_3000_NOWEIGHTS     = TH1D(     "bg_2000_3000{}".format(self.name),     "bg_2000_3000{}".format(self.name), 500, 0, 1)
        #
        # self.h_signalmass50_NOWEIGHTS = TH1D( "signalmass50{}".format(self.name), "signalmass50{}".format(self.name), 500, 0, 1)
        # self.h_bgmass50_NOWEIGHTS     = TH1D(     "bgmass50{}".format(self.name),     "bgmass50{}".format(self.name), 500, 0, 1)
        # self.h_signalmass50_pt_300_650_NOWEIGHTS = TH1D( "signalmass50_pt_300_650{}".format(self.name), "signalmass50_pt_300_650{}".format(self.name), 500, 0, 1)
        # self.h_bgmass50_pt_300_650_NOWEIGHTS     = TH1D(     "bgmass50_pt_300_650{}".format(self.name),     "bgmass50_pt_300_650{}".format(self.name), 500, 0, 1)
        # self.h_signalmass50_pt_650_1000_NOWEIGHTS = TH1D( "signalmass50_pt_650_1000{}".format(self.name), "signalmass50_pt_650_1000{}".format(self.name), 500, 0, 1)
        # self.h_bgmass50_pt_650_1000_NOWEIGHTS     = TH1D(     "bgmass50_pt_650_1000{}".format(self.name),     "bgmass50_pt_650_1000{}".format(self.name), 500, 0, 1)
        # self.h_signalmass50_pt_1000_2000_NOWEIGHTS = TH1D( "signalmass50_pt_1000_2000{}".format(self.name), "signalmass50_pt_1000_2000{}".format(self.name), 500, 0, 1)
        # self.h_bgmass50_pt_1000_2000_NOWEIGHTS     = TH1D(     "bgmass50_pt_1000_2000{}".format(self.name),     "bgmass50_pt_1000_2000{}".format(self.name), 500, 0, 1)
        # self.h_signalmass50_pt_2000_3000_NOWEIGHTS = TH1D( "signalmass50_pt_2000_3000{}".format(self.name), "signalmass50_pt_2000_3000{}".format(self.name), 500, 0, 1)
        # self.h_bgmass50_pt_2000_3000_NOWEIGHTS     = TH1D(     "bgmass50_pt_2000_3000{}".format(self.name),     "bgmass50_pt_2000_3000{}".format(self.name), 500, 0, 1)
        #
        #
        # self.h_signal_NOWEIGHTS.SetDirectory(0)
        # self.h_bg_NOWEIGHTS.SetDirectory(0)
        # self.h_signal_300_650_NOWEIGHTS.SetDirectory(0)
        # self.h_bg_300_650_NOWEIGHTS.SetDirectory(0)
        # self.h_signal_650_1000_NOWEIGHTS.SetDirectory(0)
        # self.h_bg_650_1000_NOWEIGHTS.SetDirectory(0)
        # self.h_signal_1000_2000_NOWEIGHTS.SetDirectory(0)
        # self.h_bg_1000_2000_NOWEIGHTS.SetDirectory(0)
        # self.h_signal_2000_3000_NOWEIGHTS.SetDirectory(0)
        # self.h_bg_2000_3000_NOWEIGHTS.SetDirectory(0)
        #
        # self.h_signalmass50_NOWEIGHTS.SetDirectory(0)
        # self.h_bgmass50_NOWEIGHTS.SetDirectory(0)
        # self.h_signalmass50_pt_300_650_NOWEIGHTS.SetDirectory(0)
        # self.h_bgmass50_pt_300_650_NOWEIGHTS.SetDirectory(0)
        # self.h_signalmass50_pt_650_1000_NOWEIGHTS.SetDirectory(0)
        # self.h_bgmass50_pt_650_1000_NOWEIGHTS.SetDirectory(0)
        # self.h_signalmass50_pt_1000_2000_NOWEIGHTS.SetDirectory(0)
        # self.h_bgmass50_pt_1000_2000_NOWEIGHTS.SetDirectory(0)
        # self.h_signalmass50_pt_2000_3000_NOWEIGHTS.SetDirectory(0)
        # self.h_bgmass50_pt_2000_3000_NOWEIGHTS.SetDirectory(0)
        #
        # #  _NOWEIGHTS
        # fh(self.h_signal_NOWEIGHTS, self.signal["fjet_nnscore"].values)
        # fh(self.h_bg_NOWEIGHTS,         self.bg["fjet_nnscore"].values)
        # fh(self.h_signal_300_650_NOWEIGHTS, self.signal_300_650["fjet_nnscore"].values)
        # fh(self.h_bg_300_650_NOWEIGHTS,         self.bg_300_650["fjet_nnscore"].values)
        # fh(self.h_signal_650_1000_NOWEIGHTS, self.signal_650_1000["fjet_nnscore"].values)
        # fh(self.h_bg_650_1000_NOWEIGHTS,         self.bg_650_1000["fjet_nnscore"].values)
        # fh(self.h_signal_1000_2000_NOWEIGHTS, self.signal_1000_2000["fjet_nnscore"].values)
        # fh(self.h_bg_1000_2000_NOWEIGHTS,         self.bg_1000_2000["fjet_nnscore"].values)
        # fh(self.h_signal_2000_3000_NOWEIGHTS, self.signal_2000_3000["fjet_nnscore"].values)
        # fh(self.h_bg_2000_3000_NOWEIGHTS,         self.bg_2000_3000["fjet_nnscore"].values)
        #
        # fh(self.h_signalmass50_NOWEIGHTS, self.signalmass50["fjet_nnscore"].values)
        # fh(self.h_bgmass50_NOWEIGHTS,         self.bgmass50["fjet_nnscore"].values)
        #
        # fh(self.h_signalmass50_pt_300_650_NOWEIGHTS, self.signalmass50_pt_300_650["fjet_nnscore"].values)
        # fh(self.h_bgmass50_pt_300_650_NOWEIGHTS,         self.bgmass50_pt_300_650["fjet_nnscore"].values)
        #
        # fh(self.h_signalmass50_pt_650_1000_NOWEIGHTS, self.signalmass50_pt_650_1000["fjet_nnscore"].values)
        # fh(self.h_bgmass50_pt_650_1000_NOWEIGHTS,         self.bgmass50_pt_650_1000["fjet_nnscore"].values)
        #
        # fh(self.h_signalmass50_pt_1000_2000_NOWEIGHTS, self.signalmass50_pt_1000_2000["fjet_nnscore"].values)
        # fh(self.h_bgmass50_pt_1000_2000_NOWEIGHTS,         self.bgmass50_pt_1000_2000["fjet_nnscore"].values)
        #
        # fh(self.h_signalmass50_pt_2000_3000_NOWEIGHTS, self.signalmass50_pt_2000_3000["fjet_nnscore"].values)
        # fh(self.h_bgmass50_pt_2000_3000_NOWEIGHTS,         self.bgmass50_pt_2000_3000["fjet_nnscore"].values)



    workingpoint = 0.5
    def get_roc(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal, self.h_bg,self.h_signal, self.h_bg, 0.5, massCut=False)
        return tprs, fprs, auc

    def get_roc_300_650(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_300_650, self.h_bg_300_650,self.h_signal_300_650, self.h_bg_300_650, 0.5, massCut=False)
        return tprs, fprs, auc

    def get_roc_650_1000(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_650_1000, self.h_bg_650_1000,self.h_signal_650_1000, self.h_bg_650_1000, 0.5, massCut=False)
        return tprs, fprs, auc

    def get_roc_1000_2000(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_1000_2000, self.h_bg_1000_2000,self.h_signal_1000_2000, self.h_bg_1000_2000, 0.5, massCut=False)
        return tprs, fprs, auc

    def get_roc_2000_3000(self):
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_2000_3000, self.h_bg_2000_3000,self.h_signal_2000_3000, self.h_bg_2000_3000, 0.5, massCut=False)
        return tprs, fprs, auc

    # def get_roc_mass(self):
    #     tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal, self.h_bg, self.h_signalmass50, self.h_bgmass50, wpcut=0.5, massCut=True)
    #     return tprs, fprs, auc
    #
    # def get_roc_mass_300_650(self):
    #     tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_300_650, self.h_bg_300_650,self.h_signalmass50_pt_300_650, self.h_bgmass50_pt_300_650, 0.5, massCut=True)
    #     return tprs, fprs, auc
    #
    # def get_roc_mass_650_1000(self):
    #     tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_650_1000, self.h_bg_650_1000,self.h_signalmass50_pt_650_1000, self.h_bgmass50_pt_650_1000, 0.5, massCut=True)
    #     return tprs, fprs, auc
    #
    # def get_roc_mass_1000_2000(self):
    #     tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_1000_2000, self.h_bg_1000_2000,self.h_signalmass50_pt_1000_2000, self.h_bgmass50_pt_1000_2000, 0.5, massCut=True)
    #     return tprs, fprs, auc
    #
    # def get_roc_mass_2000_3000(self):
    #     tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_2000_3000, self.h_bg_2000_3000,self.h_signalmass50_pt_2000_3000, self.h_bgmass50_pt_2000_3000, 0.5, massCut=True)
    #     return tprs, fprs, auc
    def get_roc_mass(self):
        #tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal, self.h_bg,self.h_signalmass50, self.h_bgmass50, 0.5, massCut=True)
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos_DAVIDE(self.h_signal, self.h_bg,self.h_signalmass50, self.h_bgmass50, self.h_signal_NOWEIGHTS, self.h_bg_NOWEIGHTS,self.h_signalmass50_NOWEIGHTS, self.h_bgmass50_NOWEIGHTS, 0.5, massCut=True)
        return tprs, fprs, auc

    def get_roc_mass_300_650(self):
        #tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_300_650, self.h_bg_300_650,self.h_signalmass50_pt_300_650, self.h_bgmass50_pt_300_650, 0.5, massCut=True)
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos_DAVIDE(self.h_signal_300_650, self.h_bg_300_650,self.h_signalmass50_pt_300_650, self.h_bgmass50_pt_300_650, self.h_signal_300_650_NOWEIGHTS, self.h_bg_300_650_NOWEIGHTS,self.h_signalmass50_pt_300_650_NOWEIGHTS, self.h_bgmass50_pt_300_650_NOWEIGHTS, 0.5, massCut=True)

        return tprs, fprs, auc

    def get_roc_mass_650_1000(self):
        #tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_650_1000, self.h_bg_650_1000,self.h_signalmass50_pt_650_1000, self.h_bgmass50_pt_650_1000, 0.5, massCut=True)
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos_DAVIDE(self.h_signal_650_1000, self.h_bg_650_1000,self.h_signalmass50_pt_650_1000, self.h_bgmass50_pt_650_1000,  self.h_signal_650_1000_NOWEIGHTS, self.h_bg_650_1000_NOWEIGHTS,self.h_signalmass50_pt_650_1000_NOWEIGHTS, self.h_bgmass50_pt_650_1000_NOWEIGHTS, 0.5, massCut=True)

        return tprs, fprs, auc

    def get_roc_mass_1000_2000(self):
        #tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_1000_2000, self.h_bg_1000_2000,self.h_signalmass50_pt_1000_2000, self.h_bgmass50_pt_1000_2000, 0.5, massCut=True)
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos_DAVIDE(self.h_signal_1000_2000, self.h_bg_1000_2000,self.h_signalmass50_pt_1000_2000, self.h_bgmass50_pt_1000_2000,  self.h_signal_1000_2000_NOWEIGHTS, self.h_bg_1000_2000_NOWEIGHTS,self.h_signalmass50_pt_1000_2000_NOWEIGHTS, self.h_bgmass50_pt_1000_2000_NOWEIGHTS, 0.5, massCut=True)
        return tprs, fprs, auc

    def get_roc_mass_2000_3000(self):
        #tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos(self.h_signal_2000_3000, self.h_bg_2000_3000,self.h_signalmass50_pt_2000_3000, self.h_bgmass50_pt_2000_3000, 0.5, massCut=True)
        tprs, fprs, auc, tpr_wp, fpr_wp = roc_from_histos_DAVIDE(self.h_signal_2000_3000, self.h_bg_2000_3000,self.h_signalmass50_pt_2000_3000, self.h_bgmass50_pt_2000_3000,  self.h_signal_2000_3000_NOWEIGHTS, self.h_bg_2000_3000_NOWEIGHTS,self.h_signalmass50_pt_2000_3000_NOWEIGHTS, self.h_bgmass50_pt_2000_3000_NOWEIGHTS, 0.5, massCut=True)
        return tprs, fprs, auc


def getANNBkgRejpT(axis, NN):

    xDNN, yDNN, yDNN_err, xANN, yANN, yANN_err = 'bkrejDNN_x.npy', 'bkrejDNN_y.npy', 'bkrejDNN_yerr.npy', 'bkrejANN_x.npy', 'bkrejANN_y.npy','bkrejANN_yerr.npy'
    zNN_x = np.load(xDNN)
    zNN_y = np.load(yDNN)
    zNN_yerr = np.load(yDNN_err)
    zANN_lambda10_x = np.load(xANN)
    zANN_lambda10_y = np.load(yANN)
    zANN_lambda10_yerr = np.load(yANN_err)

    if(axis=="x" and NN=="NN"): return zNN_x
    if(axis=="y" and NN=="NN"): return zNN_y
    if(axis=="yerr" and NN=="NN"): return zNN_yerr
    if(axis=="x" and NN=="ANN"): return zANN_lambda10_x
    if(axis=="y" and NN=="ANN"): return zANN_lambda10_y
    if(axis=="yerr" and NN=="ANN"): return zANN_lambda10_yerr


def getANNROCresults(axis, NN, pTrange=''):
    if pTrange=='':
        xDNN, yDNN, xANN, yANN = 'xDNN.npy', 'yDNN.npy', 'xANN.npy', 'yANN.npy'
    if pTrange=='300_650':
        xDNN, yDNN, xANN, yANN = 'xDNN_300_650.npy', 'yDNN_300_650.npy', 'xANN_300_650.npy', 'yANN_300_650.npy'
    if pTrange=='650_1000':
        xDNN, yDNN, xANN, yANN = 'xDNN_650_1000.npy', 'yDNN_650_1000.npy', 'xANN_650_1000.npy', 'yANN_650_1000.npy'
    if pTrange=='1000_2000':
        xDNN, yDNN, xANN, yANN = 'xDNN_1000_2000.npy', 'yDNN_1000_2000.npy', 'xANN_1000_2000.npy', 'yANN_1000_2000.npy'
    if pTrange=='2000_3000':
        xDNN, yDNN, xANN, yANN = 'xDNN_2000_3000.npy', 'yDNN_2000_3000.npy', 'xANN_2000_3000.npy', 'yANN_2000_3000.npy'
    zNN_x = np.load(xDNN)
    zNN_y = np.load(yDNN)
    zANN_lambda10_x = np.load(xANN)
    zANN_lambda10_y = np.load(yANN)

    if(axis=="x" and NN=="NN"): return zNN_x
    if(axis=="y" and NN=="NN"): return zNN_y
    if(axis=="x" and NN=="ANN"): return zANN_lambda10_x
    if(axis=="y" and NN=="ANN"): return zANN_lambda10_y

def getANNROCresults_mass(axis, NN, pTrange=''):
    if pTrange=='':
        # xDNN, yDNN, xANN, yANN = 'xDNN_massCut.npy', 'yDNN_massCut.npy', 'xANN_massCut.npy', 'yANN_massCut.npy'
        xDNN, yDNN, xANN, yANN = 'FromDavide/allmass_ptCut_xDNN.npy', 'FromDavide/allmass_ptCut_yDNN.npy', 'FromDavide/allmass_ptCut_xANN.npy', 'FromDavide/allmass_ptCut_yANN.npy'
    if pTrange=='300_650':
        # xDNN, yDNN, xANN, yANN = 'xDNN_massCut_300_650.npy', 'yDNN_massCut_300_650.npy', 'xANN_massCut_300_650.npy', 'yANN_massCut_300_650.npy'
        xDNN, yDNN, xANN, yANN = 'FromDavide/allmass_ptCut_300_650_xDNN.npy', 'FromDavide/allmass_ptCut_300_650_yDNN.npy', 'FromDavide/allmass_ptCut_300_650_yANN.npy', 'FromDavide/allmass_ptCut_300_650_xANN.npy'
    if pTrange=='650_1000':
        # xDNN, yDNN, xANN, yANN = 'xDNN_massCut_650_1000.npy', 'yDNN_massCut_650_1000.npy', 'xANN_massCut_650_1000.npy', 'yANN_massCut_650_1000.npy'
        xDNN, yDNN, xANN, yANN = 'FromDavide/allmass_ptCut_650_1000_yDNN.npy', 'FromDavide/allmass_ptCut_650_1000_xDNN.npy', 'FromDavide/allmass_ptCut_650_1000_xANN.npy', 'FromDavide/allmass_ptCut_650_1000_yANN.npy'
    if pTrange=='1000_2000':
        # xDNN, yDNN, xANN, yANN = 'xDNN_massCut_1000_2000.npy', 'yDNN_massCut_1000_2000.npy', 'xANN_massCut_1000_2000.npy', 'yANN_massCut_1000_2000.npy'
        xDNN, yDNN, xANN, yANN = 'FromDavide/allmass_ptCut_1000_2000_xDNN.npy', 'FromDavide/allmass_ptCut_1000_2000_yDNN.npy', 'FromDavide/allmass_ptCut_1000_2000_xANN.npy', 'FromDavide/allmass_ptCut_1000_2000_yANN.npy'
    if pTrange=='2000_3000':
        # xDNN, yDNN, xANN, yANN = 'xDNN_massCut_2000_3000.npy', 'yDNN_massCut_2000_3000.npy', 'xANN_massCut_2000_3000.npy', 'yANN_massCut_2000_3000.npy'
        xDNN, yDNN, xANN, yANN = 'FromDavide/allmass_ptCut_2000_3000_xDNN.npy', 'FromDavide/allmass_ptCut_2000_3000_yDNN.npy', 'FromDavide/allmass_ptCut_2000_3000_xANN.npy', 'FromDavide/allmass_ptCut_2000_3000_yANN.npy'
    zNN_x = np.load(xDNN)
    zNN_y = np.load(yDNN)
    zANN_lambda10_x = np.load(xANN)
    zANN_lambda10_y = np.load(yANN)

    if(axis=="x" and NN=="NN"): return zNN_x
    if(axis=="y" and NN=="NN"): return zNN_y
    if(axis=="x" and NN=="ANN"): return zANN_lambda10_x
    if(axis=="y" and NN=="ANN"): return zANN_lambda10_y



class trivar_scores():
    def __init__(self, name, score_file):
        intreename = "FlatSubstructureJetTree"
        self.name = name
        self.score_file = score_file
        #self.events = uproot.open(score_file+":"+intreename)
        #self.scores = self.events.arrays( library="pd")
        branches = []
        mycopy = tree.GetListOfBranches()
        for i in mycopy:
            branches.append(i)
        arr = tree2array(tree, branches = branches, include_weight = False)
        self.scores = pd.DataFrame(arr)

        self.scores["chris_weight"] = (self.scores["fjet_weight_pt"])
        self.scores["xsec_weight"] = np.vectorize(assign_weights)(self.scores["EventInfo_mcChannelNumber"],self.scores["EventInfo_mcEventWeight"])
        self.scores["flat_weight"] = np.vectorize(get_flat_weight)(self.scores["fjet_pt"],self.scores["EventInfo_mcChannelNumber"])
        self.scores["no_weight"] = np.ones_like(self.scores.fjet_pt)

        ## Make the pt spectrum smooth (Chris Delitzsch advice)
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364702][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364703][self.scores.fjet_pt > 1000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]
        alpha = self.scores[self.scores.EventInfo_mcChannelNumber == 364704][self.scores.fjet_pt > 2000]
        self.scores  = self.scores[self.scores.index.isin(alpha.index) == False]

        #coeffs_mass_high = [143.346574141,-0.226450777605,0.000389338881315,-3.3948387014e-07,1.6059552279e-10,-3.89697376333e-14,3.81538674411e-18]
        #coeffs_mass_low = [78.0015279678,-0.0607637891015,0.000154878939873,-1.85055756284e-07,1.06053761725e-10,-2.9181422716e-14,3.09607176224e-18]
        #coeffs_d2 = [1.86287598712,-0.00286891844597,6.51440728353e-06,-7.14076683933e-09,3.97453495445e-12,-1.07885298604e-15,1.1338084323e-19]
        #coeffs_ntrk = [18.1029210508,0.0328710277742,-4.90091461191e-05,3.72086065666e-08,-1.57111307275e-11,3.50912856537e-15,-3.2345326821e-19]


        coeffs_mass_low = [77.85195198272105,-0.04190870755297197,0.00010148243081053968,-1.2646715469383716e-07,7.579631867406234e-11,-2.1810858771189926e-14,2.4131259557938418e-18]
        coeffs_mass_high = [138.40389824173184,-0.1841270515643543,0.0003150778420142889,-2.8146937922756945e-07,1.3687749824011263e-10,-3.370270044494874e-14,3.2886002834089895e-18]
        coeffs_d2 = [1.1962224520689877,0.0007051153225402016,-7.368355018553183e-07,-5.841704226982689e-11,4.1301607038564777e-13,-1.933293321407319e-16,2.7326862198181657e-20]
        coeffs_ntrk = [15.838972910273808,0.059376592913538105,-0.00010408419300237432,9.238395877087256e-08,-4.458514804353202e-11,1.1054941188725808e-14,-1.1013796203558003e-18]

        self.scores["d2_cut"] = np.vectorize(lambda x:coeffs_d2[0]+x*coeffs_d2[1]+coeffs_d2[2]*x**2+coeffs_d2[3]*x**3+coeffs_d2[4]*x**4+coeffs_d2[5]*x**5+coeffs_d2[6]*x**6)(self.scores.fjet_pt)
        self.scores["ntrk_cut"] = np.vectorize(lambda x:coeffs_ntrk[0]+x*coeffs_ntrk[1]+coeffs_ntrk[2]*x**2+coeffs_ntrk[3]*x**3+coeffs_ntrk[4]*x**4+coeffs_ntrk[5]*x**5+coeffs_ntrk[6]*x**6)(self.scores.fjet_pt)
        self.scores["mlow_cut"] = np.vectorize(lambda x:coeffs_mass_low[0]+x*coeffs_mass_low[1]+coeffs_mass_low[2]*x**2+coeffs_mass_low[3]*x**3+coeffs_mass_low[4]*x**4+coeffs_mass_low[5]*x**5+coeffs_mass_low[6]*x**6)(self.scores.fjet_pt)
        self.scores["mhigh_cut"] = np.vectorize(lambda x:coeffs_mass_high[0]+x*coeffs_mass_high[1]+coeffs_mass_high[2]*x**2+coeffs_mass_high[3]*x**3+coeffs_mass_high[4]*x**4+coeffs_mass_high[5]*x**5+coeffs_mass_high[6]*x**6)(self.scores.fjet_pt)

        self.signal = self.scores[self.scores.EventInfo_mcChannelNumber>370000]
        self.bg = self.scores[self.scores.EventInfo_mcChannelNumber<370000]

        self.signal_tagged = self.signal[self.signal.fjet_m > self.signal["mlow_cut"]][self.signal.fjet_m < self.signal["mhigh_cut"]][self.signal.fjet_d2 < self.signal["d2_cut"]][self.signal.fjet_ntrk < self.signal["ntrk_cut"]]
        self.bg_tagged = self.bg[self.bg.fjet_m > self.bg["mlow_cut"]][self.bg.fjet_m < self.bg["mhigh_cut"]][self.bg.fjet_d2 < self.bg["d2_cut"]][self.bg.fjet_ntrk < self.bg["ntrk_cut"]]
        self.bg_untagged  = self.bg[self.bg.index.isin(self.bg_tagged.index) == False]

        print (self.name)
        print ("signal ratio:" ,len(self.signal_tagged)/len(self.signal))
        print ("bg ratio:" ,len(self.bg_tagged)/len(self.bg))

def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))


def make_efficiencies_pt(taggers,minpt, maxpt,weight="chris_weight", prefix='', cutmass=False):
    #plt.figure(figsize=(16,12))

    # colours = [ROOT.kMagenta -4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]
    colours = [ ROOT.kAzure + 7, ROOT.kMagenta -4, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    count = 0
    # mg = TMultiGraph()
    c1 = ap.canvas(num_pads=1, batch=True)
    mg = TMultiGraph()
    if minpt==200:
        if cutmass==False:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_300_650()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_300_650()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults("x", "NN",'300_650')), getANNROCresults("x", "NN", '300_650'), getANNROCresults("y", "NN", '300_650'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults("x", "ANN",'300_650')), getANNROCresults("x", "ANN",'300_650'), getANNROCresults("y", "ANN",'300_650'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")
        if cutmass==True:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_mass_300_650()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_mass_300_650()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults_mass("x", "NN",'300_650')), getANNROCresults_mass("x", "NN", '300_650'), getANNROCresults_mass("y", "NN", '300_650'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults_mass("x", "ANN",'300_650')), getANNROCresults_mass("x", "ANN",'300_650'), getANNROCresults_mass("y", "ANN",'300_650'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")


    if minpt==500:
        if cutmass==False:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_650_1000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_650_1000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults("x", "NN",'650_1000')), getANNROCresults("x", "NN", '650_1000'), getANNROCresults("y", "NN", '650_1000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults("x", "ANN",'650_1000')), getANNROCresults("x", "ANN",'650_1000'), getANNROCresults("y", "ANN",'650_1000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")
        if cutmass==True:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_mass_650_1000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_mass_650_1000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults_mass("x", "NN",'650_1000')), getANNROCresults_mass("x", "NN", '650_1000'), getANNROCresults_mass("y", "NN", '650_1000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults_mass("x", "ANN",'650_1000')), getANNROCresults_mass("x", "ANN",'650_1000'), getANNROCresults_mass("y", "ANN",'650_1000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")

    if minpt==1000:
        if cutmass==False:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_1000_2000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_1000_2000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults("x", "NN",'1000_2000')), getANNROCresults("x", "NN", '1000_2000'), getANNROCresults("y", "NN", '1000_2000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults("x", "ANN",'1000_2000')), getANNROCresults("x", "ANN",'1000_2000'), getANNROCresults("y", "ANN",'1000_2000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")
        if cutmass==True:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_mass_1000_2000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_mass_1000_2000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults_mass("x", "NN",'1000_2000')), getANNROCresults_mass("x", "NN", '1000_2000'), getANNROCresults_mass("y", "NN", '1000_2000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults_mass("x", "ANN",'1000_2000')), getANNROCresults_mass("x", "ANN",'1000_2000'), getANNROCresults_mass("y", "ANN",'1000_2000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")

    if minpt==2000:
        if cutmass==False:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_2000_3000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_2000_3000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults("x", "NN",'2000_3000')), getANNROCresults("x", "NN", '2000_3000'), getANNROCresults("y", "NN", '2000_3000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults("x", "ANN",'2000_3000')), getANNROCresults("x", "ANN",'2000_3000'), getANNROCresults("y", "ANN",'2000_3000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")
        if cutmass==True:
            tprs, fprs, auc = taggers["LundNet_class"].get_roc_mass_2000_3000()
            inv = 1/fprs
            inv[inv == np.inf] = 1e10
            label = "LundNet^{NN}"
            h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            h = c1.graph(h, linestyle=1, linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
            mg.Add(h, "L")

            # tprs, fprs, auc = taggers["LundNet"].get_roc_mass_2000_3000()
            # inv = 1/fprs
            # inv[inv == np.inf] = 1e10
            # label = "LundNet^{ANN}"
            # h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
            # h = c1.graph(h, linestyle=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="AL", label=label)
            # mg.Add(h, "L")
            #
            # ## DNN
            # h = TGraph(len(getANNROCresults_mass("x", "NN",'2000_3000')), getANNROCresults_mass("x", "NN", '2000_3000'), getANNROCresults_mass("y", "NN", '2000_3000'))
            # h = c1.graph(h, linestyle=3, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
            # mg.Add(h, "L")
            #
            # ## ANN
            # h = TGraph(len(getANNROCresults_mass("x", "ANN",'2000_3000')), getANNROCresults_mass("x", "ANN",'2000_3000'), getANNROCresults_mass("y", "ANN",'2000_3000'))
            # h = c1.graph(h, linestyle=4, linecolor=ROOT.kBlue, markercolor=ROOT.kBlue, markerstyle=1, option="AL", label="z_{ANN}^{#lambda=10}")
            # mg.Add(h, "L")




    mg.Draw()
    c1.xlabel('Signal efficiency')
    c1.ylabel('Background rejection')
    c1.xlim(0.2, 1) ## c1.xlim(0.2, 1)
    c1.ylim(1, 1e7)

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #         "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
    #         "p_{T} #in [%d, %d] GeV" % (minpt, maxpt),
    #         ], qualifier='Simulation Preliminary')
    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 "#scale[0.85]{p_{T} #in [%d, %d] GeV}"% (minpt, maxpt),
                 ( "#scale[0.85]{Cut on m_{J} from 3-var tagger}" if cutmass else "")
            ], qualifier='Simulation Preliminary')
    c1.log()
    c1.legend(xmin=0.7, xmax=0.9)
    c1.xlim(0.2, 1)
    if cutmass==False:
        c1.save("{}/Efficiencies_taggers_{}_{}.png".format(prefix, minpt, maxpt))
        c1.save("{}/Efficiencies_taggers_{}_{}.pdf".format(prefix, minpt, maxpt))
        c1.save("{}/Efficiencies_taggers_{}_{}.eps".format(prefix, minpt, maxpt))
        pass
    if cutmass==True:
        c1.save("{}/Efficiencies_taggers_{}_{}_massCut.png".format(prefix, minpt, maxpt))
        c1.save("{}/Efficiencies_taggers_{}_{}_massCut.pdf".format(prefix, minpt, maxpt))
        c1.save("{}/Efficiencies_taggers_{}_{}_massCut.eps".format(prefix, minpt, maxpt))
        pass


def mass_sculpting(taggers, weight="chris_weight", prefix='', wp=0.5):

    bins = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)
    # bins = np.linspace(50, 300, (300 - 50) // 7 + 1, endpoint=True)
    # bins = np.linspace(50, 300, (300 - 50) // 10 + 1, endpoint=True)

    # ##print(tagger.name)

    jsd_classifier = 0
    jsd_combined   = 0

    c1 = ap.canvas(num_pads=1, batch=True)
    p0= c1.pads()


    #hTotalBG  = c1.hist(np.asarray(tagger.bg["fjet_m"]), bins=bins, weights=np.asarray(tagger.bg[weight]), fillstyle = 3353, fillcolor=ROOT.KGray + 2, label="Total BG", linestyle=1, option='HIST', normalise=True)

    for idx, tagger in enumerate(taggers):
        if idx ==0:
            hTotalBG  = c1.hist(np.asarray(taggers[tagger].bg["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].bg[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True) ## Untagged BG
            #hUntagBG  = c1.hist(np.asarray(tagger.bg_untagged["fjet_m"]), bins=bins, weights=np.asarray(tagger.bg_untagged[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True) ## Untagged BG
            hSignal   = c1.hist(np.asarray(taggers[tagger].signal["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].signal[weight]), fillstyle = 3353, fillcolor=ROOT.kGray+1, label="Signal", linestyle=1, linecolor=ROOT.kGray+1, option='HIST', normalise=True)
        if taggers[tagger].name=="LundNet_class":
            hTaggedBG = c1.hist(np.asarray(taggers[tagger].bg_tagged["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].bg_tagged[weight]), label="LundNet^{NN}", linecolor=ROOT.kAzure + 7, linestyle=1, option='HIST', normalise=True)  ## Tagged BG
            # p, _ = np.histogram(np.asarray(taggers[tagger].signal["fjet_m"]), bins=bins, density=1.)
            # f, _ = np.histogram(np.asarray(taggers[tagger].bg_tagged["fjet_m"]), bins=bins, density=1.)
            jsd_classifier = jsd_divergence(hSignal, hTaggedBG)
        if taggers[tagger].name=="LundNet":
            hTaggedBG = c1.hist(np.asarray(taggers[tagger].bg_tagged["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].bg_tagged[weight]), label="LundNet^{ANN}", linecolor=ROOT.kMagenta-4, linestyle=2, option='HIST', normalise=True)  ## Tagged BG
            # p, _ = np.histogram(np.asarray(taggers[tagger].bg["fjet_m"]), bins=bins, density=1.)
            # f, _ = np.histogram(np.asarray(taggers[tagger].bg_tagged["fjet_m"]), bins=bins, density=1.)
            jsd_combined = jsd_divergence(hTotalBG, hTaggedBG)
        # if taggers[tagger].name=="LundNet_class": hTaggedBG = c1.hist(np.asarray(taggers[tagger].signal_tagged["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].signal_tagged[weight]), label="LundNet^{NN}", linecolor=ROOT.kAzure + 7, linestyle=2, option='HIST', normalise=True)  ## Tagged BG
        # if taggers[tagger].name=="LundNet":       hTaggedBG = c1.hist(np.asarray(taggers[tagger].signal_tagged["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].signal_tagged[weight]), label="LundNet^{ANN}", linecolor=ROOT.kMagenta-4, linestyle=2, option='HIST', normalise=True)  ## Tagged BG

        ## calculate JSD


    # hTotalBG  = c1.hist(np.asarray(taggers[tagger].bg["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].bg[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True, display=False) ## Untagged BG
    # hTaggedBG = c1.hist(np.asarray(taggers[tagger].bg_tagged["fjet_m"]), bins=bins, weights=np.asarray(taggers[tagger].bg_tagged[weight]), label="LundNet^{NN}", linecolor=ROOT.kAzure + 7, linestyle=2, option='HIST', normalise=True, display=False)  ## Tagged BG
    # sum = 0
    # for i in range(1, hTotalBG.GetNbinsX()+1):
    #     sum += (hTaggedBG.GetBinContent(i) - hTotalBG.GetBinContent(i))* (hTaggedBG.GetBinContent(i) - hTotalBG.GetBinContent(i)) / (hTotalBG.GetBinContent(i))
    # print(sum / hTotalBG.GetNbinsX())
    c1.xlabel('Large-R jet mass [GeV]')
    c1.ylabel('Fraction of jets / 5 GeV')

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #         "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
    #         # "JSD = {}".format(round(jsd,5)),
    #         ], qualifier='Simulation Preliminary')

    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 # "#scale[0.85]{KL^{NN} = %.4f, KL^{ANN} = %.4f}",
            ], qualifier='Simulation Preliminary')
    c1.log()
    c1.ylim(1e-5, 10)
    c1.legend(xmin=0.7, xmax=0.9)
    c1.save("{}/mass_sculpting.png".format(prefix))
    c1.save("{}/mass_sculpting.pdf".format(prefix))
    c1.save("{}/mass_sculpting.eps".format(prefix))


def mass_sculpting_ptcut(taggers, minpt,maxpt,weight="chris_weight", prefix='', wp=0.5):

    bins = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)


    # p, _ = np.histogram(bg_all, bins=bins, density=1.)
    # f, _ = np.histogram(bgt, bins=bins, density=1.)
    # jsd = JSD(p,f)

    jsd_classifier = 0
    jsd_combined   = 0

    c1 = ap.canvas(num_pads=1, batch=True)
    p0= c1.pads()

    for idx, tagger in enumerate(taggers):
        if idx==0:
            bg_all = taggers[tagger].bg[taggers[tagger].bg.fjet_pt < maxpt ][taggers[tagger].bg.fjet_pt > minpt ]
            signal = taggers[tagger].signal[taggers[tagger].signal.fjet_pt < maxpt ][taggers[tagger].signal.fjet_pt > minpt ]
            hUntagBG  = c1.hist(np.asarray(bg_all["fjet_m"]), bins=bins, weights=np.asarray(bg_all[weight]), fillcolor=ROOT.kGray + 2,  label="QCD jets", option='HIST', normalise=True) ## Untagged BG
            hSignal   = c1.hist(np.asarray(signal["fjet_m"]), bins=bins, weights=np.asarray(signal[weight]), fillstyle = 3353, fillcolor=ROOT.kGray+1, label="Signal", linestyle=1, linecolor=ROOT.kGray+1, option='HIST', normalise=True)

        #bgu = tagger.bg_untagged[tagger.bg_untagged.fjet_pt < maxpt ][tagger.bg_untagged.fjet_pt > minpt ]
        bgt = taggers[tagger].bg_tagged[taggers[tagger].bg_tagged.fjet_pt < maxpt ][taggers[tagger].bg_tagged.fjet_pt > minpt ]
        # bgt = taggers[tagger].signal_tagged[taggers[tagger].signal_tagged.fjet_pt < maxpt ][taggers[tagger].signal_tagged.fjet_pt > minpt ]

        if taggers[tagger].name=="LundNet_class":
            hTaggedBG = c1.hist(np.asarray(bgt["fjet_m"]), bins=bins, weights=np.asarray(bgt[weight]), label="LundNet^{NN}", linecolor=ROOT.kAzure + 7, linestyle=1, option='HIST', normalise=True)  ## Tagged BG
            jsd_classifier = jsd_divergence(hSignal, hTaggedBG)
        if taggers[tagger].name=="LundNet":
            hTaggedBG = c1.hist(np.asarray(bgt["fjet_m"]), bins=bins, weights=np.asarray(bgt[weight]), label="LundNet^{ANN}", linecolor=ROOT.kMagenta-4, linestyle=2, option='HIST', normalise=True)  ## Tagged BG
            jsd_combined = jsd_divergence(hUntagBG, hTaggedBG)


    c1.xlabel('Large-R jet mass [GeV]')
    c1.ylabel('Fraction of jets / 5 GeV')

    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #         "anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets",
    #         "p_{T} #in [%d, %d] GeV" % (minpt, maxpt),
    #         # "JSD = {}".format(round(jsd,5)),
    #         ], qualifier='Simulation Preliminary')
    c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}")+" #scale[0.85]{p_{T} #in [%d, %d] GeV}"% (minpt, maxpt),
            ], qualifier='Simulation Preliminary')
    c1.log()
    c1.legend(xmin=0.7, xmax=0.9)
    c1.ylim(1e-5, 100)
    c1.save("{}/mass_sculpting_{}_{}.png".format(prefix, minpt, maxpt))
    c1.save("{}/mass_sculpting_{}_{}.pdf".format(prefix, minpt, maxpt))
    c1.save("{}/mass_sculpting_{}_{}.eps".format(prefix, minpt, maxpt))



def pt_spectrum(taggers,weight="chris_weight", prefix=''):

    colours = [ROOT.kMagenta -4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]
    c = ap.canvas(num_pads=1, batch=True)
    p0= c.pads()

    nbins = 50
    h_bg_total = TH1D ("bg_total","bg_total",nbins,0,3000)
    fh(h_bg_total,taggers["LundNet_class"].bg["fjet_pt"],taggers["LundNet_class"].bg[weight])
    h_bg_total = c.hist(h_bg_total, linecolor=colours[0], linestyle=2, option='HIST', label="All background")
    #fh(h_bg_total,taggers["LundNet_class"].bg["fjet_pt"])
    a_bkg = h2a(h_bg_total)



    h_bg_plots = []
    bg_plots = []
    count = 1
    for t in taggers:
        bg_plots.append (TH1D ("bg_{}".format(taggers[t].name),"bg_{}".format(taggers[t].name),nbins,0,3000))
        fh(bg_plots[-1],taggers[t].bg_tagged["fjet_pt"],taggers[t].bg_tagged[weight])
        h = c.hist(bg_plots[-1], linecolor=colours[count], linestyle=2, option='HIST', label=taggers[t].name)
        #fh(bg_plots[-1],taggers[t].bg_tagged["fjet_pt"])
        bg_plots.append( h2a(bg_plots[-1]))
        count+=1
        #plt.semilogy(np.linspace(0, 3000, nbins),bg_plots[-1],
        #             label="tagged background {}".format(taggers[t].name))
    #plt.semilogy(np.linspace(0, 3000, nbins), a_bkg, label="all background")


    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",] , qualifier='Simulation Preliminary')

    # -- Axis labels
    c.xlabel('Large-R jet p_{T} [GeV]')
    c.ylabel('Events')

    # -- Log
    c.pads()[0].ylim(0.001, 1e16)
    #p0.ylim(1, 1e16)
    c.log()
    c.legend()
    c.save("{}/ptspectrum.png".format(prefix))


def bgrej_mu(taggers,weight="chris_weight", prefix='', wp=0.5):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    # bins = np.linspace(0, 70, 25 + 1, endpoint=True)
    # bins =np.array([16, 20, 24, 28, 32, 36, 44, 52, 60, 68], dtype=float)
    bins =np.array([16, 20, 24, 28, 32, 36, 44, 55, 68], dtype=float)
    # bins =np.array([16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72], dtype=float)

    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if t=="LundNet_class":  label = "LundNet^{NN}"
        if t=="LundNet"      :  label = "LundNet^{ANN}"
        ## Get total background
        h_bg_total = c.hist(np.array(taggers[t].bg["mu"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        ## Get tagged background
        h_bg = c.hist(np.array(taggers[t].bg_tagged["mu"]), weights=np.array(taggers[t].bg_tagged[weight]), bins=bins, display=False)
        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        hratio = ROOT.TH1D("", "", len(bins)-1, bins)
        hratio.Divide(h_bg_total, h_bg)
        c.hist(hratio, option='P E2', bins=bins, label=label, markercolor=colours[count], linecolor=colours[count], fillcolor=colours[count], alpha=0.3)

        # c.ratio_plot((h_bg_total,      h_bg), option='P E2', bins=bins, label=label, linecolor=colours[count])
        count +=1

    c.xlabel('< #mu >')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 # "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.9)
    c.ylim(1, 1e5)
    c.save("{}/bgrej_mu.png".format(prefix))
    c.save("{}/bgrej_mu.pdf".format(prefix))
    c.save("{}/bgrej_mu.eps".format(prefix))

def bgrej_npv(taggers,weight="chris_weight", prefix='', wp=0.5):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    # bins = np.linspace(0, 70, 25 + 1, endpoint=True)
    bins =np.array([10, 15, 20, 25, 30, 35, 40], dtype=float)
    # bins =np.array([16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72], dtype=float)

    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if t=="LundNet_class":  label = "LundNet^{NN}"
        if t=="LundNet"      :  label = "LundNet^{ANN}"
        ## Get total background
        h_bg_total = c.hist(np.array(taggers[t].bg["npv"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        ## Get tagged background
        h_bg = c.hist(np.array(taggers[t].bg_tagged["npv"]), weights=np.array(taggers[t].bg_tagged[weight]), bins=bins, display=False)
        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        hratio = ROOT.TH1D("", "", len(bins)-1, bins)
        hratio.Divide(h_bg_total, h_bg)
        # hratio = GetBinominalErrors(h_bg_total, h_bg)
        c.hist(hratio, option='P E2', bins=bins, label=label, markercolor=colours[count], linecolor=colours[count], fillcolor=colours[count], alpha=0.3)

        # c.ratio_plot((h_bg_total,      h_bg), option='P E2', bins=bins, label=label, linecolor=colours[count])
        count +=1

    c.xlabel('NPV')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 # "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.9)
    c.ylim(1, 1e5)
    c.save("{}/bgrej_npv.png".format(prefix))


def pt_bgrej(taggers,weight="chris_weight", prefix='', wp=0.5):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    # bins = np.linspace(250, 3250, 14 + 1, endpoint=True)
    bins = np.linspace(350, 3150, 15+1) ## use same binning as Kevin
    kevin_results = np.load('Nominal_metrics.npz')

    total_binned_br_50 = kevin_results['total_binned_br_50']
    total_binned_br_80 = kevin_results['total_binned_br_80']
    ## add kevin results
    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if t=="LundNet_class":  label = "LundNet^{NN}"
        if t=="LundNet"      :  label = "LundNet^{ANN}"

        if t=="HerwigAngular" or t=="HerwigDipole" or t=="SherpaCluster": 
            continue

        ## Get total background
        h_bg_total = c.hist(np.array(taggers[t].bg["fjet_pt"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        ## Get tagged background
        h_bg = c.hist(np.array(taggers[t].bg_tagged["fjet_pt"]), weights=np.array(taggers[t].bg_tagged[weight]), bins=bins, display=False)
        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        hratio = ROOT.TH1D("", "", len(bins)-1, bins)
        hratio.Divide(h_bg_total,h_bg, 1., 1., "B")
        if t=="LundNet_class":  markerstyle,linestyle  = 20,1
        if t=="LundNet"      :  markerstyle,linestyle =4,9

        c.hist(hratio, option='P E2', bins=bins, label=label, linestyle=linestyle, markerstyle=markerstyle, markercolor=colours[count], linecolor=colours[count], fillcolor=colours[count], alpha=0.3)

        # c.ratio_plot((h_bg_total,      h_bg), option='P E2', bins=bins, label=label, linecolor=colours[count])
        count +=1

    if wp==0.5:
        c.hist(total_binned_br_50, option='P E2', bins=bins, label='ParticleNet', linestyle=9, markerstyle=4, markercolor=colours[1], linecolor=colours[1], fillcolor=colours[1] ,alpha=0.3)
    if wp==0.8:
        c.hist(total_binned_br_80, option='P E2', bins=bins, label='ParticleNet', linestyle=9, markerstyle=4, markercolor=colours[1], linecolor=colours[1], fillcolor=colours[1], alpha=0.3)



    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 # "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.9)
    c.ylim(1, 1e5)
    # c.ylim(25, 225)
    c.save("{}/pt_bgrej.png".format(prefix))
    c.save("{}/pt_bgrej.pdf".format(prefix))
    c.save("{}/pt_bgrej.eps".format(prefix))

def pt_bgrej_all(taggers,weight="chris_weight", prefix='', wp=0.5):

    l=TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextSize(0.042)
    s=TLatex()
    s.SetNDC();
    s.SetTextFont(42)
    s.SetTextSize(0.04)
    legend=ROOT.TLegend(0.18,0.56,0.85,0.73)
    legend.SetNColumns(2)
    legend.Clear()
    legend.SetFillStyle(0)

    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982'), ROOT.kRed]

    # bins = np.linspace(200, 3000, 16 + 1, endpoint=True)
    bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)



    c = ap.canvas(num_pads=1, batch=True)
    h_bg_total = c.hist(np.array(taggers["LundNet_class"].bg["fjet_pt"]), weights=np.array(taggers["LundNet_class"].bg[weight]), bins=bins, display=False)
    h_bg = c.hist(np.array(taggers["LundNet_class"].bg_tagged["fjet_pt"]), weights=np.array(taggers["LundNet_class"].bg_tagged[weight]), bins=bins, display=False)
    hratio = ROOT.TH1D("", "", len(bins)-1, bins)
    hratio.Divide(h_bg_total,h_bg, 1., 1., "B")


    if wp==0.5:
        file = ROOT.TFile.Open("/home/jmsardain/LJPTagger/ljptagger/Plotting/FromShudong/figs_bkgrej_vs_pt_50.root")
    # if wp==0.8:
    #     file = ROOT.TFile.Open("/home/jmsardain/LJPTagger/ljptagger/Plotting/FromShudong/figs_bkgrej_vs_pt_80.root")
    hratio = c.hist(hratio, option='P E2', bins=bins, label="LundNet^{NN}", linestyle=1, markerstyle=20, markercolor=colours[0], linecolor=colours[0], fillcolor=colours[0], alpha=0.3)
    legend.AddEntry(hratio, "LundNet^{NN}", "pe")

    # # h_ParticleTransformer_bkgrej_vs_pt
    # h_ParticleTransformer_bkgrej_vs_pt = file.Get("h_ParticleTransformer_bkgrej_vs_pt")
    # h = ROOT.TH1D("", "", len(bins)-1, bins)
    # for i in range(1, h.GetNbinsX()+1):
    #     h.SetBinContent(i, h_ParticleTransformer_bkgrej_vs_pt.GetBinContent(i))
    #     h.SetBinError(i, h_ParticleTransformer_bkgrej_vs_pt.GetBinError(i))
    #
    # h = c.hist(h, option='P E2', bins=bins, label="ParticleT", linestyle=1, markerstyle=22, markercolor=colours[1], linecolor=colours[1], fillcolor=colours[1], alpha=0.3)
    # legend.AddEntry(h, "ParT", "pe")
    # # h_ParticleNet_bkgrej_vs_pt
    # h_ParticleNet_bkgrej_vs_pt = file.Get("h_ParticleNet_bkgrej_vs_pt")
    # h_ParticleNet_bkgrej_vs_pt = c.hist(h_ParticleNet_bkgrej_vs_pt, option='P E2', bins=bins, label="ParticleNet", linestyle=1, markerstyle=29, markercolor=colours[2], linecolor=colours[2], fillcolor=colours[2], alpha=0.3)
    # legend.AddEntry(h_ParticleNet_bkgrej_vs_pt, "ParticleNet", "pe")
    # # h_PFN_bkgrej_vs_pt
    # h_PFN_bkgrej_vs_pt = file.Get("h_PFN_bkgrej_vs_pt")
    # h_PFN_bkgrej_vs_pt = c.hist(h_PFN_bkgrej_vs_pt, option='P E2', bins=bins, label="PFN", linestyle=1, markerstyle=21, markercolor=colours[3], linecolor=colours[3], fillcolor=colours[3], alpha=0.3)
    # legend.AddEntry(h_PFN_bkgrej_vs_pt, "PFN", "pe")
    # # h_EFN_bkgrej_vs_pt
    # h_EFN_bkgrej_vs_pt = file.Get("h_EFN_bkgrej_vs_pt")
    # h_EFN_bkgrej_vs_pt = c.hist(h_EFN_bkgrej_vs_pt, option='P E2', bins=bins, label="EFN", linestyle=1, markerstyle=23, markercolor=colours[4], linecolor=colours[4], fillcolor=colours[4], alpha=0.3)
    # legend.AddEntry(h_EFN_bkgrej_vs_pt, "EFN", "pe")
    #
    # # zDNN
    # hDNN = ROOT.TH1D("", "", len(bins)-1, bins)
    #
    # if wp==0.5: xDNN, yDNN, yDNN_err, xANN, yANN, yANN_err = getANNBkgRejpT("x", "NN"), getANNBkgRejpT("y", "NN"), getANNBkgRejpT("yerr", "NN"), getANNBkgRejpT("x", "ANN"), getANNBkgRejpT("y", "ANN"), getANNBkgRejpT("yerr", "ANN")
    # for i in range(1, hDNN.GetNbinsX()+1):
    #     hDNN.SetBinContent(i, yDNN[i-1])
    #     hDNN.SetBinError(i, yDNN_err[i-1])
    #
    # hDNN = c.hist(hDNN, option='P E2', bins=bins, label="z_{NN}", linestyle=1, markerstyle=24, markercolor=colours[5], linecolor=colours[5], fillcolor=colours[5], alpha=0.3)
    # legend.AddEntry(hDNN, "z_{NN}", "pe")

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection (1/#varepsilon^{rel}_{bkg})')
    # c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #              "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
    #              ("#scale[0.85]{|#eta| < 2.0, #varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{|#eta| < 2.0, #varepsilon^{rel}_{sig} = 80%}"),
    #              # "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
    #         ], qualifier='Simulation Preliminary')
    # c.legend(xmin=0.7, xmax=0.9)
    c.log()
    l.DrawLatex(0.18, 0.89,        "ATLAS")
    s.DrawLatex(0.18+(0.14), 0.89, " Simulation Preliminary")
    s.DrawLatex(0.18, 0.84,        "#sqrt{s} = 13 TeV, #it{top} tagging")
    s.DrawLatex(0.18, 0.79,        "anti-#it{k_{t}} #it{R}=1.0 UFO Soft-Drop CS+SK jets")
    if wp==0.5:
        s.DrawLatex(0.18, 0.74, " #varepsilon^{rel}_{sig} = 50%, |#eta| < 2.0")
    else:
        s.DrawLatex(0.18, 0.74, "|#eta| < 2.0, #varepsilon^{rel}_{sig} = 80%")

    legend.Draw()
    c.ylim(1, 1e6)
    c.save("{}/fig_01.png".format(prefix))
    c.save("{}/fig_01.pdf".format(prefix))
    c.save("{}/fig_01.eps".format(prefix))

def make_efficiencies_pt_all(taggers,minpt, maxpt,weight="chris_weight", prefix='', cutmass=False):

    l=TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextSize(0.042)
    s=TLatex()
    s.SetNDC();
    s.SetTextFont(42)
    s.SetTextSize(0.04)
    legend=ROOT.TLegend(0.18,0.16,0.65,0.33)
    legend.SetNColumns(2)
    legend.Clear()
    legend.SetFillStyle(0)

    # colours = [ROOT.kMagenta -4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]
    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982'), ROOT.kRed]

    count = 0
    # mg = TMultiGraph()
    c1 = ap.canvas(num_pads=1, batch=True)
    # mg = TMultiGraph()
    file = ROOT.TFile.Open("/home/jmsardain/LJPTagger/ljptagger/Plotting/FromShudong/ROCs.root")



    if minpt==300:  tprs, fprs, auc = taggers["LundNet_class"].get_roc_300_650()
    if minpt==650:  tprs, fprs, auc = taggers["LundNet_class"].get_roc_650_1000()
    if minpt==1000: tprs, fprs, auc = taggers["LundNet_class"].get_roc_1000_2000()
    if minpt==2000: tprs, fprs, auc = taggers["LundNet_class"].get_roc_2000_3000()

    inv = 1/fprs
    inv[inv == np.inf] = 1e10
    label = "LundNet^{NN}"
    print(len(tprs))
    h = TGraph(len(np.array(tprs[::5], dtype=np.float64)), np.array(tprs[::5], dtype=np.float64), np.array(inv[::5], dtype=np.float64))
    h = c1.graph(h, linestyle=1, linewidth=2,linecolor=colours[0], markercolor=colours[0], markerstyle=1, option="AL", label=label)
    h.Draw()
    legend.AddEntry(h, "LundNet^{NN}", 'l')

    # ROC_pt_full_Wtag_ParT = file.Get("ROC_pt_"+str(minpt)+"_"+str(maxpt)+"_Wtag_ParT")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_ParT)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::1000], dtype=np.float64)), np.array(x_values[::1000], dtype=np.float64), np.array(y_values[::1000], dtype=np.float64))
    # h = c1.graph(h, linestyle=2,linewidth=2, linecolor=colours[1], markercolor=colours[1], markerstyle=1, option="L", label="Particle transfomer")
    # h.Draw('SAME')
    # legend.AddEntry(h, "ParT", 'l')
    #
    # # ParticleNet
    # ROC_pt_full_Wtag_PN = file.Get("ROC_pt_"+str(minpt)+"_"+str(maxpt)+"_Wtag_PN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_PN)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::1000], dtype=np.float64)), np.array(x_values[::1000], dtype=np.float64), np.array(y_values[::1000], dtype=np.float64))
    # h = c1.graph(h, linestyle=3,linewidth=2, linecolor=colours[2], markercolor=colours[2], markerstyle=1, option="L", label="ParticleNet")
    # h.Draw('SAME')
    # legend.AddEntry(h, "ParticleNet", 'l')
    #
    # ## PFN
    # ROC_pt_full_Wtag_PFN = file.Get("ROC_pt_"+str(minpt)+"_"+str(maxpt)+"_Wtag_PFN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_PFN)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::2000], dtype=np.float64)), np.array(x_values[::2000], dtype=np.float64), np.array(y_values[::2000], dtype=np.float64))
    # h = c1.graph(h, linestyle=4,linewidth=2, linecolor=colours[3], markercolor=colours[3], markerstyle=1, option="L", label="PFN")
    # h.Draw('SAME')
    # legend.AddEntry(h, "PFN", 'l')
    #
    # ## EFN
    # ROC_pt_full_Wtag_EFN = file.Get("ROC_pt_"+str(minpt)+"_"+str(maxpt)+"_Wtag_EFN")
    # x_values, y_values = get_xy_values_from_tgraph(ROC_pt_full_Wtag_EFN)
    # print(len(x_values))
    # h = TGraph(len(np.array(x_values[::2000], dtype=np.float64)), np.array(x_values[::2000], dtype=np.float64), np.array(y_values[::2000], dtype=np.float64))
    # h = c1.graph(h, linestyle=5,linewidth=2, linecolor=colours[4], markercolor=colours[4], markerstyle=1, option="L", label="EFN")
    # h.Draw('SAME')
    # legend.AddEntry(h, "EFN", 'l')
    #
    # ## DNN
    # h = TGraph(len(getANNROCresults("x", "NN",str(minpt)+'_'+str(maxpt))), getANNROCresults("x", "NN", str(minpt)+'_'+str(maxpt)), getANNROCresults("y", "NN", str(minpt)+'_'+str(maxpt)))
    # h = c1.graph(h, linestyle=6, linewidth=2, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, option="L", label="z_{NN}")
    # h.Draw('SAME')
    # legend.AddEntry(h, "z_{NN}", 'l')



    # mg.Draw()
    c1.xlabel('Signal efficiency (#varepsilon^{rel}_{sig})')
    c1.ylabel('Background rejection (1/#varepsilon^{rel}_{bkg})')
    c1.xlim(0.2, 1) ## c1.xlim(0.2, 1)
    if minpt==200:   c1.ylim(1, 2e4)
    if minpt==500:   c1.ylim(1, 1e5)
    if minpt==1000:  c1.ylim(1, 2e5)
    if minpt==2000:  c1.ylim(1, 1e5)


    # c1.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #              "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
    #              "#scale[0.85]{p_{T} #in [%d, %d] GeV}"% (minpt, maxpt),
    #              ( "#scale[0.85]{Cut on m_{J} from 3-var tagger}" if cutmass else "")
    #         ], qualifier='Simulation Preliminary')
    c1.log()
    # c1.legend(xmin=0.7, xmax=0.9)
    c1.xlim(0.2, 1)

    l.DrawLatex(0.18, 0.89,        "ATLAS")
    s.DrawLatex(0.18+(0.14), 0.89, " Simulation Preliminary")
    s.DrawLatex(0.18, 0.84,        "#sqrt{s} = 13 TeV, #it{top} tagging")
    s.DrawLatex(0.18, 0.79,        "anti-#it{k_{t}} #it{R}=1.0 UFO Soft-Drop CS+SK jets")
    s.DrawLatex(0.18, 0.74, "p_{T} #in [%d, %d] GeV, |#eta| < 2.0"% (minpt, maxpt))

    legend.Draw()

    # c1.save("{}/Efficiencies_taggers_{}_{}.png".format(prefix, minpt, maxpt))
    # c1.save("{}/Efficiencies_taggers_{}_{}.pdf".format(prefix, minpt, maxpt))
    # c1.save("{}/Efficiencies_taggers_{}_{}.eps".format(prefix, minpt, maxpt))
    if minpt==200:
        c1.save("{}/fig_02b.png".format(prefix))
        c1.save("{}/fig_02b.pdf".format(prefix))
        c1.save("{}/fig_02b.eps".format(prefix))
    if minpt==500:
        c1.save("{}/fig_02c.png".format(prefix))
        c1.save("{}/fig_02c.pdf".format(prefix))
        c1.save("{}/fig_02c.eps".format(prefix))
    if minpt==1000:
        c1.save("{}/fig_02d.png".format(prefix))
        c1.save("{}/fig_02d.pdf".format(prefix))
        c1.save("{}/fig_02d.eps".format(prefix))
    if minpt==2000:
        c1.save("{}/fig_02e.png".format(prefix))
        c1.save("{}/fig_02e.pdf".format(prefix))
        c1.save("{}/fig_02e.eps".format(prefix))

    pass


def pt_bgrej_mass(taggers,weight="chris_weight", prefix='', wp=0.5):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    # bins = np.linspace(200, 3000, 16 + 1, endpoint=True)
    bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)

    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if t=="LundNet_class":  label = "LundNet^{NN}"
        if t=="LundNet"      :  label = "LundNet^{ANN}"
        ## Get total background
        #h_bg_total = c.hist(np.array(taggers[t].bgmass50["fjet_pt"]), weights=np.array(taggers[t].bgmass50[weight]), bins=bins, display=False)

        h_bg_total = c.hist(np.array(taggers[t].bg["fjet_pt"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        # h_bg_total = c.hist(np.array(taggers[t].bg["fjet_pt"]), bins=bins, display=False)

        ## Get tagged background
        if wp==0.5:
            h_bg = c.hist(np.array(taggers[t].bgmass50_tagged["fjet_pt"]), weights=np.array(taggers[t].bgmass50_tagged[weight]), bins=bins, display=False)
            # h_bg = c.hist(np.array(taggers[t].bgmass50_tagged["fjet_pt"]), bins=bins, display=False)
        if wp==0.8:
            h_bg = c.hist(np.array(taggers[t].bgmass80_tagged["fjet_pt"]), weights=np.array(taggers[t].bgmass80_tagged[weight]), bins=bins, display=False)

        # print("len(np.array(taggers[t].bgmass50_tagged[fjet_pt]))->", len(np.array(taggers[t].bgmass50_tagged["fjet_pt"])) )
        # print("len(np.array(taggers[t].bgmass50_tagged[weight]))->", len(np.array(taggers[t].bgmass50_tagged[weight])) )


        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        hratio = ROOT.TH1D("", "", len(bins)-1, bins)
        hratio.Divide(h_bg_total,h_bg, 1., 1., "B")
        if t=="LundNet_class":  markerstyle,linestyle  = 20,1
        if t=="LundNet"      :  markerstyle,linestyle =4,9
        c.hist(hratio, option='P E2', bins=bins, label=label, linestyle=linestyle, markerstyle=markerstyle, markercolor=colours[count], linecolor=colours[count], fillcolor=colours[count], alpha=0.3)
        # c.hist(hratio, option='HIST', bins=bins, linecolor=colours[count])
        # c.ratio_plot((h_bg_total,      h_bg), option='HIST', bins=bins, label=label, linecolor=colours[count])
        # c.ratio_plot((h_bg_total,      h_bg), option='E2', bins=bins, linecolor=colours[count])
        if count == len(taggers)-1:
            if wp==0.5:
                xDNN, yDNN, yDNN_err, xANN, yANN, yANN_err = getANNBkgRejpT("x", "NN"), getANNBkgRejpT("y", "NN"), getANNBkgRejpT("yerr", "NN"), getANNBkgRejpT("x", "ANN"), getANNBkgRejpT("y", "ANN"), getANNBkgRejpT("yerr", "ANN")
                bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)
                hDNN = ROOT.TH1D("", "", len(bins)-1, bins)
                hANN = ROOT.TH1D("", "", len(bins)-1, bins)
                # hDNN = ROOT.TGraphErrors()
                # hANN = ROOT.TGraphErrors()
                # for i in range(len(xDNN)):
                #     hDNN.SetPoint(i, xDNN[i], yDNN[i])
                #     hANN.SetPoint(i, xANN[i], yANN[i])
                #     hDNN.SetPointError(i, 0, yDNN_err[i])
                #     hANN.SetPointError(i, 0, yANN_err[i])


                for i in range(1, hDNN.GetNbinsX()+1):
                    hDNN.SetBinContent(i, yDNN[i-1])
                    hDNN.SetBinError(i, yDNN_err[i-1])
                    hANN.SetBinContent(i, yANN[i-1])
                    hANN.SetBinError(i, yANN_err[i-1])

                # mg.Add(h, "L")
                c.hist(hDNN, bins=bins, linestyle=2, linecolor=ROOT.kRed, markerstyle=22, markercolor=ROOT.kRed, fillcolor=ROOT.kRed, alpha=0.3, option="P E2", label='z_{NN}')
                c.hist(hANN, bins=bins, linestyle=4, linecolor=ROOT.kBlue, markerstyle=26, markercolor=ROOT.kBlue, fillcolor=ROOT.kBlue, alpha=0.3, option="P E2", label='z_{ANN}^{#lambda=10}')

            if wp==0.5:
                # md2ntrkCut = np.array([41.006, 59.9157, 82.1475, 89.1412, 96.5043, 99.2185, 85.7983, 77.525, 73.5525, 60.8967, 59.4876, 49.9543, 49.9432, 41.4343, 45.5938, 40.6416 ])
                # md2ntrkCutError = np.array([1.03359, 1.41564, 2.02833, 2.61064, 2.94638, 4.06692, 3.86636, 2.61072, 2.96975, 2.36394, 2.61225, 2.42941, 2.85013, 1.79587, 2.10322, 1.74866 ])
                md2ntrkCut = np.array([34.2932,52.5485,68.6128,78.5824,87.0319,83.3411,74.1299,67.0704,60.3494,51.0008,49.3955,47.3197,38.3147,42.1102,39.1164,36.9524])
                md2ntrkCutError =  np.array([0.576104,0.913001,1.28134,1.7413,2.21326,2.57566,2.60416,1.95647,2.09549,1.74478,1.8297,2.42843,2.00397,2.10696,1.39881,1.64316])
            if wp==0.8:
                # md2ntrkCut = np.array([7.54374, 14.6774, 19.6761, 21.5219, 20.987, 19.2489, 17.6715, 15.7592, 14.4646, 12.8379, 12.3075, 10.6703, 9.49324, 8.18283, 7.43013, 7.2466 ])
                # md2ntrkCutError = np.array([0.00987912, 0.160989, 0.2261, 0.316753, 0.293784, 0.326347, 0.348374, 0.23501, 0.246103, 0.217953, 0.233865, 0.233994, 0.23169, 0.178122, 0.123532, 0.12525 ])
                md2ntrkCut =  np.array([5.81862,7.91733,9.66673,11.0802,11.5294,11.2321,11.6349,10.6324,10.1774,9.44143,8.84036,8.59357,7.55452,7.73503,7.20732,7.17621])
                md2ntrkCutError = np.array([0.0655417,0.0625564,0.0713613,0.106252,0.112123,0.140568,0.172694,0.124742,0.143951,0.13848,0.145318,0.178407,0.179143,0.175055,0.127307,0.136716])

            h3VarTagger = ROOT.TH1D('', '', 16, 200, 3000)
            for i in range(16):
                h3VarTagger.SetBinContent(i+1, md2ntrkCut[i])
                h3VarTagger.SetBinError(i+1, md2ntrkCutError[i])
            c.hist(h3VarTagger, bins=bins, linestyle=1, linecolor=ROOT.kBlack, markercolor=ROOT.kBlack, markerstyle=21, fillcolor=ROOT.kBlack, alpha=0.3, option="P E2", label='3-var tagger')
            # c.hist(h3VarTagger, bins=bins, linestyle=1, linecolor=ROOT.kRed, markercolor=ROOT.kRed, markerstyle=1, alpha=0.3, option="HIST")
            # c.hist(h3VarTagger, bins=bins, option="E2")


        count +=1


    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.9)
    c.ylim(1, 1e5)
    c.save("{}/pt_bgrej_massCut.png".format(prefix))
    c.save("{}/pt_bgrej_massCut.pdf".format(prefix))
    c.save("{}/pt_bgrej_massCut.eps".format(prefix))

def pt_sigeff(taggers,weight="chris_weight", prefix=''):

    colours = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    bins = np.linspace(200, 3000, 16 + 1, endpoint=True)
    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:

        ## Get total background
        h_signal_total = c.hist(np.array(taggers[t].signal["fjet_pt"]), weights=np.array(taggers[t].signal[weight]), bins=bins, display=False)
        ## Get tagged background
        h_signal = c.hist(np.array(taggers[t].signal_tagged["fjet_pt"]), weights=np.array(taggers[t].signal_tagged[weight]), bins=bins, display=False)

        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        c.ratio_plot((h_signal, h_signal_total), option='HIST', offset=1, bins=bins, label=taggers[t].name, linecolor=colours[count])

        count +=1

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Signal efficiency #epsilon^{sig}')

    # c.text(["#sqrt{s} = 13 TeV, #it{top} tagging"],
    #         (["anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets"]),
    #         qualifier='Simulation Preliminary')
    c.log()
    c.legend()
    c.ylim(1, 1e5)
    c.save("{}/pt_sigeff.png".format(prefix))




# def pt_sigeff(taggers,weight="chris_weight"):
#     plt.figure(figsize=[16,12])
#     nbins = 50
#     for t in taggers:
#         some_tagger = t

#     h_sig_total = TH1D ("sig_total","sigtotal",nbins,0,3000)
# #    MASSBINS = np.linspace(200, 3000, (300 - 40) // 5 + 1, endpoint=True)

#     fh(h_sig_total,taggers[some_tagger].signal["fjet_pt"],taggers[some_tagger].signal[weight])
#     a_sig = h2a(h_sig_total)
# #    plt.semilogy(np.linspace(0, 3000, 100), a_sig, label="all signal")

#     for t in taggers:
#         h_sig = TH1D ("sig_{}".format(taggers[t].name),"sig_{}".format(taggers[t].name),nbins,0,3000)
#         fh(h_sig,taggers[t].signal_tagged["fjet_pt"],taggers[t].signal_tagged[weight])
#         sig = h2a(h_sig)
#         plt.plot(np.linspace(0, 3000, nbins),sig/a_sig,
#                      label="tagged signal {}".format(taggers[t].name))
#         rat = sig/a_sig
#         rat = rat[np.logical_not(np.isnan(rat))]
#         print ("sig eff mean:", (rat).mean())


#     plt.ylabel("Reweighted event counts")
#     plt.xlabel("Jet pT $[GeV]$")
#     #plt.xlim(5,500)
#     plt.ylim(10**(-16), 10**(-0))
#     plt.legend()
#     plt.legend(prop={'size': 15})

#     #plt.gca().invert_yaxis()
#     plt.show()



def weights(tagger):
    plt.figure(figsize=[16,8])
    kwargs = dict(alpha = 0.75, bins = 50, density = True, stacked = True,range=(0,1))
    plt.title(label=tagger.name, fontdict=None, loc='center', pad=None)

    #plt.hist(tagger.signal["fjet_weight_pt_dR"],  ** kwargs, label="Signal")
    #plt.hist(tagger.bg["fjet_weight_pt_dR"], ** kwargs, label="Background")

    plt.hist(tagger.signal["fjet_weight_pt_dR"], label="Signal")
    plt.hist(tagger.bg["fjet_weight_pt_dR"], label="Background")
    plt.legend(fontsize=30)
    plt.show()


def plot_metrics(infile, lambda_param, prefix=''):
    metrics = pd.read_csv(infile)
    ##print(metrics)
    metrics = metrics.dropna()


    # -- Loss classifier
    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.plot(metrics.index.values, np.array(metrics["Train_Loss_clsf"].values), label="Train")
    ax.plot(metrics.index.values, np.array(metrics["Val_loss_Class"].values), label="Validation")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss_{clsf}')
    # ax.set_xlim(20, 80)
    plt.legend()
    plt.savefig('{}/Loss_clf.png'.format(prefix))
    plt.clf()

    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.plot(metrics.index.values, np.array(metrics["Train_Loss_adv"].values), label="Train")
    ax.plot(metrics.index.values, np.array(metrics["Val_Loss_Adv"].values), label="Validation")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss_{adv}')
    plt.legend()
    plt.savefig('{}/Loss_adv.png'.format(prefix))
    plt.clf()

    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.plot(metrics.index.values, np.array(metrics["Train_jds"].values), label="Train")
    ax.plot(metrics.index.values, np.array(metrics["Val_jds"].values), label="Validation")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('JSD')
    plt.legend()
    plt.savefig('{}/JSD.png'.format(prefix))
    plt.clf()

    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    ax.plot(metrics.index.values, np.array(metrics["Train_bgrej"].values), label="Train")
    ax.plot(metrics.index.values, np.array(metrics["Val_bgrej"].values), label="Validation")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('BKrej')
    plt.legend()
    plt.savefig('{}/BKrej.png'.format(prefix))
    plt.clf()

    fig, ax = plt.subplots()
    bins = np.linspace(0, 2, 21, endpoint=True)
    train_num = np.array(metrics.Train_bgrej.values+lambda_param/metrics.Train_jds.values)
    test_num  = np.array(metrics.Val_bgrej.values+lambda_param/metrics.Val_jds.values)
    ax.plot(metrics.index.values, train_num, label="Train")
    ax.plot(metrics.index.values, test_num, label="Validation")
    ax.set_xlabel('Epochs')
    ax.set_ylabel("1/BKrej + "+r"$\lambda$"+"/JSD")
    # ax.set_xlim(20, 100)
    plt.legend()
    plt.savefig('{}/ObsOptimizer.png'.format(prefix))
    plt.clf()


    '''

    # -- 1/BKrej
    c3 = ap.canvas(num_pads=1, batch=True)
    hTrainClf  = c3.hist(np.array(metrics.Train_bgrej.values), bins=bins,  label="Train",     linecolor=TColor.GetColor(colours[0]),fillcolor=TColor.GetColor(colours[0]), option='HIST')
    hValClf    = c3.hist(np.array(metrics.Val_bgrej.values),  bins=bins,  label="Validation", linecolor=TColor.GetColor(colours[1]),fillcolor=TColor.GetColor(colours[1]), option='HIST')
    c3.ylabel('BKrej')
    c3.xlabel('Epochs')
    c3.legend()
    c3.save("BKrej.png")

    # -- 1/BKrej - 1/JSD
    c4 = ap.canvas(num_pads=1, batch=True)
    train_num = np.array(metrics.Train_bgrej.values-1/metrics.Train_jds.values)
    test_num  = np.array(metrics.Val_bgrej.values-1/metrics.Val_jds.values)
    hTrainClf  = c4.hist(train_num, bins=bins,  label="Train",     linecolor=TColor.GetColor(colours[0]),fillcolor=TColor.GetColor(colours[0]), option='HIST')
    hValClf    = c4.hist(test_num,  bins=bins,  label="Validation", linecolor=TColor.GetColor(colours[1]),fillcolor=TColor.GetColor(colours[1]), option='HIST')
    c4.ylabel('1/BKrej - 1/JSD')
    c4.xlabel('Epochs')
    c4.legend()
    c4.save("ObsOptimizer.png")
    '''

def plotLundPlane( energy, resp, prefix='', wp=0.5):

    # ml response vs input variables raw
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.linspace(0, 10,  100 + 1, endpoint=True)
    yaxis = np.linspace(0, 10,  100 + 1, endpoint=True)

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    mesh = np.vstack((energy, resp)).T
    fill_hist(h1, mesh)


    c.ylim(0, 2)
    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,         option='COLZ')
    c.hist2d(h1_backdrop, option='AXIS')

    c.xlabel('ln(1/#Delta R)')
    c.ylabel('ln(k_{t})')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
            ], qualifier='Simulation Preliminary')
    c.save("{}/lundplane.png".format(prefix))


def plotPtAlternative(hPythia, hSherpaLund, hSherpaCluster, hHerwigAngular, totalOrTag='', prefix=''):
    colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3, ROOT.kBlue]
    c = ap.canvas(num_pads=1, batch=True)
    c.hist(hPythia,        linecolor=colours[0], markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')
    c.hist(hSherpaLund,    linecolor=colours[1], markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2", label='Sherpa Lund')
    c.hist(hSherpaCluster, linecolor=colours[2], markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2", label='Sherpa Cluster')
    c.hist(hHerwigAngular, linecolor=colours[3], markercolor=colours[3], fillcolor=colours[3], alpha=0.3, option="P E2", label='Herwig Angular')
    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Events')
    c.ylim(0.001, 1e18)
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{Total}" if totalOrTag=='total' else "#scale[0.85]{Tagged}" )
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.8)
    c.save("{}/alternative_pT_{}.png".format(prefix, totalOrTag))
    c.save("{}/alternative_pT_{}.pdf".format(prefix, totalOrTag))
    c.save("{}/alternative_pT_{}.eps".format(prefix, totalOrTag))

def plotAlternative(wp, hPythia_total, hPythia_tagged, hSherpaLund_total, hSherpaLund_tagged, hSherpaCluster_total, hSherpaCluster_tagged, hHerwigAngular_total, hHerwigAngular_tagged, hHerwigDipole_total, hHerwigDipole_tagged, NNorANN='', prefix=''):

    if NNorANN=='NN':
        # colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3, ROOT.kCyan-6, ROOT.kGreen-2]
        colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]
    if NNorANN=='ANN':
        # colours = [ROOT.kMagenta - 4, ROOT.kOrange-3, ROOT.kYellow+3, ROOT.kCyan-6, ROOT.kGreen-2]
        colours = [ROOT.kMagenta - 4, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]

    bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)

    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()
    ## Pythia
    hBkgRej_Pythia = ROOT.TH1D("", "", len(bins)-1, bins)
    hBkgRej_Pythia.Divide(hPythia_total,hPythia_tagged)
    ## SherpaLund
    hBkgRej_SherpaLund = ROOT.TH1D("", "", len(bins)-1, bins)
    hBkgRej_SherpaLund.Divide(hSherpaLund_total,hSherpaLund_tagged)
    ## SherpaCluster
    hBkgRej_SherpaCluster = ROOT.TH1D("", "", len(bins)-1, bins)
    hBkgRej_SherpaCluster.Divide(hSherpaCluster_total,hSherpaCluster_tagged)
    ## HerwigAngular
    hBkgRej_HerwigAngular = ROOT.TH1D("", "", len(bins)-1, bins)
    hBkgRej_HerwigAngular.Divide(hHerwigAngular_total, hHerwigAngular_tagged)
    ## HerwigDipole
    hBkgRej_HerwigDipole = ROOT.TH1D("", "", len(bins)-1, bins)
    hBkgRej_HerwigDipole.Divide(hHerwigDipole_total, hHerwigDipole_tagged)

    if NNorANN=='NN':
        c.hist(hBkgRej_Pythia,        bins=bins, linecolor=colours[0], markerstyle=20, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')
    if NNorANN=='ANN':
        c.hist(hBkgRej_Pythia,        bins=bins, linecolor=colours[0], markerstyle=4, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')

    c.hist(hBkgRej_SherpaLund,    bins=bins, linecolor=colours[1], markerstyle=22, markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2", label='Sherpa Lund')
    c.hist(hBkgRej_SherpaCluster, bins=bins, linecolor=colours[2], markerstyle=29, markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2", label='Sherpa Cluster')
    c.hist(hBkgRej_HerwigAngular, bins=bins, linecolor=colours[3], markerstyle=21, markercolor=colours[3], fillcolor=colours[3], alpha=0.3, option="P E2", label='Herwig Angular')
    c.hist(hBkgRej_HerwigDipole,  bins=bins, linecolor=colours[4], markerstyle=23, markercolor=colours[4], fillcolor=colours[4], alpha=0.3, option="P E2", label='Herwig Dipole')


    h1 = c.ratio_plot((hBkgRej_Pythia, hBkgRej_Pythia), option="E2")
    h2 = c.ratio_plot((hBkgRej_SherpaLund, hBkgRej_Pythia), markerstyle=22, linecolor=colours[1], markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2")
    h3 = c.ratio_plot((hBkgRej_SherpaCluster, hBkgRej_Pythia), markerstyle=29, linecolor=colours[2], markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2")
    h4 = c.ratio_plot((hBkgRej_HerwigAngular, hBkgRej_Pythia), markerstyle=21, linecolor=colours[3], markercolor=colours[3], fillcolor=colours[3], alpha=0.3, option="P E2")
    h5 = c.ratio_plot((hBkgRej_HerwigDipole, hBkgRej_Pythia), markerstyle=23, linecolor=colours[4], markercolor=colours[4], fillcolor=colours[4], alpha=0.3, option="P E2")

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.ylim(3, 8e3)
    p1.yline(1.0)
    p1.ylim(0.5, 1.5)
    p1.ylabel('Alternative / Pythia')

    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 ("#scale[0.85]{LundNet^{NN}}" if NNorANN=='NN' else "#scale[0.85]{LundNet^{ANN}}"),
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.8)
    workingpointName = ''
    if wp==0.5:
        workingpointName = '0p5'
    else:
        workingpointName = '0p8'
    c.save("{}/alternative_{}_wp{}.png".format(prefix, NNorANN, workingpointName))
    c.save("{}/alternative_{}_wp{}.pdf".format(prefix, NNorANN, workingpointName))
    c.save("{}/alternative_{}_wp{}.eps".format(prefix, NNorANN, workingpointName))

    file = ROOT.TFile.Open("figs_modeldepen_varbin_LundNet_eff50.root", "recreate")
    hBkgRej_Pythia.Write('h_Pythia_bkgrej_vs_pt')
    hBkgRej_SherpaLund.Write('h_SherpaString_bkgrej_vs_pt')
    hBkgRej_SherpaCluster.Write('h_SherpaCluster_bkgrej_vs_pt')
    hBkgRej_HerwigAngular.Write('h_HerwigAngular_bkgrej_vs_pt')
    hBkgRej_HerwigDipole.Write('h_HerwigDipole_bkgrej_vs_pt')
    h1.Write('h_ratio_Pythia2Pythia_bkgrej_vs_pt')
    h2.Write('h_ratio_SherpaString2Pythia_bkgrej_vs_pt')
    h3.Write('h_ratio_SherpaCluster2Pythia_bkgrej_vs_pt')
    h4.Write('h_ratio_HerwigAngular2Pythia_bkgrej_vs_pt')
    h5.Write('h_ratio_HerwigDipole2Pythia_bkgrej_vs_pt')


def plotSigEffModeling(wp,
                      hPythiaNom_tagged, hPythiaNom_total,
                      hPythiaWlong_tagged, hPythiaWlong_total,
                      hPythiaWtrans_tagged, hPythiaWtrans_total,
                      hSherpaLund_tagged, hSherpaLund_total,
                      hHerwigpp_tagged, hHerwigpp_total,
                      NNorANN='', prefix=''):

    if NNorANN=='NN':
        # colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3]
        colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]

    if NNorANN=='ANN':
        # colours = [ROOT.kMagenta - 4, ROOT.kOrange-3, ROOT.kYellow+3]
        colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]

    bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)

    c = ap.canvas(num_pads=1, batch=True)
    # p0, p1 = c.pads()
    ## Pythia
    hSig_PythiaNom = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_PythiaNom.Divide(hPythiaNom_tagged,hPythiaNom_total, 1, 1, "B")
    ## Pythia Wlong
    hSig_PythiaWlong = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_PythiaWlong.Divide(hPythiaWlong_tagged,hPythiaWlong_total, 1, 1, "B")
    ## Pythia Wtrans
    hSig_PythiaWtrans = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_PythiaWtrans.Divide(hPythiaWtrans_tagged,hPythiaWtrans_total, 1, 1, "B")
    ## Sherpa W
    hSig_SherpaLund = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_SherpaLund.Divide(hSherpaLund_tagged,hSherpaLund_total, 1, 1, "B")
    ## Herwigpp W
    hSig_Herwigpp = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_Herwigpp.Divide(hHerwigpp_tagged,hHerwigpp_total, 1, 1, "B")


    ## old code
    # if NNorANN=='NN':
    #     c.hist(hSig_PythiaNom,        bins=bins, linecolor=colours[0], markerstyle=20, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')
    # if NNorANN=='ANN':
    #     c.hist(hSig_PythiaNom,        bins=bins, linecolor=colours[0], markerstyle=4, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')
    #
    # c.hist(hSig_PythiaWlong,    bins=bins, linecolor=colours[1], markerstyle=22, markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2", label='W_{long}')
    # c.hist(hSig_PythiaWtrans, bins=bins, linecolor=colours[2], markerstyle=29, markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2", label='W_{trans}')
    #
    # h1 = c.ratio_plot((hSig_PythiaNom, hSig_PythiaNom), option="E2")
    # h2 = c.ratio_plot((hSig_PythiaWlong, hSig_PythiaNom), markerstyle=22, linecolor=colours[1], markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2")
    # h3 = c.ratio_plot((hSig_PythiaWtrans, hSig_PythiaNom), markerstyle=29, linecolor=colours[2], markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2")
    #
    # c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    # c.ylabel('Signal efficiency #epsilon^{rel}_{sig}')
    # c.ylim(0, 2)
    # p1.yline(1.0)
    # p1.ylim(0.5, 1.5)
    # p1.ylabel('Alternative / Pythia')
    #
    # c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
    #              "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
    #              ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
    #              ("#scale[0.85]{LundNet^{NN}}" if NNorANN=='NN' else "#scale[0.85]{LundNet^{ANN}}"),
    #         ], qualifier='Simulation Preliminary')
    # # c.log()
    # c.legend(xmin=0.7, xmax=0.8)
    # workingpointName = ''
    # if wp==0.5:
    #     workingpointName = '0p5'
    # else:
    #     workingpointName = '0p8'
    # c.save("{}/alternativeSig_{}_wp{}.png".format(prefix, NNorANN, workingpointName))
    # c.save("{}/alternativeSig_{}_wp{}.pdf".format(prefix, NNorANN, workingpointName))
    # c.save("{}/alternativeSig_{}_wp{}.eps".format(prefix, NNorANN, workingpointName))

    hSig_PythiaWlong_PythiaNom = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_PythiaWlong_PythiaNom.Divide(hSig_PythiaWlong, hSig_PythiaNom)
    hSig_PythiaWtrans_PythiaNom = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_PythiaWtrans_PythiaNom.Divide(hSig_PythiaWtrans, hSig_PythiaNom)
    hSig_SherpaLund_PythiaNom = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_SherpaLund_PythiaNom.Divide(hSig_SherpaLund, hSig_PythiaNom)
    hSig_Herwigpp_PythiaNom = ROOT.TH1D("", "", len(bins)-1, bins)
    hSig_Herwigpp_PythiaNom.Divide(hSig_Herwigpp, hSig_PythiaNom)

    c.hist(hSig_PythiaWlong_PythiaNom,  bins=bins, linecolor=colours[0], markerstyle=20, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='W_{long}')
    c.hist(hSig_PythiaWtrans_PythiaNom, bins=bins, linecolor=colours[1], markerstyle=22, markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2", label='W_{trans}')
    c.hist(hSig_SherpaLund_PythiaNom,   bins=bins, linecolor=colours[2], markerstyle=29, markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2", label='Sherpa V+jets')
    c.hist(hSig_Herwigpp_PythiaNom,     bins=bins, linecolor=colours[3], markerstyle=23, markercolor=colours[3], fillcolor=colours[3], alpha=0.3, option="P E2", label='Herwig V+jets')

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('#varepsilon^{rel}_{sig} / #varepsilon^{rel}_{sig} (Nominal)')
    c.ylim(0., 1.8)
    c.yline(1.0)

    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 ("#scale[0.85]{LundNet^{NN}}" if NNorANN=='NN' else "#scale[0.85]{LundNet^{ANN}}"),
            ], qualifier='Simulation Preliminary')

    c.legend(xmin=0.7, xmax=0.8)
    workingpointName = ''
    if wp==0.5:
        workingpointName = '0p5'
    else:
        workingpointName = '0p8'
    c.save("{}/alternativeSig_{}_wp{}.png".format(prefix, NNorANN, workingpointName))
    # c.save("{}/alternativeSig_{}_wp{}.pdf".format(prefix, NNorANN, workingpointName))
    # c.save("{}/alternativeSig_{}_wp{}.eps".format(prefix, NNorANN, workingpointName))

def plotAlternativeSignal(wp, hPythia_total, hPythia_tagged, hSherpaLund_total, hSherpaLund_tagged, hSherpaCluster_total, hSherpaCluster_tagged, NNorANN='', prefix=''):

    if NNorANN=='NN':
        colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3]
    if NNorANN=='ANN':
        colours = [ROOT.kMagenta - 4, ROOT.kOrange-3, ROOT.kYellow+3]

    bins = np.array([200, 300, 400, 500, 600, 750, 950, 1200, 1600, 2000, 2500, 3000], dtype=float)

    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()
    ## Pythia
    hBkgRej_Pythia = ROOT.TH1D("", "", len(bins)-1, bins)
    # hBkgRej_Pythia.Divide(hPythia_total,hPythia_tagged, 1., 1., "B")
    hBkgRej_Pythia.Divide(hPythia_tagged, hPythia_total)
    ## SherpaLund
    hBkgRej_SherpaLund = ROOT.TH1D("", "", len(bins)-1, bins)
    # hBkgRej_SherpaLund.Divide(hSherpaLund_total,hSherpaLund_tagged, 1., 1., "B")
    hBkgRej_SherpaLund.Divide(hSherpaLund_tagged, hSherpaLund_total)
    ## SherpaCluster
    hBkgRej_SherpaCluster = ROOT.TH1D("", "", len(bins)-1, bins)
    # hBkgRej_SherpaCluster.Divide(hSherpaCluster_total,hSherpaCluster_tagged, 1., 1., "B")
    hBkgRej_SherpaCluster.Divide(hSherpaCluster_tagged, hSherpaCluster_total)

    c.hist(hBkgRej_Pythia,        bins=bins, linecolor=colours[0], markerstyle=20, markercolor=colours[0], fillcolor=colours[0], alpha=0.3, option="P E2", label='Pythia')
    c.hist(hBkgRej_SherpaLund,    bins=bins, linecolor=colours[1], markerstyle=22, markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2", label='Sherpa V+jets')
    c.hist(hBkgRej_SherpaCluster, bins=bins, linecolor=colours[2], markerstyle=29, markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2", label='Herwig V+jets')


    c.ratio_plot((hBkgRej_Pythia, hBkgRej_Pythia), option="E2")
    c.ratio_plot((hBkgRej_SherpaLund, hBkgRej_Pythia), markerstyle=22, linecolor=colours[1], markercolor=colours[1], fillcolor=colours[1], alpha=0.3, option="P E2")
    c.ratio_plot((hBkgRej_SherpaCluster, hBkgRej_Pythia), markerstyle=29, linecolor=colours[2], markercolor=colours[2], fillcolor=colours[2], alpha=0.3, option="P E2")

    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Signal efficiency 1/#epsilon^{rel}_{bkg}')
    # c.ylim(1, 1e4)
    p1.yline(1.0)
    p1.ylim(0., 2)
    p1.ylabel('Alternative / Pythia')

    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 ("#scale[0.85]{LundNet^{NN}}" if NNorANN=='NN' else "#scale[0.85]{LundNet^{ANN}}"),
            ], qualifier='Simulation Preliminary')
    # c.log()
    c.legend(xmin=0.7, xmax=0.8)
    c.save("{}/alternative_signal_{}_wp{}.png".format(prefix, NNorANN, wp))
    c.save("{}/alternative_signal_{}_wp{}.pdf".format(prefix, NNorANN, wp))
    c.save("{}/alternative_signal_{}_wp{}.eps".format(prefix, NNorANN, wp))












def get_wp_tag_pol_func(tagger, wp):
    ROOT.gStyle.SetPalette(ROOT.kBird)
    h_pt_nn   = TH2D( "h_pt_nn_other{}".format(tagger.name), "h_pt_nn_other{}".format(tagger.name), 100, 0., 3000, 400,0,1 )

    for pt,nn,weight in zip(tagger.signal["fjet_pt"],tagger.signal["fjet_nnscore"],tagger.signal["fjet_weight_pt"]):
        h_pt_nn.Fill(pt,nn,weight)
    pts, scores = get_eff_score(h_pt_nn,wp)
    scores = scores[11:]
    print("Normal scores cuts->",scores)
    pts = pts [11:]
    gra = TGraph(len(pts), np.array(pts).astype("float"), np.array(scores).astype("float"))
    fitfunc = ROOT.TF1("fit", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 350, 2900) #exponential sigmoid fit (best so far)
    gra.Fit(fitfunc,"R,S")
    c = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)
    h_pt_nn.Draw('colz')
    c.SetRightMargin(0.2)
    c.SetLogz()
    c.SaveAs('2dplot.png')
    gra.Draw()

    p = fitfunc.GetParameters()

    return p

def get_tag_other_MC(tagger, p, wp):
    
    tagger.scores["tag_cut"] = np.vectorize(lambda x:p[0]+p[1]/(p[2]+math.exp(p[3]*(x+p[4]))))(tagger.scores.fjet_pt)
    tagger.signal = tagger.scores[tagger.scores.EventInfo_mcChannelNumber>370000]
    tagger.bg = tagger.scores[tagger.scores.EventInfo_mcChannelNumber<370000]
    tagger.bg_tagged = tagger.bg[tagger.bg.fjet_nnscore > tagger.bg.tag_cut]
    tagger.bg_untagged = tagger.bg[(tagger.bg.fjet_nnscore < tagger.bg.tag_cut) & (tagger.bg.fjet_nnscore>=0)]
    tagger.signal_tagged = tagger.signal[tagger.signal.fjet_nnscore > tagger.signal.tag_cut]

    h_pt_nn   = TH2D( "h_pt_nn_other{}".format(tagger.name), "h_pt_nn_other{}".format(tagger.name), 100, 0., 3000, 400,0,1 )
    h_pt_nn_mass50   = TH2D( "h_pt_nn_mass50_other{}".format(tagger.name), "h_pt_nn_mass50_other{}".format(tagger.name), 100, 0., 3000, 400,0,1 )
    
    for pt,nn,weight in zip(tagger.signal["fjet_pt"],tagger.signal["fjet_nnscore"],tagger.signal["fjet_weight_pt"]):
        h_pt_nn.Fill(pt,nn,weight)


    
    ''' # if I want to include this I should also put pmass50 as input, like was done for p 
    if wp ==0.5:

        for pt,nn,weight in zip(tagger.signalmass50["fjet_pt"], tagger.signalmass50["fjet_nnscore"], tagger.signalmass50["fjet_weight_pt"]):
            h_pt_nn_mass50.Fill(pt,nn,weight)

        ptsmass50, scoresmass50 = get_eff_score_mass(h_pt_nn_mass50, h_pt_nn, wp)
        print("my scoresmass50->", scoresmass50, "len(scoresmass50)->",len(scoresmass50) )
        #ptsmass50, scoresmass50 = get_eff_score(h_pt_nn_mass50,wp)

        scoresmass50 = scoresmass50[6:]
        ptsmass50 = ptsmass50 [6:]
        gramass50 = TGraph(len(ptsmass50), np.array(ptsmass50).astype("float"), np.array(scoresmass50).astype("float"))
        fitfuncmass50 = ROOT.TF1("fitfuncmass50", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700)

        gramass50.Fit(fitfuncmass50,"R,S")
        cmass50 = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)

        gramass50.Draw()

        pmass50 = fitfuncmass50.GetParameters()
        tagger.signalmass50["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.signalmass50.fjet_pt)
        tagger.bgmass50["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.bgmass50.fjet_pt)

        aa= np.array([200,500,1000,2000,3000])
        vfunc = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))
        print("tag_score[:100]", vfunc(aa) )
        
        tagger.bgmass50_tagged = tagger.bgmass50[tagger.bgmass50.fjet_nnscore > tagger.bgmass50.tag_cut_mass]
        tagger.bgmass50_untagged = tagger.bg_untagged
        tagger.signalmass50_tagged = tagger.signalmass50[tagger.signalmass50.fjet_nnscore > tagger.signalmass50.tag_cut_mass]

        #print(len(tagger.bgmass50_tagged))
        #print(len(tagger.signalmass50_tagged))
        #print(len(tagger.bg))

        ############################################### 80% case ######################################################
    if wp ==0.8:
        for pt,nn,weight in zip(tagger.signalmass80["fjet_pt"], tagger.signalmass80["fjet_nnscore"],tagger.signalmass80["fjet_weight_pt"]):
            h_pt_nn_mass50.Fill(pt,nn,weight)

        ptsmass50, scoresmass50 = get_eff_score_mass(h_pt_nn_mass50, h_pt_nn, wp)
        #print("my scoresmass50->", scoresmass50, "len(scoresmass50)->",len(scoresmass50) )

        scoresmass50 = scoresmass50[6:]
        ptsmass50 = ptsmass50 [6:]
        gramass50 = TGraph(len(ptsmass50), np.array(ptsmass50).astype("float"), np.array(scoresmass50).astype("float"))
        fitfuncmass50 = ROOT.TF1("fitfuncmass50", "[p0]+[p1]/([p2]+exp([p3]*(x+[p4])))", 200, 2700)

        gramass50.Fit(fitfuncmass50,"R,S")
        cmass50 = ROOT.TCanvas("myCanvasName{}".format(tagger.name),"The Canvas Title{}",800,600)

        gramass50.Draw()

        pmass50 = fitfuncmass50.GetParameters()
        tagger.signalmass80["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.signalmass80.fjet_pt)
        tagger.bgmass80["tag_cut_mass"] = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))(tagger.bgmass80.fjet_pt)

        aa= np.array([200,500,1000,2000,3000])
        vfunc = np.vectorize(lambda x:pmass50[0]+pmass50[1]/(pmass50[2]+math.exp(pmass50[3]*(x+pmass50[4]))))
        print("tag_score[:100]", vfunc(aa) )

        tagger.bgmass80_tagged = tagger.bgmass80[tagger.bgmass80.fjet_nnscore > tagger.bgmass80.tag_cut_mass]
        tagger.bgmass80_untagged = tagger.bg_untagged
        tagger.signalmass80_tagged = tagger.signalmass80[tagger.signalmass80.fjet_nnscore > tagger.signalmass80.tag_cut_mass]
    '''




def pt_bgrej_otherMC(taggers,weight="chris_weight", prefix='', wp=0.5):

    colours = [ROOT.kMagenta - 4, ROOT.kAzure + 7, ROOT.kTeal, ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink,  ROOT.kPink+3]

    # bins = np.linspace(250, 3250, 14 + 1, endpoint=True)
    bins = np.linspace(350, 3150, 15+1) ## use same binning as Kevin
    kevin_results = np.load('Nominal_metrics.npz')

    total_binned_br_50 = kevin_results['total_binned_br_50']
    total_binned_br_80 = kevin_results['total_binned_br_80']
    ## add kevin results
    c = ap.canvas(num_pads=1, batch=True)
    count = 0
    for t in taggers:
        if t=="LundNet_class":  label = "LundNet^{NN}"
        if t=="LundNet"      :  label = "LundNet^{ANN}"

        if t=="HerwigAngular":  label = "HerwigAngular"
        if t=="HerwigDipole"      :  label = "HerwigDipole"
        if t=="SherpaCluster"      :  label = "SherpaCluster"

        
        ## Get total background
        h_bg_total = c.hist(np.array(taggers[t].bg["fjet_pt"]), weights=np.array(taggers[t].bg[weight]), bins=bins, display=False)
        ## Get tagged background
        h_bg = c.hist(np.array(taggers[t].bg_tagged["fjet_pt"]), weights=np.array(taggers[t].bg_tagged[weight]), bins=bins, display=False)
        ## Calculate bkg rejection (1/epsilon bkg = total bkg / tagged bkg)
        hratio = ROOT.TH1D("", "", len(bins)-1, bins)
        hratio.Divide(h_bg_total,h_bg, 1., 1., "B")
        if t=="LundNet_class":  markerstyle,linestyle  = 20,1
        if t=="LundNet"      :  markerstyle,linestyle =4,9

        if t=="HerwigAngular":  markerstyle,linestyle  = 20,2
        if t=="HerwigDipole"      :  markerstyle,linestyle =20,3
        if t=="SherpaCluster"      :  markerstyle,linestyle =20,3
        
        c.hist(hratio, option='P E2', bins=bins, label=label, linestyle=linestyle, markerstyle=markerstyle, markercolor=colours[count], linecolor=colours[count], fillcolor=colours[count], alpha=0.3)

        # c.ratio_plot((h_bg_total,      h_bg), option='P E2', bins=bins, label=label, linecolor=colours[count])
        count +=1


    c.xlabel('Large-#it{R} jet p_{T} [GeV]')
    c.ylabel('Background rejection 1/#epsilon^{rel}_{bkg}')
    c.text(["#sqrt{s} = 13 TeV, #it{top} tagging",
                 "#scale[0.85]{anti-k_{t} R=1.0 UFO Soft-Drop CS+SK jets}",
                 ("#scale[0.85]{#varepsilon^{rel}_{sig} = 50%}" if wp==0.5 else "#scale[0.85]{#varepsilon^{rel}_{sig} = 80%}"),
                 # "#scale[0.85]{Cut on m_{J} from 3-var tagger}",
            ], qualifier='Simulation Preliminary')
    c.log()
    c.legend(xmin=0.7, xmax=0.9)
    c.ylim(1, 1e5)
    # c.ylim(25, 225)
    c.save("{}/pt_bgrej_otherMC.png".format(prefix))
    c.save("{}/pt_bgrej_otherMC.pdf".format(prefix))
    c.save("{}/pt_bgrej_otherMC.eps".format(prefix))



