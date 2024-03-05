#!/usr/bin/env python

from utils_plots import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import ROOT
from root_numpy import fill_hist  as fh
import warnings
warnings.filterwarnings('ignore')
import os



for variation in range(1):
    taggers = {}
    tagger_files = {}
    if variation==0:
        outdir='/home/jmsardain/LJPTagger/ljptagger/Plotting_Top/PublicPlots'

        # tagger_files["LundNet_class"]       = '/data/jmsardain/LJPTagger/Models/TopTagger/Scores/treeTopTagging_LundNetEpoch30_Mar12024.root'
        # tagger_files["LundNet_class"]       = '/data/jmsardain/LJPTagger/Models/TopTagger/Scores/treeTopTagging_LundNetEpoch60_Mar22024.root'
        # tagger_files["LundNet_class"]       = '/data/jmsardain/LJPTagger/Models/TopTagger/Scores/treeTopTagging_LundNetwithSFWpoch40_Mar42024.root'
        # tagger_files["LundNet_class"]       = '/data/jmsardain/LJPTagger/Models/TopTagger/Scores/treeTopTagging_LundNetSFBiggerEpoch28_Mar42024.root'
        tagger_files["LundNet_class"]       = '/data/jmsardain/LJPTagger/Models/TopTagger/Scores/treeTopTagging_LundNetwithSFe4LR1e-5_Mar52024.root'
        # tagger_files["LundNet_class"]       = '/home/jmsardain/LJPTagger/ljptagger/Plotting/FromRafael/meeting_plots/LundNet_LOGNormalized_4096_e40_.root'


    try:
        os.system("mkdir {}".format(outdir))
    except ImportError:
        print("{} already exists".format(prefix))
    pass

    working_point = 0.5
    for t in tagger_files:
        taggers[t] = tagger_scores(t,tagger_files[t], working_point)


    for t in taggers:
        if taggers[t].name == "3var":
            continue
        get_wp_tag(taggers[t],working_point, prefix=outdir)  ## smooth function


    make_rocs(taggers,prefix=outdir)
    # # ## Make plot vs mu
    # bgrej_mu(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # bgrej_npv(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # pt_spectrum(taggers,weight="fjet_weight_pt", prefix=outdir)
    # pt_sigeff(taggers,weight="fjet_weight_pt", prefix=outdir)
    # # ## Make plot background rejection vs pT
    make_efficiencies_all(taggers, prefix=outdir) ## fixed
    # pt_bgrej_all(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # make_efficiencies_pt_all(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed

    pt_bgrej(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # pt_bgrej_mass(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # # # # # #
    # # # # # # ## Make mass sculpting plots (inclusive and in bins of pT)
    # mass_sculpting(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 300,  650, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 650, 1000, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 1000, 3000, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed


    # # # for t in taggers:
    # #     mass_sculpting(taggers[t], t, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t,  200,  500, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t,  500, 1000, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t, 1000, 3000, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    #
    # ## Make background rejection vs signal efficiency plots (inclusive and in bins of pT)
    make_efficiencies_3var(taggers, prefix=outdir) ## fixed
    # make_efficiencies_3var_massCut(taggers, prefix=outdir) ## fixed
    # make_efficiencies_pt(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
