import argparse
import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from torch_geometric.data import DataListLoader, DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *

import gc
print("Libraries loaded!")


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_file = config['data']['path_to_trainfiles']
    # path_to_file = '/data/jmsardain/LJPTagger/FewSplittings/15_train_5_test_9splittings/user.rvinasco.31*000003.tree.root_train.root'
    files = glob.glob(path_to_file)
    #files = glob.glob("/sps/atlas/k/khandoga/MySamplesS40/user.rvinasco.27045978._000004.tree.root_train.root")
    #files = files[:1]
    jet_type = "Akt10UFOJet" #UFO jets
    save_trained_model = True
    intreename = "AnalysisTree"

    print("Training tagger on files", len(files))
    t_start = time.time()


    dataset = []
    for file in files:
        print("Loading file",file)
        with uproot.open(file) as infile:
            tree = infile[intreename]
            
            dsids_test = tree["dsid"].array(library="np")
            if dsids_test[0] == 364702 :
                continue
            dsids = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array(library="ak")) )

            parent1 = ak.flatten(tree["jetLundIDParent1"].array(library="ak")) 
            parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak")) 
            jet_ms = ak.to_numpy(ak.flatten(tree["LRJ_mass"].array(library="ak")))
            all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak")) 
            all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak")) 
            all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak")) 
            N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak")) )
            jet_pts = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )

            labels = dsids

            #flat_weights = GetPtWeight( dsids, jet_pts, 1)
            flat_weights = GetPtWeight_2( dsids, jet_pts, 5)
            
            #dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels, all_Ntrk, jet_pts, jet_ms)
            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts , jet_ms)

            gc.collect()

    print("Dataset created!")
    delta_t_fileax = time.time() - t_start
    print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))


    #dataset = create_train_dataset_fulld_new(all_lund_zs[s_evt:events], all_lund_kts[s_evt:events], all_lund_drs[s_evt:events], parent1[s_evt:events], parent2[s_evt:events], labels[s_evt:events])



    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']


    dataset= shuffle(dataset,random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

    delta_t_fileax = time.time() - t_start
    print("Splitted datasets in {:.4f} seconds.".format(delta_t_fileax))


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))

    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()

    flag = config['retrain']['flag']
    path_to_ckpt = config['retrain']['path_to_ckpt']

    if flag==True:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))

    #device = torch.device('cpu')
    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=4*learning_rate)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name']
    path_to_save = config['data']['path_to_save']
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    metrics_filename = path_to_save+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    for epoch in range(n_epochs):
        train_loss.append(train_clas(train_loader, model, device, optimizer, optimizer2, optimizer3, epoch))
        val_loss.append(my_test(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))
        if (save_every_epoch):
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
        elif epoch == n_epochs-1:
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")

    metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss})
    metrics.to_csv(metrics_filename, index = False)
    return


if __name__ == "__main__":
    main()
