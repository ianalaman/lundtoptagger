import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import yaml
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from scipy.stats import entropy
import math
import networkx as nx
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from ..GNN_model_weight.models import mdn_loss, mdn_loss_new

# weights_file = uproot.open("/data/jmsardain/LJPTagger/FullSplittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FullSplittings/5_Signal_and_BKG/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FewSplittings/15_train_5_test_9splittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/Samples/MakeSamples/75_train_25_test_9splittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FewSplittings/15_train_5_test_2splittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FewSplittings/15_train_5_test_5splittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FewSplittings/15_train_5_test_3splittings/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/Samples/MakeSamples/10_train_5_test_full/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/Samples/MakeSamples/15_train_5_test_full/flat_weights.root")
# weights_file = uproot.open("/data/jmsardain/LJPTagger/Samples/MakeSamples/20_train_5_test_full/flat_weights.root")
## used at the end for W tagging :
# weights_file = uproot.open("/data/jmsardain/LJPTagger/Samples/MakeSamples/5_train_5_test_full/flat_weights.root")

## for top tagging:
weights_file = uproot.open("/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/flat_weights.root")

flatweights_bg = weights_file["bg_inv"].to_numpy()
flatweights_sig = weights_file["h_sig_inv"].to_numpy()



def GetPtWeight( dsid , pt, SF):

    length_sig = len(flatweights_bg[0])
    length_bkg = len(flatweights_bg[0])
    scale_factor = 14.475606
    weight_out = []
    for i in range ( 0,len(dsid) ):
        pt_bin = int( (pt[i]/3000)*length_sig )
        if pt_bin==length_sig :
            pt_bin = length_sig-1
        if dsid[i] < 370000 :
            weight_out.append( (flatweights_bg[0][pt_bin]*scale_factor)*10**4 ) ## used for W tagging
            # weight_out.append( (flatweights_bg[0][pt_bin])*10**(-1) )
            # weight_out.append( (flatweights_bg[0][pt_bin]*scale_factor)*10**(-1) )
        if dsid[i] > 370000 :
            weight_out.append( (flatweights_sig[0][pt_bin])*10**4 ) ## used for W tagging
            # weight_out.append( (flatweights_sig[0][pt_bin])*10**(-1) )
    return np.array(weight_out)


def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def create_dataset_train(graphs, z, k, d, edge1, edge2, weight, labelss, Ntracks, jet_pts, jet_ms):

    for i in range(len(z)):
        k[i] = np.where(k[i] == 0, 1e-5, k[i])
        z[i] = np.where(z[i] == 0, 1e-5, z[i])
        d[i] = np.where(d[i] == 0, 1e-5, d[i])

        z[i] = np.log(1/z[i])
        k[i] = np.log(k[i])
        d[i] = np.log(1/d[i])

        graph = nx.DiGraph()
        nodes_to_remove = []
        for j in range(len(edge1[i])):
            graph.add_edge(edge1[i][j], edge2[i][j])

        # for edge in graph.edges():
        #     if edge[0] in remaining_nodes and edge[1] in remaining_nodes:
        #         new_graph.add_edge(edge[0], edge[1])

        k_values = [k[i][node] for node in graph.nodes()]
        d_values = [d[i][node] for node in graph.nodes()]
        z_values = [z[i][node] for node in graph.nodes()]

        for node in graph.nodes():
            if k[i][node] <= 2. or d[i][node] <0:
                nodes_to_remove.append(node)


        graph.remove_nodes_from(nodes_to_remove)
        remaining_nodes = list(graph.nodes())

        # new_graph = nx.DiGraph()
        # new_graph.add_nodes_from(remaining_nodes)

        # for edge in graph.edges():
        #         if edge[0] in remaining_nodes and edge[1] in remaining_nodes:
        #             new_graph.add_edge(edge[0], edge[1])

        graph.remove_nodes_from(list(nx.isolates(graph)))
        if (graph.number_of_nodes() < 2):
            continue
        if (graph.number_of_edges() < 1):
            continue

        largest_component = max(nx.strongly_connected_components(graph), key=len)
        # print(largest_component)
        edges_to_remove = []
        for edge in graph.edges():
            if edge[0] not in largest_component or edge[1] not in largest_component:
                edges_to_remove.append(edge)
        graph.remove_edges_from(edges_to_remove)

        edge_out = []
        edge_in  = []
        for edge in graph.edges:
            edge_out.append(edge[0])
            edge_in.append(edge[1])

        edge_index = torch.tensor(np.array([edge_out, edge_in]) , dtype=torch.long)
        all_labels = torch.unique(edge_index)

        # Create a mapping between original and new labels
        label_mapping = {}
        new_label = 0
        for label in all_labels:
            if label.item() not in label_mapping:
                label_mapping[label.item()] = new_label
                new_label += 1

        # Transform the original edge tensor with new labels
        transformed_edge_index = torch.tensor([[label_mapping[label.item()] for label in edge] for edge in edge_index])

        k_values = [k[i][node] for node in graph.nodes()]
        d_values = [d[i][node] for node in graph.nodes()]
        z_values = [z[i][node] for node in graph.nodes()]

        vec = []
        vec.append(np.array([d_values, z_values, k_values]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)


        # Ensure all tensors have the same size
        # print(transformed_edge_index)
        # print(vec)

        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(transformed_edge_index).detach(),
                           Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           weights =torch.tensor(weight[i], dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(labelss[i], dtype=torch.float).detach() ))
    return graphs

def create_dataset_test(graphs, z, k, d, edge1, edge2, labelss, Ntracks, jet_pts, jet_ms):

    for i in range(len(z)):
        k[i] = np.where(k[i] == 0, 1e-5, k[i])
        z[i] = np.where(z[i] == 0, 1e-5, z[i])
        d[i] = np.where(d[i] == 0, 1e-5, d[i])

        z[i] = np.log(1/z[i])
        k[i] = np.log(k[i])
        d[i] = np.log(1/d[i])

        graph = nx.DiGraph()
        nodes_to_remove = []
        for j in range(len(edge1[i])):
            graph.add_edge(edge1[i][j], edge2[i][j])

        # for edge in graph.edges():
        #     if edge[0] in remaining_nodes and edge[1] in remaining_nodes:
        #         new_graph.add_edge(edge[0], edge[1])

        k_values = [k[i][node] for node in graph.nodes()]
        d_values = [d[i][node] for node in graph.nodes()]
        z_values = [z[i][node] for node in graph.nodes()]

        for node in graph.nodes():
            if k[i][node] <= 2. or d[i][node] <0:
                nodes_to_remove.append(node)


        graph.remove_nodes_from(nodes_to_remove)
        remaining_nodes = list(graph.nodes())

        # new_graph = nx.DiGraph()
        # new_graph.add_nodes_from(remaining_nodes)

        # for edge in graph.edges():
        #         if edge[0] in remaining_nodes and edge[1] in remaining_nodes:
        #             new_graph.add_edge(edge[0], edge[1])

        graph.remove_nodes_from(list(nx.isolates(graph)))
        if (graph.number_of_nodes() < 2):
            continue
        if (graph.number_of_edges() < 1):
            continue

        largest_component = max(nx.strongly_connected_components(graph), key=len)
        # print(largest_component)
        edges_to_remove = []
        for edge in graph.edges():
            if edge[0] not in largest_component or edge[1] not in largest_component:
                edges_to_remove.append(edge)
        graph.remove_edges_from(edges_to_remove)

        edge_out = []
        edge_in  = []
        for edge in graph.edges:
            edge_out.append(edge[0])
            edge_in.append(edge[1])

        edge_index = torch.tensor(np.array([edge_out, edge_in]) , dtype=torch.long)
        all_labels = torch.unique(edge_index)

        # Create a mapping between original and new labels
        label_mapping = {}
        new_label = 0
        for label in all_labels:
            if label.item() not in label_mapping:
                label_mapping[label.item()] = new_label
                new_label += 1

        # Transform the original edge tensor with new labels
        transformed_edge_index = torch.tensor([[label_mapping[label.item()] for label in edge] for edge in edge_index])

        k_values = [k[i][node] for node in graph.nodes()]
        d_values = [d[i][node] for node in graph.nodes()]
        z_values = [z[i][node] for node in graph.nodes()]

        vec = []
        vec.append(np.array([d_values, z_values, k_values]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)


        # Ensure all tensors have the same size
        # print(transformed_edge_index)
        # print(vec)

        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(transformed_edge_index).detach(),
                           Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(labelss[i], dtype=torch.float).detach() ))
    return graphs


def create_train_dataset_fulld_new_Ntrk_pt_weight_file(graphs, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms):

    for i in range(len(z)):
        # if i>2: continue
        k[i][k[i] == 0] = 1e-10
        d[i][d[i] == 0] = 1e-10

        z[i] = np.log(1/z[i])
        k[i] = np.log(k[i])
        d[i] = np.log(1/d[i])

        ## cut lund plane where effects are the same
        # indices = np.argwhere(k[i]<2.7)

        # z[i] = np.delete(z[i], indices)
        # k[i] = np.delete(k[i], indices)
        # d[i] = np.delete(d[i], indices)
        # edge1[i] = np.delete(edge1[i], indices)
        # edge2[i] = np.delete(edge2[i], indices)

        mean_z, std_z = 2.523076295852661, 5.264721870422363
        mean_dr, std_dr = 11.381295204162598, 13.63073444366455
        mean_kt, std_kt = -10.042571067810059, 15.398056030273438
        mean_ntrks, std_ntrks = 33.35614197897151, 12.064001858459823


        z[i] = (z[i] - mean_z) / std_z
        k[i] = (k[i] - mean_kt) / std_kt
        d[i] = (d[i] - mean_dr) / std_dr
        Ntracks = (Ntracks - mean_ntrks) / std_ntrks


        if (len(edge1[i])== 0) or (len(edge2[i])== 0) or (len(k[i])== 0) or (len(z[i])== 0) or (len(d[i])== 0):
            continue
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)

        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        # print(edge)
        # print(vec)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(edge).detach(),
                           Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           weights =torch.tensor(weight[i], dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label[i], dtype=torch.float).detach() ))
    return graphs

def create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(graphs, z, k, d, edge1, edge2, label, Ntracks, jet_pts, jet_ms):


    for i in range(len(z)):
        # if i > 10: continue
#         z[i] = np.nan_to_num(z[i], nan=1e-10, posinf=1e-10, neginf=-1e-10)
#         k[i] = np.nan_to_num(k[i], nan=1e-10, posinf=1e-10, neginf=-1e-10)
#         d[i] = np.nan_to_num(d[i], nan=1e-10, posinf=1e-10, neginf=-1e-10)

#         z[i][z[i] == 0] = 1e-10
        k[i][k[i] == 0] = 1e-10
        d[i][d[i] == 0] = 1e-10


        z[i] = np.log(1/z[i])
        k[i] = np.log(k[i])
        d[i] = np.log(1/d[i])


        mean_z, std_z = 2.523076295852661, 5.264721870422363
        mean_dr, std_dr = 11.381295204162598, 13.63073444366455
        mean_kt, std_kt = -10.042571067810059, 15.398056030273438
        mean_ntrks, std_ntrks = 33.35614197897151, 12.064001858459823


        z[i] = (z[i] - mean_z) / std_z
        k[i] = (k[i] - mean_kt) / std_kt
        d[i] = (d[i] - mean_dr) / std_dr
        Ntracks = (Ntracks - mean_ntrks) / std_ntrks



        if (len(edge1[i])== 0) or (len(edge2[i])== 0):
            continue
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)

        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(edge).detach(),
                           Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label[i], dtype=torch.float).detach() ))
    return graphs



def train(loader, model, device, optimizer):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        l2_lambda = 0.01 # regularization strength
        for param in model.parameters():
            if param.dim() > 1:
                # apply L2 regularization to all parameters except biases
                loss = loss + l2_lambda * nn.MSELoss()(param, torch.zeros_like(param))

        loss.backward()

        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def train_adversary(loader, clsf, adv, optimizer, device, loss_parameter):
    clsf.eval()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0

    for data in loader:
        batch_counter+=1
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1)) ## add weights

        mask_bkg = new_y.lt(0.5)
        optimizer.zero_grad()
        cl_out = clsf(cl_data)
        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)

        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)

        pi, sigma, mu = adv(adv_inp)
        print(pi)

        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)),new_w[mask_bkg])
        loss2.backward()
        loss = loss1 + loss_parameter*loss2

        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer.step()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)

######
def train_adversary_2(loader, clsf, adv, optimizer, device, loss_parameter, loss_weights):
    clsf.eval()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0

    for data in loader:
        batch_counter+=1

        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1)) ## add weights

        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)

        mask_bkg = new_y.lt(0.5)
        optimizer.zero_grad()
        cl_out = clsf(cl_data)
        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)


        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)

        pi, sigma, mu = adv(adv_inp)

        #print("batch_counter",batch_counter)
        loss2 = loss_weights[1] * mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])

        l2_lambda = 0.01  # L2 regularization parameter


        # Apply L2 regularization to the combined loss
        l2_regularization = 0
        for param in clsf.parameters():
            l2_regularization += nn.MSELoss()(param, torch.zeros_like(param))
        for param in adv.parameters():
            l2_regularization += nn.MSELoss()(param, torch.zeros_like(param))



        # loss = loss_weights[0] * loss1 + loss_weights[1]*loss_parameter*loss2
        loss = loss_weights[0] * loss1 + loss_weights[1]*loss_parameter*loss2 + l2_lambda * l2_regularization

        loss2.backward()


        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer.step()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)




def train_combined(loader, clsf, adv, optimizer_cl, optimizer_adv, device, loss_parameter,loss_weights):
    clsf.train()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0
    jsd_total = 0
 #   print ("batches in the dataset:", len(loader))
    for data in loader:
        batch_counter+=1
 #       print ("processing batch number",batch_counter)
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1))

        mask_bkg = new_y.lt(0.5)
        optimizer_cl.zero_grad()
        optimizer_adv.zero_grad()
        cl_out = clsf(cl_data)

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0

        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        pi, sigma, mu = adv(adv_inp)

        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)),new_w[mask_bkg])
        #loss2 = mdn_loss_new(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)),new_w[mask_bkg])

        loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2

        loss.backward()
        loss_clsf += cl_data.num_graphs * loss1.item() * loss_weights[0]
        loss_adv += cl_data.num_graphs * loss2.item() * loss_weights[1]
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step()
        optimizer_adv.step()
#        mask_tag = cl_out.lt(0.5)
#        mask_untag = cl_out.ge(0.5)
#        p, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
#        f, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)
#        jsd = JSD(p,f)
#        jsd_total +=jsd
#        print ("jsd",jsd)

    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


def train_combined_2(loader, clsf, adv, optimizer_cl, optimizer_adv, device, loss_parameter, loss_weights):
    clsf.train()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0
    jsd_total = 0

    for data in loader:
        batch_counter+=1
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1))

        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)

        mask_bkg = new_y.lt(0.5)
        optimizer_cl.zero_grad()
        optimizer_adv.zero_grad()
        cl_out = clsf(cl_data)

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0

        #adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)
        pi, sigma, mu = adv(adv_inp)

        #print("pi[:2]---------------------------------------")
        #print(pi[:2])
        '''
        print("---------------------------------------")
        #print( torch.reshape(new_mass[mask_bkg], (len(new_pt[mask_bkg]),1) )   )
        print("---------------------------------------")
        print("mu size->", mu.size(), "   pi size->",pi.size() ,"   sigma size->", sigma.size()  )
        print("mu---------------------------------------")
        print(mu[:2])
        print("pi---------------------------------------")
        print(pi[:2])
        print("sigma---------------------------------------")
        print(sigma[:2])
        print("---------------------------------------")
        '''
        #print(len(loader.dataset))


        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)
        #loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)),new_w[mask_bkg])
        #loss2 = mdn_loss_new(pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])
        loss2 = mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])

        l2_lambda_clf = 0.01  # L2 regularization parameter
        l2_lambda_adv = 0.01

        # Apply L2 regularization to the combined loss
        l2_regularization_clf = 0
        l2_regularization_adv = 0
        for param in clsf.parameters():
            l2_regularization_clf += nn.MSELoss()(param, torch.zeros_like(param))
        for param in adv.parameters():
            l2_regularization_adv += nn.MSELoss()(param, torch.zeros_like(param))


        # loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2
        loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2  + l2_lambda_clf * l2_regularization_clf  + l2_lambda_adv * l2_regularization_adv
        loss.backward()

        loss_clsf += loss_weights[0] * cl_data.num_graphs * loss1.item()
        loss_adv += loss_weights[1] * cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step()
        optimizer_adv.step()

    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)





def test_combined(loader, clsf, adv, device, loss_parameter, loss_weights ):
    clsf.eval()
    adv.eval()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    for data in loader:
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        cl_out = clsf(cl_data)
        new_w = torch.reshape(cl_data.weights, (int(list(cl_data.weights.shape)[0]),1))

        new_pt = torch.reshape(cl_data.pt, (int(list(cl_data.pt.shape)[0]),1) )
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        new_pt = torch.log(new_pt)

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0

        loss1 = F.binary_cross_entropy(cl_out, new_y, weight = new_w)

        #adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(cl_data.pt[mask_bkg], (len(cl_data.pt[mask_bkg]), 1))), 1)
        adv_inp = torch.cat( (torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]),1)) , torch.reshape(new_pt[mask_bkg], (len(new_pt[mask_bkg]),1) ))  ,1)

        pi, sigma, mu = adv(adv_inp)

        loss2 = mdn_loss_new(device, pi, sigma, mu, torch.reshape(new_mass[mask_bkg], (len(new_mass[mask_bkg]),1) ) , new_w[mask_bkg])

        loss = loss_weights[0] * loss1 + loss_weights[1] * loss_parameter*loss2
        loss_clsf += loss_weights[0] * cl_data.num_graphs * loss1.item()
        loss_adv += loss_weights[1] * cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        # print("loss_adv->",loss_adv)
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader, model, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y, weight=new_w)
        loss_all += data.num_graphs * loss.item()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader, model, device):
    model.eval()
    total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        # print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]


def aux_metrics(loader, clsf, adv, device, MASSBINS):
    clsf.eval()
    adv.eval()
    counter = 0
    bkg_tagged = 0
    bkg_total = 0
    jsd_total = 0
    nans = 0
    jsd_counter = 0
    mass_tagged = np.array([])
    mass_untagged = np.array([])
    for data in loader:
        cl_data = data.to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        #    print ("true labels",new_y)
        new_mass = torch.reshape(cl_data.mass, (int(list(cl_data.mass.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
    #    print ("mask_bkg",mask_bkg)
        cl_out = clsf(cl_data)
   #     print ("preds",cl_out)
        mask_tag = cl_out.lt(0.5)
        mask_untag = cl_out.ge(0.5)
 #       print ("mask_tag",mask_tag)
 #       print ("mask_untag",mask_untag)
 #       print ("sum bg",torch.count_nonzero(mask_bkg))
 #       print ("sum tag bg",torch.count_nonzero(mask_tag))
        bkg_tagged+=torch.count_nonzero(mask_untag&mask_bkg)
        bkg_total+=torch.count_nonzero(mask_bkg)
#        print ("adv_data.x",adv_data.y)
#        print ("mass tag",adv_data.y[mask_bkg&mask_tag])
#        print ("mass untag",adv_data.y[mask_bkg&mask_untag])

        #p, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
        #f, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)

        p, _ = np.histogram(np.array(new_mass[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
        f, _ = np.histogram(np.array(new_mass[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)

        jsd = JSD(p,f)
        if math.isnan(jsd):
            nans+=1
        else:
            jsd_total +=jsd
            jsd_counter+=1
  #      print ("jsd",jsd)
    if bkg_tagged:
        eff = bkg_total/bkg_tagged
    else:
        eff = bkg_total*0

    if jsd_counter:
        jsd_total = jsd_total/jsd_counter
    else:
        jsd_total = 0

    return float(eff.cpu()), jsd_total


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
