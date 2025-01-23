import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, DenoisingUNet, p_losses, sample, apply_noise
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=800, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=128, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=64, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=4, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=4, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=800, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=False, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=False, help="Flag to enable/disable denoiser training (default: enabled)")

#parser.add_argument('--train_denoiser_denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition-decode', type=int, default=64, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")
parser.add_argument('--dim-condition-denoise', type=int, default=64, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=10, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)

# # initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


########################  TRAINING  ################################
# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, args.n_condition, args.dim_condition_decode).to(device)

optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train=0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld  = autoencoder.loss_function(data, data.stats)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            loss, recon, kld = autoencoder.loss_function(data, data.stats)
            val_loss_all_recon += recon.item()
            val_loss_all_kld += kld.item()
            val_loss_all += loss.item()
            cnt_val+=1
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
            
        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])


autoencoder.eval()

# # define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition_denoise).to(device)
optimizer = torch.optim.AdamW(denoise_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber") #data.stats <- cond
            loss.backward() 
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])


denoise_model.eval()

##########################################################

##################  GRAPHS GENERATION  ###################
from networkx.algorithms import community

def extract_graph_properties(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    nb_of_nodes = G.number_of_nodes()
    nb_of_edges = G.number_of_edges()
    average_degree = 2 * nb_of_edges / nb_of_nodes if nb_of_nodes > 0 else 0
    nb_of_triangles = sum(nx.triangles(G).values()) // 3
    global_clustering_coefficient = nx.transitivity(G)
    maximum_k_core = max(nx.core_number(G).values()) if nb_of_nodes > 0 else 0
    nb_of_communities = len(list(community.greedy_modularity_communities(G)))

    return [nb_of_nodes,nb_of_edges,average_degree,nb_of_triangles,global_clustering_coefficient,maximum_k_core,nb_of_communities]

property_stats = {
    "nb_of_nodes": (30.35, 11.6991610968638),
    "nb_of_edges": (222.303, 234.317549289218),
    "average_degree": (12.816147624135000, 10.236861265941500),
    "nb_of_triangles": (1370.722, 2773.57840341244),
    "global_clustering_coefficient": (0.504197527112439, 0.325987972390425),
    "maximum_k_core": (11.393, 10.0706488851995),
    "nb_of_communities": (3.442, 10.0706488851995),
}

def normalize_property(value, prop_name):
    mean, std = property_stats[prop_name]
    return (value - mean) / std

def normalize_properties(properties):
    keys = ["nb_of_nodes","nb_of_edges","average_degree","nb_of_triangles","global_clustering_coefficient","maximum_k_core","nb_of_communities"]
    return [normalize_property(value, key) for value, key in zip(properties, keys)]


# with distances and our mae's
denoise_model.eval()
if True:
    total_absolute_errors = [0] * 7
    total_graphs = 0

    with open("output_THE_LAST_OF_US.csv", "w", newline="") as csvfile:
        print("processing graphs \n")
        writer = csv.writer(csvfile)
        writer.writerow(["graph_id", "edge_list"])  # Header

        for k, data in enumerate(tqdm(test_loader, desc="Processing test set")):
            data = data.to(device)
            ground_truth_properties = data.stats.detach().cpu().numpy()
            graph_ids = data.filename

            bs = ground_truth_properties.shape[0]
            total_graphs += bs

            all_best_graphs = []  
            all_best_properties = []  

            ground_truth_normalized = [normalize_properties(gt) for gt in ground_truth_properties]

            for _ in range(300):
                samples = sample(
                    denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs
                )
                x_sample = samples[-1]
                adj_batch = autoencoder.decode_mu(x_sample, cond=data.stats)

                for i in range(bs):
                    adj_matrix = adj_batch[i, :, :].detach().cpu().numpy()
                    edges = construct_nx_from_adj(adj_matrix).edges()
                    generated_properties = extract_graph_properties(edges)

                    generated_normalized = normalize_properties(generated_properties)

                    if len(all_best_graphs) <= i:
                        all_best_graphs.append(edges)
                        all_best_properties.append(generated_normalized)
                    else:
                        gt_normalized = ground_truth_normalized[i]

                        weights = [1, 1, 1, 1, 1, 1, 1]
                        distance = sum(w * abs(g - gt) for w, g, gt in zip(weights, generated_normalized, gt_normalized))
                        best_distance = sum(w* abs(bp - gt) for w, bp, gt in zip(weights, all_best_properties[i], gt_normalized))

                        if distance < best_distance:
                            all_best_graphs[i] = edges
                            all_best_properties[i] = generated_normalized

            for i in range(bs):
                graph_id = graph_ids[i]
                best_edges = all_best_graphs[i]
                edge_list_text = ", ".join([f"({u}, {v})" for u, v in best_edges])
                writer.writerow([graph_id, edge_list_text])

                for j in range(7):
                    total_absolute_errors[j] += abs(all_best_properties[i][j] - ground_truth_normalized[i][j])

    total_MAE = sum(total_absolute_errors) / (total_graphs * 7)

    print("Optimized graphs have been saved to 'output the last of us.csv'.")
    print("Total Final MAE: ", total_MAE)
    print("Errors for each property: ", [total_absolute_errors[i]/total_graphs for i in range(7)])

    print(np.mean([total_absolute_errors[i]/total_graphs for i in range(7)]))

################################################################

################# calculate variance properties ####################
if False:
    nb_generations=50

    denoise_model.eval()

    def calculate_variance_per_property(values):
        return np.var(values, axis=0)

    output_file = "graph_variance_notremodel.csv"

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_id", "nb_of_nodes_variance", "nb_of_edges_variance", 
                        "average_degree_variance", "nb_of_triangles_variance", 
                        "global_clustering_coefficient_variance", "maximum_k_core_variance", 
                        "nb_of_communities_variance"])
        
        for k, data in enumerate(tqdm(test_loader, desc="Processing test set")):
            data = data.to(device)
            graph_ids = data.filename
            bs = len(graph_ids)

            all_properties = {graph_id: [] for graph_id in graph_ids}

            for _ in range(nb_generations):
                samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, 
                                timesteps=args.timesteps, betas=betas, batch_size=bs)
                x_sample = samples[-1]
                adj_batch = autoencoder.decode_mu(x_sample, data.stats)

                for i in range(bs):
                    adj_matrix = adj_batch[i, :, :].detach().cpu().numpy()
                    edges = construct_nx_from_adj(adj_matrix).edges()
                    
                    properties = extract_graph_properties(edges)
                    normalized_properties = normalize_properties(properties)

                    graph_id = graph_ids[i]
                    all_properties[graph_id].append(normalized_properties)

            for graph_id, properties_list in all_properties.items():
                properties_array = np.array(properties_list)
                variances = calculate_variance_per_property(properties_array)
                writer.writerow([graph_id] + list(variances))

    print(f"Property variances have been saved to '{output_file}'.")


######################## generate several graphs for one selected graph #################
if False:
    # 100 générations pour graph_25 

    nb_generations = 100  
    output_file = "graph_25_properties.csv"

    denoise_model.eval()

    target_data = test_loader.dataset[25]
    data = target_data.to(device)

    ground_truth_properties = data.stats.detach().cpu().numpy()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow([
            "generation_id", 
            "nb_of_nodes", 
            "nb_of_edges", 
            "average_degree", 
            "nb_of_triangles", 
            "global_clustering_coefficient", 
            "maximum_k_core", 
            "nb_of_communities"
        ])
        
        for gen_id in tqdm(range(nb_generations), desc="Generating graphs"):
            samples = sample(
                denoise_model, 
                data.stats.unsqueeze(0), 
                latent_dim=args.latent_dim, 
                timesteps=args.timesteps, 
                betas=betas, 
                batch_size=1
            )
            x_sample = samples[-1]
            
            adj_matrix = autoencoder.decode_mu(x_sample, data.stats).squeeze(0).detach().cpu().numpy()
            edges = construct_nx_from_adj(adj_matrix).edges()
            
            properties = extract_graph_properties(edges)
            
            writer.writerow([gen_id] + properties)

    print(f"Properties of 100 generations for graph_25 have been saved to '{output_file}'.")
