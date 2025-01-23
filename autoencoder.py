import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import global_add_pool


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, n_cond, d_cond):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim+d_cond, hidden_dim)]
        mlp_layers.append(nn.Linear(hidden_dim+ d_cond, hidden_dim*2))
        mlp_layers.append(nn.Linear(hidden_dim*2 + d_cond, hidden_dim*3))
        mlp_layers.append(nn.Linear(hidden_dim*3 + d_cond, hidden_dim*4))
        mlp_layers.append(nn.Linear(hidden_dim*4, 2*n_nodes*(n_nodes-1)//2))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim)]
        bn_layers.append(nn.BatchNorm1d(hidden_dim*2))
        bn_layers.append(nn.BatchNorm1d(hidden_dim*3))
        bn_layers.append(nn.BatchNorm1d(hidden_dim*4))
        self.bn = nn.ModuleList(bn_layers)

        self.embedding_cond = nn.Sequential(
            nn.Linear(n_cond, d_cond // 2),
            nn.ReLU(),
            nn.Linear(d_cond // 2, d_cond)
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond, temp=1):
        cond = self.embedding_cond(cond)
        residual = x
        for i in range(len(self.mlp)-1):
            x = torch.cat((x, cond), dim=1)
            x = self.mlp[i](x)
            x = self.bn[i](x)
            x = self.relu(x)

            if residual.size(1) == x.size(1):
                x = x + residual
            residual = x

        x = self.mlp[-1](x)

        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=temp, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class GAT(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, num_heads = 4):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True))     

        for _ in range(n_layers-1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True))
        
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.last_bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  
            x = self.bn[i](x) 
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.last_bn(out)
        out = self.fc(out)
        return out
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, n_cond, d_cond, current_temp = 1.0):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder_gat = GAT(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder_gin = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Sequential(nn.Linear(2*hidden_dim_enc, hidden_dim_enc), nn.ReLU(), nn.Linear(hidden_dim_enc, latent_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(2*hidden_dim_enc, hidden_dim_enc), nn.ReLU(), nn.Linear(hidden_dim_enc, latent_dim))
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, n_cond, d_cond)
        self.current_temp = current_temp
        self.decay_rate = 0.99

    def forward(self, data, cond):
        x_g_gat = self.encoder_gat(data)
        x_g_gin = self.encoder_gin(data)
        x_g = torch.concat((x_g_gat, x_g_gin), dim = 1)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond, self.current_temp)
        return adj
    
    def encode(self, data):
        x_g_gat = self.encoder_gat(data)
        x_g_gin = self.encoder_gin(data)
        x_g = torch.concat((x_g_gat, x_g_gin), dim = 1)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar, cond):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g, cond, self.current_temp)
       return adj

    def decode_mu(self, mu, cond):
       adj = self.decoder(mu, cond, 0.99)
       return adj

    def loss_function(self, data, cond, beta=0.03):
        x_g_gat = self.encoder_gat(data)
        x_g_gin = self.encoder_gin(data)
        x_g = torch.concat((x_g_gat, x_g_gin), dim = 1)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g, cond, self.current_temp)

        recon = F.l1_loss(adj, data.A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
