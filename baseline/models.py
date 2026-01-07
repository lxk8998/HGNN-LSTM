import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

# GNN-LSTM
class GNN_LSTM(torch.nn.Module):
    def __init__(self, hidden_dim_gnn, hidden_dim_lstm, futurelen,
                 num_hydro_stations, edge_index, device):
        super(GNN_LSTM, self).__init__()

        self.num_hydro_stations = num_hydro_stations
        self.device = device
        self.edge_index = edge_index.to(self.device)
        self.hidden_dim_gnn = hidden_dim_gnn
        self.hidden_dim_lstm = hidden_dim_lstm

        self.gnn = GraphConv(1, self.hidden_dim_gnn)
        self.lstm_list = nn.ModuleList([
        nn.LSTM(self.hidden_dim_gnn, self.hidden_dim_lstm, batch_first=True)
        for _ in range(self.num_hydro_stations)
        ])
        self.linear = torch.nn.Linear(self.hidden_dim_lstm, futurelen)  
        self.reset_parameters()

    def reset_parameters(self):
        # initialization for GNN
        for name, param in self.gnn.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            elif 'bias' in name:
                init.constant_(param, 0)

        # initialization for LSTM
        for lstm in self.lstm_list:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    init.xavier_normal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # initialization for Output layer
        if self.linear.weight is not None:
            init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01) 
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0) 
    
    # Create batched graph
    def get_batch_edge_index(self, batch_size, seq_len):
        num_graphs = batch_size * seq_len
        num_edges = self.edge_index.size(1)
        offsets = torch.arange(num_graphs,device=self.device) * self.total_nodes
        repeated_edge_index = self.edge_index.repeat(1, num_graphs)
        offsets_expanded = offsets.repeat_interleave(num_edges)
        batch_edge_index = repeated_edge_index + offsets_expanded
        
        return batch_edge_index

    def forward(self, data_meteo, data_hydro):
        batch_size, seq_len, num_nodes_hydro, feature = data_hydro.size()
        batch_size, seq_len, num_nodes_meteo, feature = data_meteo.size()
        total_nodes = num_nodes_hydro + num_nodes_meteo
        self.total_nodes = total_nodes
        # (batchsize, seq_len, total_nodes, feature)
        x = torch.cat([data_hydro,data_meteo], dim=2)
        
        # (batchsize * seq_len * total_nodes, feature)
        x_reshaped = x.view(-1, feature).to(self.device)
        batch_edge_index = self.get_batch_edge_index(batch_size, seq_len)

        # (batchsize * seq_len * total_nodes, hidden_dim_gnn)
        x_gnn = F.leaky_relu(self.gnn(x_reshaped, batch_edge_index))
        # (batchsize, seq_len, total_nodes, hidden_dim_gnn)
        x_gnn = x_gnn.view(batch_size, seq_len, self.total_nodes, -1)
        # (batchsize, seq_len, num_nodes_hydro, hidden_dim_gnn)
        hydro_gnn = x_gnn[:, :, :num_nodes_hydro, :] 
        
        # LSTM for future prediction
        outputs = []
        for node in range(num_nodes_hydro):
            # (batch_size, seq_len, hidden_dim_gnn)
            node_series = hydro_gnn[:, :, node, :]
            lstm_out, _ = self.lstm_list[node](node_series)
            # (batch_size, hidden_dim_lstm)
            last_output = lstm_out[:, -1, :]

            # (batch_size, futurelen)
            node_prediction = self.linear(last_output)
            outputs.append(node_prediction)

        # (batch_size, num_nodes, futurelen)
        predictions = torch.stack(outputs, dim=1)  
        return F.leaky_relu(predictions)
    
# LSTM
class LSTM(torch.nn.Module):
    def __init__(self, hidden_dim, feature_dim, futurelen, num_hydro_stations):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_hydro_stations = num_hydro_stations

        self.lstm_list = nn.ModuleList([
        nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True)
        for _ in range(self.num_hydro_stations)
        ])
        self.linear = torch.nn.Linear(self.hidden_dim, futurelen)  
        self.reset_parameters()

    def reset_parameters(self):
        # initialization for LSTM
        for lstm in self.lstm_list:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    init.xavier_normal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)
        # initialization for Output layer
        for name, param in self.linear.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            elif 'bias' in name:
                init.constant_(param, 0)

    def forward(self, data_hydro):
        batch_size, seq_len, num_nodes_hydro, feature = data_hydro.size()

        outputs = []
        for node in range(num_nodes_hydro):
            # (batch_size, seq_len, hidden_dim)
            node_series = data_hydro[:, :, node, :]

            lstm_out, _ = self.lstm_list[node](node_series)

            # (batch_size, hidden_dim)
            last_output = lstm_out[:, -1, :]            
            node_prediction = self.linear(last_output)
            outputs.append(node_prediction)
       
        predictions = torch.stack(outputs, dim=1)  
        return F.leaky_relu(predictions)


# GRU
class GRU(torch.nn.Module):
    def __init__(self, hidden_dim, feature_dim, futurelen, num_hydro_stations):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_hydro_stations = num_hydro_stations

        self.gru_list = nn.ModuleList([
        nn.GRU(self.feature_dim, self.hidden_dim, batch_first=True)
        for _ in range(self.num_hydro_stations)
        ])
        self.linear = torch.nn.Linear(self.hidden_dim, futurelen)  
        self.reset_parameters()

    def reset_parameters(self):
        # initialization for GRU
        for gru in self.gru_list:
            for name, param in gru.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    init.xavier_normal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # initialization for Output layer
        for name, param in self.linear.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            elif 'bias' in name:
                init.constant_(param, 0)

    def forward(self, data_hydro):
        batch_size, seq_len, num_nodes_hydro, feature = data_hydro.size()

        outputs = []
        for node in range(num_nodes_hydro):
            node_series = data_hydro[:, :, node, :]
            gru_out, _ = self.gru_list[node](node_series)
            last_output = gru_out[:, -1, :]
            node_prediction = self.linear(last_output)
            outputs.append(node_prediction)
        predictions = torch.stack(outputs, dim=1)  
            
        return F.leaky_relu(predictions)
    
# Transformer
class Transformer(torch.nn.Module):
    def __init__(self, hidden_dim, feature_dim, futurelen, num_hydro_stations, nhead, num_layers):
        super(Transformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_hydro_stations = num_hydro_stations

        # feature_dim -> hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim)
            for _ in range(num_hydro_stations)
        ])
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_list = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            for _ in range(num_hydro_stations)
        ])
        
        self.linear = torch.nn.Linear(self.hidden_dim, futurelen)  
        self.reset_parameters()

    def reset_parameters(self):
        # initialization for projection layer
        for proj in self.input_proj:
            init.xavier_normal_(proj.weight)
            init.constant_(proj.bias, 0)

        # initialization for Transformer 
        for transformer in self.transformer_list:
            for p in transformer.parameters():
                if p.dim() > 1:
                    init.xavier_normal_(p)

        # initialization for Output layer
        init.kaiming_normal_(
            self.linear.weight,
            mode='fan_in',
            nonlinearity='leaky_relu',
            a=0.01
        )
        init.constant_(self.linear.bias, 0)

    def forward(self, data_hydro):
        batch_size, seq_len, num_nodes_hydro, feature = data_hydro.size()

        outputs = []
        for node in range(num_nodes_hydro):
            # (batch_size, seq_len, feature)
            node_series = data_hydro[:, :, node, :]

            # (batch_size, seq_len, hidden_dim)
            node_emb = self.input_proj[node](node_series)
            trans_out = self.transformer_list[node](node_emb)

            # (batch_size, hidden_dim)
            last_output = trans_out[:, -1, :]
            # (batch_size, futurelen)
            node_prediction = self.linear(last_output)

            outputs.append(node_prediction)

        # (batch_size, num_nodes, futurelen)
        predictions = torch.stack(outputs, dim=1)
        return F.leaky_relu(predictions)