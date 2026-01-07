import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv,GraphConv
import torch.nn.functional as F
import torch.nn.init as init

# HGNN-LSTM
class HGNN_LSTM(torch.nn.Module):
    def __init__(self, hidden_dim_gnn, hidden_dim_lstm, meteo_feature_dim, hydro_feature_dim,
                 futurelen, num_hydro_stations, hydro_edge_index, meteo_edge_index, device):
        super(HGNN_LSTM, self).__init__()
        self.device = device
        self.meteo_edge_index = meteo_edge_index.to(self.device)
        self.hydro_edge_index = hydro_edge_index.to(self.device)
        self.num_hydro_stations = num_hydro_stations
        self.hidden_dim_gnn = hidden_dim_gnn
        self.hidden_dim_lstm = hidden_dim_lstm
        self.meteo_feature_dim = meteo_feature_dim
        self.hydro_feature_dim = hydro_feature_dim
        # HGNN
        self.convs = HeteroConv(convs={
            ('meteorological_station', 'affects', 'hydrological_station'):  GraphConv((meteo_feature_dim, hydro_feature_dim), self.hidden_dim_gnn),
            ('hydrological_station', 'connected', 'hydrological_station'): GraphConv(hydro_feature_dim, self.hidden_dim_gnn)
        }, aggr='mean') 
        # LSTM
        self.lstm_list = nn.ModuleList([
        nn.LSTM(self.hidden_dim_gnn, self.hidden_dim_lstm, batch_first=True)
        for _ in range(self.num_hydro_stations)
        ])
        # Output layer
        self.linear = torch.nn.Linear(self.hidden_dim_lstm, futurelen)  
        self.reset_parameters()

    def reset_parameters(self):
        # initialization for HGNN
        for name, param in self.convs.named_parameters():
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
        edge_index_dict = {}
        # Process hydrological station connections
        num_graphs = batch_size * seq_len
        num_edges_hydro = self.hydro_edge_index.size(1)
        offsets_hydro = torch.arange(num_graphs, device=self.device) * self.num_hydro_stations
        repeated_hydro_edge_index = self.hydro_edge_index.repeat(1, num_graphs)
        hydro_offsets_expanded = offsets_hydro.repeat_interleave(num_edges_hydro)
        batch_hydro_edge_index = repeated_hydro_edge_index + hydro_offsets_expanded
    
        # Process meteorological affects hydrological
        num_edges_meteo = self.meteo_edge_index.size(1)
        offsets_meteo = torch.arange(num_graphs, device=self.device) * self.num_nodes_meteo
        repeated_meteo_edge_index = self.meteo_edge_index.repeat(1, num_graphs)

        meteo_offsets_expanded = offsets_meteo.repeat_interleave(num_edges_meteo)
        hydro_offsets_expanded = offsets_hydro.repeat_interleave(num_edges_meteo)
        batch_index_m_source = repeated_meteo_edge_index[0, :] + meteo_offsets_expanded
        batch_index_m_target = repeated_meteo_edge_index[1, :] + hydro_offsets_expanded
        batch_meteo_edge_index = torch.stack([batch_index_m_source, batch_index_m_target], dim=0)
      
        edge_index_dict[('hydrological_station', 'connected', 'hydrological_station')] = batch_hydro_edge_index
        edge_index_dict[('meteorological_station', 'affects', 'hydrological_station')] = batch_meteo_edge_index
        return edge_index_dict

    def forward(self, data_meteo, data_hydro):
        batch_size, seq_len, num_nodes_hydro, feature = data_hydro.size()
        batch_size, seq_len, num_nodes_meteo, feature = data_meteo.size()
        
        self.num_nodes_meteo = num_nodes_meteo
        hydro_feat_flat = data_hydro.view(batch_size * seq_len * num_nodes_hydro, -1)
        meteo_feat_flat = data_meteo.view(batch_size * seq_len * num_nodes_meteo, -1)
        x_dict = {
        'hydrological_station': hydro_feat_flat,
        'meteorological_station': meteo_feat_flat,
        }   
        edge_index_dict  = self.get_batch_edge_index(batch_size, seq_len)
        x_dict = self.convs(x_dict, edge_index_dict)
        
        hydro_gnn_feat_flat = F.leaky_relu(x_dict['hydrological_station'])
        # (batch_size, seq_len, num_nodes_hydro, hidden_dim_gnn)
        hydro_gnn_feat = hydro_gnn_feat_flat.view(batch_size, seq_len, num_nodes_hydro, -1)

        # LSTM for future prediction
        outputs = []
        for node in range(num_nodes_hydro):
            # (batch_size, seq_len, hidden_dim_gnn)
            node_series = hydro_gnn_feat[:, :, node, :]
            lstm_out, _ = self.lstm_list[node](node_series)
            # (batch_size, hidden_dim_lstm)
            last_output = lstm_out[:, -1, :]
            # (batch_size, futurelen)
            node_prediction = self.linear(last_output)
            outputs.append(node_prediction)

        # (batch_size, num_nodes_hydro, futurelen)
        predictions = torch.stack(outputs, dim=1)
        return F.leaky_relu(predictions) 

