import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import GNN_main
from itertools import product
from tqdm import tqdm

seed = 42
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
set_seed(seed)
# set gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Load Dataset
data = pd.read_excel("data.xlsx")
start_meteo = 9
df_meteo = data.iloc[:,start_meteo:]
dict_meteo = {}
count_meteo = 0
for name in df_meteo.columns:
    dict_meteo[name] = count_meteo
    count_meteo += 1

start_hydro = 1
end_hydro = 9
df_hydro = data.iloc[:, start_hydro:end_hydro]
dict_hydro = {}
count_hydro = 0
for name in df_hydro.columns:
    dict_hydro[name] = count_hydro
    count_hydro += 1
reverse_dict_hydro = {v: k for k, v in dict_hydro.items()}

# train: validation: test  = 7: 1: 2
train_split = round(len(data) * 0.7)
valid_split = round(len(data) * 0.1)

meteo_training = df_meteo[:train_split].values  
meteo_validation = df_meteo[train_split:train_split + valid_split].values  
meteo_testing = df_meteo[train_split + valid_split:].values  
hydro_training = df_hydro[:train_split].values
hydro_validation = df_hydro[train_split:train_split + valid_split].values
hydro_testing = df_hydro[train_split + valid_split:].values

# Normalization 
scaler_meteo = MinMaxScaler(feature_range=(0, 1))
scaler_hydro = MinMaxScaler(feature_range=(0, 1))

meteo_training_scaled = scaler_meteo.fit_transform(meteo_training)
meteo_validation_scaled = scaler_meteo.transform(meteo_validation)
meteo_testing_scaled = scaler_meteo.transform(meteo_testing)

hydro_training_scaled = scaler_hydro.fit_transform(hydro_training)
hydro_validation_scaled = scaler_hydro.transform(hydro_validation)
hydro_testing_scaled = scaler_hydro.transform(hydro_testing)

# Hyperparameter search space
lr_list = [1e-2, 1e-3, 5e-3]
bs_list = [16, 32, 64, 128]
futurelen = 3  

config = {
        "epochs": 300,
        "seqlen": 8,
        "hidden_dim_gnn": 16,
        "hidden_dim_lstm": 16,
        "num_meteo_stations": 5,
        "num_hydro_stations": 8,
        "meteo_feature_dim": 1,
        "hydro_feature_dim": 1,
        "device": device,
        "seed": seed,
        "patience": 30,

        "df_meteo": df_meteo,
        "df_hydro": df_hydro,
        "meteo_training_scaled": meteo_training_scaled,
        "meteo_validation_scaled": meteo_validation_scaled,
        "meteo_testing_scaled": meteo_testing_scaled,
        "hydro_training_scaled": hydro_training_scaled,
        "hydro_validation_scaled": hydro_validation_scaled,
        "hydro_testing_scaled": hydro_testing_scaled,
        "scaler_hydro":scaler_hydro,
        "hydro_training":hydro_training,
        "hydro_validation":hydro_validation,
        "hydro_testing":hydro_testing,
        "dict_meteo": dict_meteo,
        "dict_hydro": dict_hydro,
        "reverse_dict_hydro": reverse_dict_hydro,
    }

edge_index = torch.tensor([
    [ 0, 1, 2, 3, 5, 6, 7, 8, 8, 8, 10, 10, 10, 10, 11, 12, 12, 12, 12, 12],# Source: Meteo and Hydro station indices
    [ 6, 4, 6, 1, 6, 1, 6, 1, 3, 4,  7,  4,  1,  6,  2,  0,  1,  4,  5,  6]] # Target: Hydro station indices
)
config["edge_index"] = edge_index

total =  len(lr_list) * len(bs_list)
path = os.path.join("GNN-LSTM", f"experiments_GNN-LSTM_seed{seed}")
for lr, bs in tqdm(product(lr_list, bs_list), total=total, desc="Task"):

    config["lr"] = lr
    config["batch_size"] = bs
    exp_name = f"lr{lr}_bs{bs}_ep{config['epochs']}_gnn{config['hidden_dim_gnn']}_lstm{config['hidden_dim_lstm']}_seed{config['seed']}"
    exp_dir = os.path.join(path, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpt_dir_best = os.path.join(exp_dir, "checkpoints_best")
    plot_dir = os.path.join(exp_dir, "plots")
    node_dir = os.path.join(exp_dir, "node_metrics")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ckpt_dir_best, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(node_dir, exist_ok=True)

    res = GNN_main.train_and_evaluate(
        config=config,
        futurelen=futurelen,
        exp_dir=exp_dir,
        ckpt_path=ckpt_dir,
        ckpt_path_best=ckpt_dir_best,
        plot_dir=plot_dir,
        node_dir=node_dir
    )