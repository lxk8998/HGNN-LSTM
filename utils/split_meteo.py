import torch
import numpy as np

def create_meteo(data, seqlen, futurelen, station_num, feature_dim):
    dataX = []     
    for i in range(0, len(data) - seqlen - futurelen + 1):
        dataX.append(data[i: i + seqlen, station_num])
    return np.array(dataX).reshape(-1, seqlen, feature_dim)

def stack_meteo(data, data_train, data_valid, data_test, seqlen, futurelen, dict, feature_dim):
    all_train = []
    all_valid = []
    all_test = []
    for name in data.columns:
        # (num, seqlen, features)
        train = create_meteo(data_train, seqlen, futurelen, dict[name], feature_dim)
        valid = create_meteo(data_valid, seqlen, futurelen, dict[name], feature_dim)
        test = create_meteo(data_test, seqlen, futurelen, dict[name], feature_dim)
        all_train.append(train)
        all_valid.append(valid)
        all_test.append(test)

    # (num, seqlen, numnodes, features)
    all_train = np.stack(all_train, axis=2)
    all_valid = np.stack(all_valid, axis=2)
    all_test = np.stack(all_test, axis=2)

    all_train = torch.tensor(all_train, dtype=torch.float32)
    all_valid = torch.tensor(all_valid, dtype=torch.float32)
    all_test = torch.tensor(all_test, dtype=torch.float32)
    
    return all_train, all_valid, all_test











