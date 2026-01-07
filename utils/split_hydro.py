import torch
import numpy as np

def create_hydro(data, seqlen, futurelen, station_num, feature_dim):
    dataX = []
    dataY = []
    for i in range(0, len(data) - seqlen - futurelen + 1):
        dataX.append(data[i: i + seqlen, station_num])
        dataY.append(data[i + seqlen: i + seqlen + futurelen, station_num])
    return np.array(dataX).reshape(-1, seqlen, feature_dim), np.array(dataY)

def stack_hydro(data, data_train, data_valid, data_test, seqlen, futurelen, dict, feature_dim):
    all_train = []
    all_valid = []
    all_test = []
    all_train_label = []
    all_valid_label = []
    all_test_label = []
    for name in data.columns:
        # X(num, seqlen, features)
        # Y(num, futurelen)
        train, label_train = create_hydro(data_train, seqlen, futurelen, dict[name], feature_dim)
        valid, label_valid = create_hydro(data_valid, seqlen, futurelen, dict[name], feature_dim)
        test, label_test = create_hydro(data_test, seqlen, futurelen, dict[name], feature_dim)
        all_train.append(train)
        all_train_label.append(label_train)
        all_valid.append(valid)
        all_valid_label.append(label_valid)
        all_test.append(test)
        all_test_label.append(label_test)

    # (num, seqlen, numnodes, features)
    all_train = np.stack(all_train, axis=2)
    # (num, numnodes, futurenlen)
    all_train_label = np.stack(all_train_label, axis=1)

    all_valid = np.stack(all_valid, axis=2)
    all_valid_label = np.stack(all_valid_label, axis=1)

    all_test = np.stack(all_test, axis=2)
    all_test_label = np.stack(all_test_label, axis=1)

    all_train = torch.tensor(all_train, dtype=torch.float32)
    all_train_label = torch.tensor(all_train_label, dtype=torch.float32)
    all_valid = torch.tensor(all_valid, dtype=torch.float32)
    all_valid_label = torch.tensor(all_valid_label, dtype=torch.float32)
    all_test = torch.tensor(all_test, dtype=torch.float32)
    all_test_label = torch.tensor(all_test_label, dtype=torch.float32)
    
    return all_train, all_train_label, all_valid, all_valid_label, all_test, all_test_label
