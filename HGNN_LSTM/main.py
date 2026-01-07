import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import utils.split_meteo as split_meteo
import utils.split_hydro as split_hydro
import utils.save_checkpoint as save_checkpoint
import os
import models
from utils.save_npz import save_results_npz
from trainer import run_model
from utils.save_node_metrics import save_metrics_to_excel
from utils.early_stop import EarlyStopping
import pandas as pd

def train_and_evaluate(config, futurelen, exp_dir, ckpt_path, ckpt_path_best, plot_dir, node_dir):
    # load hyperparameter and data
    seqlen = config["seqlen"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    hidden_dim_gnn = config["hidden_dim_gnn"]
    hidden_dim_lstm = config["hidden_dim_lstm"]
    meteo_feature_dim = config["meteo_feature_dim"]
    hydro_feature_dim = config["hydro_feature_dim"]
    device = config["device"]
    patience = config["patience"]

    num_meteo_stations = config["num_meteo_stations"]
    num_hydro_stations = config["num_hydro_stations"]
    df_meteo = config["df_meteo"]
    df_hydro = config["df_hydro"]
    meteo_training_scaled = config["meteo_training_scaled"]
    meteo_validation_scaled = config["meteo_validation_scaled"]
    meteo_testing_scaled = config["meteo_testing_scaled"]
    hydro_training_scaled = config["hydro_training_scaled"]
    hydro_validation_scaled = config["hydro_validation_scaled"]
    hydro_testing_scaled = config["hydro_testing_scaled"]
    dict_meteo = config["dict_meteo"]
    dict_hydro = config["dict_hydro"]
    scaler_hydro = config["scaler_hydro"]
    hydro_training = config["hydro_training"]
    hydro_validation = config["hydro_validation"]
    hydro_testing = config["hydro_testing"]
    reverse_dict_hydro = config["reverse_dict_hydro"]
    meteo_edge_index = config["meteo_edge_index"]
    hydro_edge_index = config["hydro_edge_index"]

    
    # split data
    meteo_all_train, meteo_all_valid, meteo_all_test = split_meteo.stack_meteo(df_meteo, meteo_training_scaled,
                                                                            meteo_validation_scaled,
                                                                            meteo_testing_scaled,seqlen, futurelen, dict_meteo, meteo_feature_dim)

    hydro_all_train, train_label, hydro_all_valid, valid_label, hydro_all_test, test_label = split_hydro.stack_hydro(df_hydro,
                                                                            hydro_training_scaled,
                                                                            hydro_validation_scaled,
                                                                            hydro_testing_scaled, seqlen,
                                                                            futurelen, dict_hydro,
                                                                            hydro_feature_dim)

    train_dataset = TensorDataset(meteo_all_train, hydro_all_train, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    train_loader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    valid_dataset = TensorDataset(meteo_all_valid, hydro_all_valid, valid_label)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_dataset = TensorDataset(meteo_all_test, hydro_all_test, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = models.HGNN_LSTM(hidden_dim_gnn, hidden_dim_lstm, meteo_feature_dim, hydro_feature_dim, futurelen, num_hydro_stations,
                             hydro_edge_index, meteo_edge_index, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(0, epochs), desc="Epoch", leave=False):
        # train loss
        metrics_train, train_loss = run_model(
        model, train_loader, device,
        num_hydro_stations, hydro_training, scaler_hydro,
        criterion, optimizer, train_mode=True
        )
        # validation loss
        metrics_val, valid_loss = run_model(
        model, valid_loader, device,
        num_hydro_stations, hydro_validation, scaler_hydro,
        criterion, train_mode=False
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        early_stopping(valid_loss)

        ckpt_path_ = os.path.join(ckpt_path, f"checkpoint.pth")
        ckpt_path_best_ = os.path.join(ckpt_path_best, f"checkpoint.pth")
        # Save model weights 
        save_checkpoint.save_checkpoint(model, optimizer, ckpt_path_)

        if early_stopping.is_improved:
            # Save best model weights 
            save_checkpoint.save_checkpoint(model, optimizer, ckpt_path_best_)

        if early_stopping.early_stop:
            print(f"stop at epoch{epoch}")
            break
    
    # draw loss curve
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_list, label="Train loss")
    plt.plot(val_loss_list, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"loss.png"))
    plt.close()        

    best_file = os.path.join(ckpt_path_best, "checkpoint.pth")
    save_checkpoint.load_checkpoint(model, optimizer, best_file)
    # train metrics
    metrics_train_noshuffle, train_noshuffle_loss = run_model(
    model, train_loader_noshuffle, device,
    num_hydro_stations, hydro_training, scaler_hydro,
    criterion, train_mode=False
    )
    # validation metrics
    metrics_val, valid_loss = run_model(
    model, valid_loader, device,
    num_hydro_stations, hydro_validation, scaler_hydro,
    criterion, train_mode=False
    )
    
    metrics_list = []
    for step in range(futurelen):
        metrics_train_step = metrics_train_noshuffle[step]
        metrics_val_step = metrics_val[step]

        avg_RMSE_train = np.mean(metrics_train_step[0]).round(3)
        avg_NSE_train = np.mean(metrics_train_step[1]).round(3)
        avg_MAE_train = np.mean(metrics_train_step[2]).round(3)
    
        avg_RMSE_valid = np.mean(metrics_val_step[0]).round(3)
        avg_NSE_valid = np.mean(metrics_val_step[1]).round(3)
        avg_MAE_valid = np.mean(metrics_val_step[2]).round(3)

        metrics_list.append({
            "futurelen": step + 1,
            "train_RMSE": avg_RMSE_train,
            "train_NSE": avg_NSE_train,
            "train_MAE": avg_MAE_train,
            "train_loss":train_noshuffle_loss,
            
            "valid_RMSE": avg_RMSE_valid,
            "valid_NSE": avg_NSE_valid,
            "valid_MAE": avg_MAE_valid,
            "valid_loss":valid_loss,
        })
        # Save metrics for each staion(train)
        save_metrics_to_excel(
        metrics_train_step,
        reverse_dict_hydro,
        step+1,
        save_dir=node_dir,
        prefix="train"
        )
        # Save metrics for each staion(validation)
        save_metrics_to_excel(
        metrics_val_step,
        reverse_dict_hydro,
        step+1,
        save_dir=node_dir,
        prefix="valid"
        ) 

    # test stage 

    # test metrics
    # metrics_test, test_loss = run_model(
    # model, test_loader, device,
    # num_hydro_stations, hydro_testing, scaler_hydro,
    # criterion, train_mode=False
    # )

    # for step in range(futurelen):
    #     metrics_test_step = metrics_test[step]
    #     avg_RMSE_test = np.mean(metrics_test_step[0]).round(3)
    #     avg_NSE_test = np.mean(metrics_test_step[1]).round(3)
    #     avg_MAE_test = np.mean(metrics_test_step[2]).round(3)
        
    #     metrics_list.append({
    #         "futurelen": step + 1,
    #         "test_RMSE": avg_RMSE_test,
    #         "test_NSE": avg_NSE_test,
    #         "test_MAE": avg_MAE_test,
    #         "test_loss":test_loss,

    #     })
    #     # Save metrics for each staion(test)
    #     save_metrics_to_excel(
    #     metrics_test_step,
    #     reverse_dict_hydro,
    #     step+1,
    #     save_dir=node_dir,
    #     prefix="test"
    #     )
    #     train_act_inverse_pernode, train_pre_inverse_pernode = metrics_train_step[6], metrics_train_step[7]
    #     val_act_inverse_pernode, val_pre_inverse_pernode = metrics_val_step[6], metrics_val_step[7]
    #     test_act_inverse_pernode, test_pre_inverse_pernode = metrics_test_step[6], metrics_test_step[7]

    #     # 保存预测和实际曲线为 npz 文件
    #     save_results_npz(
    #     exp_dir=exp_dir,
    #     futurelen=step+1,
    #     train_actual_inverse_per_node=train_act_inverse_pernode,
    #     train_predictions_inverse_per_node=train_pre_inverse_pernode,
    #     val_actual_inverse_per_node=val_act_inverse_pernode,
    #     val_predictions_inverse_per_node=val_pre_inverse_pernode,
    #     test_actual_inverse_per_node=test_act_inverse_pernode,
    #     test_predictions_inverse_per_node=test_pre_inverse_pernode,
    #     reverse_dict_hydro=reverse_dict_hydro,
    #     num_hydro_stations=num_hydro_stations
    #     )

    return pd.DataFrame(metrics_list)

