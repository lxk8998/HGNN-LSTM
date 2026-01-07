import torch
import numpy as np
import utils.reverse_normalization as reverse_normalization

def run_model(model, loader, device,
              num_hydro_stations, hydro_origin, scaler,
              criterion=None, optimizer=None, train_mode=False):
    total_loss = 0.0
    batch_count = 0
    if train_mode:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(train_mode):
        for meteo, hydro, labels in loader:
            meteo, hydro, labels = meteo.to(device), hydro.to(device), labels.to(device)
            
            outputs = model(meteo, hydro)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batch_count += 1
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
            
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        avg_loss = total_loss / batch_count  
    # (batchsize, num_hydro_stations, futurelen)
    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    futurelen = preds_np.shape[2]
    metrics = []
    for step in range(futurelen):
        labels_np_step = labels_np[:,:,step].reshape(-1, num_hydro_stations)
        preds_np_step = preds_np[:,:,step].reshape(-1, num_hydro_stations)
        step_metrics = reverse_normalization.reverse_cal(
             labels_np_step, preds_np_step,
             num_hydro_stations, hydro_origin, scaler,
        )
        metrics.append(step_metrics)

    return metrics, avg_loss
