import numpy as np
import utils.metrics as metrics

def reverse_cal(all_actuals_np, all_predictions_np, num_hydro_stations, 
                hydro_origin, scaler_hydro):
    RMSE_node = []
    NSE_node = []
    MAE_node = []
    actual_inverse_node = []
    predictions_inverse_node = []
    
    i = 0
    for node_idx in range(num_hydro_stations):
        predictions = all_predictions_np[:, node_idx].flatten()
        actual = all_actuals_np[:, node_idx].flatten()

        predictions_expanded = np.zeros((predictions.shape[0], hydro_origin.shape[1]))
        actual_expanded = np.zeros((actual.shape[0], hydro_origin.shape[1]))

        predictions_expanded[:, i] = predictions
        actual_expanded[:, i] = actual

        predictions_inverse = scaler_hydro.inverse_transform(predictions_expanded)
        actual_inverse = scaler_hydro.inverse_transform(actual_expanded)

        predictions_inverse = predictions_inverse[:, i]
        actual_inverse = actual_inverse[:, i]
        i += 1
        actual_inverse_node.append(actual_inverse)
        predictions_inverse_node.append(predictions_inverse)
        
        nse = metrics.calculate_NSE(actual_inverse, predictions_inverse)
        rmse = metrics.calculate_RMSE(actual_inverse, predictions_inverse)
        mae = metrics.calculate_MAE(actual_inverse, predictions_inverse)
        RMSE_node.append(rmse)
        NSE_node.append(nse)
        MAE_node.append(mae)
    return RMSE_node, NSE_node, MAE_node, actual_inverse_node, predictions_inverse_node