import numpy as np
import os

def save_results_npz(exp_dir, futurelen,
                     train_actual_inverse_node,
                     train_predictions_inverse_node,
                     val_actual_inverse_node,
                     val_predictions_inverse_node,
                     test_actual_inverse_node,
                     test_predictions_inverse_node,
                     reverse_dict_hydro,
                     num_hydro_stations):

    train_actual = np.stack(train_actual_inverse_node, axis=0)
    train_pred = np.stack(train_predictions_inverse_node, axis=0)
    val_actual = np.stack(val_actual_inverse_node, axis=0)
    val_pred = np.stack(val_predictions_inverse_node, axis=0)
    test_actual = np.stack(test_actual_inverse_node, axis=0)
    test_pred = np.stack(test_predictions_inverse_node, axis=0)
    stations = np.array([reverse_dict_hydro[i] for i in range(num_hydro_stations)])

    save_path = os.path.join(exp_dir, f"results_flen{futurelen}.npz")
    np.savez(
        save_path,
        train_actual = train_actual,
        train_pred = train_pred,
        val_actual = val_actual,
        val_pred = val_pred,
        test_actual = test_actual,
        test_pred = test_pred,
        stations = stations
    )
