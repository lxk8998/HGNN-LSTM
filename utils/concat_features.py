import torch

def concat_hydro_meteo_features(
    hydro_all_train,
    hydro_all_valid,
    hydro_all_test,
    meteo_all_train,
    meteo_all_valid,
    meteo_all_test,
    meteo_edge_index,
    num_hydro_stations
):
    all_data_train = []
    all_data_valid = []
    all_data_test = []

    for hydro_idx in range(num_hydro_stations):
        hydro_features_train = hydro_all_train[:, :, hydro_idx, :]
        hydro_features_valid = hydro_all_valid[:, :, hydro_idx, :]
        hydro_features_test  = hydro_all_test[:, :, hydro_idx, :]

        connected_meteo_train = []
        connected_meteo_valid = []
        connected_meteo_test  = []

        # Find meteorological stations connected to the hydrological station
        for i in range(meteo_edge_index.shape[1]):
            if meteo_edge_index[1, i] == hydro_idx:
                meteo_idx = meteo_edge_index[0, i].item()

                connected_meteo_train.append(
                    meteo_all_train[:, :, meteo_idx, :]
                )
                connected_meteo_valid.append(
                    meteo_all_valid[:, :, meteo_idx, :]
                )
                connected_meteo_test.append(
                    meteo_all_test[:, :, meteo_idx, :]
                )

        meteo_avg_train = torch.mean(torch.stack(connected_meteo_train), dim=0)
        meteo_avg_valid = torch.mean(torch.stack(connected_meteo_valid), dim=0)
        meteo_avg_test  = torch.mean(torch.stack(connected_meteo_test), dim=0)

        combined_train = torch.cat([hydro_features_train, meteo_avg_train], dim=2)
        combined_valid = torch.cat([hydro_features_valid, meteo_avg_valid], dim=2)
        combined_test  = torch.cat([hydro_features_test, meteo_avg_test], dim=2)

        all_data_train.append(combined_train)
        all_data_valid.append(combined_valid)
        all_data_test.append(combined_test)

    all_data_train = torch.stack(all_data_train, dim=2)
    all_data_valid = torch.stack(all_data_valid, dim=2)
    all_data_test  = torch.stack(all_data_test, dim=2)

    return all_data_train, all_data_valid, all_data_test
