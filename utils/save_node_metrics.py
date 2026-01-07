import pandas as pd
import os

def save_metrics_to_excel(metrics, reverse_dict_hydro, futurelen, save_dir, prefix):
    station_names = [reverse_dict_hydro[i] for i in range(len(metrics[0]))]
    metric_names = ["RMSE", "NSE", "MAE"]

    df_dict = {}
    for name, vals in zip(metric_names, metrics[:len(metric_names)]):
        df_dict[name] = [v for v in vals]

    df = pd.DataFrame(df_dict, index=station_names)
    df = df.round(3).astype(str)

    filename = f"{prefix}_{futurelen}.xlsx"
    file_path = os.path.join(save_dir, filename)
    df.to_excel(file_path)

