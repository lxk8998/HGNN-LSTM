# HGNN-LSTM

This repository provides the implementation of **HGNN-LSTM**, a hybrid deep learning framework that integrates a Heterogeneous Graph Neural Network (HGNN) with a Long Short-Term Memory (LSTM) network for multi-station river water level prediction.

## Repository Structure

```text
HGNN-LSTM/
│── HGNN_LSTM/
│   ├── main.py        # Data preprocessing script
│   ├── models.py        # Data preprocessing script
│   ├── run_experiments.py        # Data preprocessing script
│   └── trainer.py           # Dataset loading and formatting
│── baseline/
│   ├── GNN_main.py               # Heterogeneous graph neural network module
│   ├── GNN_run_experiments.py               # Heterogeneous graph neural network module
│   ├── models.py               # Heterogeneous graph neural network module
│   ├── temporal_model_main.py               # Heterogeneous graph neural network module
│   ├── temporal_model_run_experiments.py               # LSTM temporal modeling module
│   └── trainer.py          # Integrated HGNN-LSTM model
│── utils/
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   ├── graph_utils.py        # Graph construction utilities
│   └── metrics.py            # Evaluation metrics
│── train.py                  # Model training script
│── test.py                   # Model evaluation script
│── config.yaml               # Configuration file
│── README.md

