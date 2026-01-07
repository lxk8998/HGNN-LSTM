# HGNN-LSTM

This repository provides the implementation of **HGNN-LSTM**, a hybrid deep learning framework that integrates a Heterogeneous Graph Neural Network (HGNN) with a Long Short-Term Memory (LSTM) network for multi-station river water level prediction.

## Repository Structure

```
HGNN-LSTM/
│── HGNN_LSTM/
│   ├── main.py        
│   ├── models.py        
│   ├── run_experiments.py        
│   └── trainer.py          
│── baseline/
│   ├── GNN_main.py              
│   ├── GNN_run_experiments.py              
│   ├── models.py               
│   ├── temporal_model_main.py               
│   ├── temporal_model_run_experiments.py              
│   └── trainer.py
│── data/
│   ├── data.xlsx             
│── utils/
│   ├── __init__.py       
│   ├── concat_features.py      
│   ├── early_stop.py        
│   ├── metrics.py       
│   ├── reverse_normalization.py        
│   ├── save_checkpoint.py        
│   ├── save_node_metrics.py        
│   ├── save_npz.py        
│   ├── split_hydro.py        
│   └── split_meteo.py           
│── .gitignore                 
│── LICENSE             
│── README.md

```

# File Description

## HGNN_LSTM/

**main.py**  
Main entry script for training, validation, and testing the HGNN-LSTM model.

**models.py**  
Definitions of the HGNN-LSTM model architecture.

**run_experiments.py**  
Script for launching training experiments and defining hyperparameter settings.

**trainer.py**  
Training pipeline that loads data from the dataloader, performs forward computation, loss calculation, and backpropagation.

---

## baseline/

**GNN_main.py**  
Main script for training and evaluating temporal GNN-LSTM model.

**GNN_run_experiments.py**  
Experiment launcher for GNN-LSTM model.

**models.py**  
Model architecture definitions for baseline models.

**temporal_model_main.py**  
Main script for training and evaluating temporal baseline models (e.g., LSTM).

**temporal_model_run_experiments.py**  
Experiment launcher for temporal baseline models.

**trainer.py**  
Training and evaluation pipeline for baseline models.

---
## data/
**data.xlsx**  
The data of this study. Due to data confidentiality, only part of the dataset used in this study is provided in the data directory. Data from the BY meteorological station are included for completeness, although this station was not used in the experiments.

---

## utils/

**__init__.py**  
Utility module initialization.

**concat_features.py**  
Utilities for concatenating meteorological and hydrological station features.

**early_stop.py**  
Early stopping strategy based on validation performance.

**metrics.py**  
Evaluation metrics for model performance assessment.

**reverse_normalization.py**  
Functions for reversing normalized data to original scales.

**save_checkpoint.py**  
Utilities for saving and loading model checkpoints.

**save_node_metrics.py**  
Functions for saving node-level evaluation metrics.

**save_npz.py**  
Utilities for saving prediction data in NPZ format.

**split_hydro.py**  
Data splitting utilities for hydrological station data.

**split_meteo.py**  
Data splitting utilities for meteorological station data.

---


