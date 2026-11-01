# Trajectory Forecasting and Anomaly Detection on AIS Vessel Data

### Authors
- Marc Saade (s251555)
- Rachid Tazi (s251709)
- Florian Küfner (s251969)

### Deep Learning Project — DTU 2025

This repository contains our project on trajectory prediction and anomaly detection for maritime vessels using AIS (Automatic Identification System) data.
We train recurrent neural networks to model short-term vessel motion and use the prediction errors to flag potentially unusual behaviours at sea.

## Project Overview
Automatic Identification System (AIS) data is irregular, noisy, and highly variable across seasons, regions, and vessel types.
Our project addresses three core questions:

- Can we learn short-term vessel trajectories from AIS sequences using RNNs?
- Are prediction errors a meaningful proxy for detecting abnormal maritime behaviour?
- How stable is the model across different ship types, seasons, and geographical regions?

Our approach combines trajectory resampling, sequence modeling, and error-based anomaly scoring.

## Repository Structure
.
├── notebooks/              # Exploratory notebooks and experiments
├── src/                    # Data loader, resampling logic, model definitions, utilities
├── models/                 # Trained LSTM/GRU checkpoints
└── data/processed/         # Preprocessed AIS trajectories (not included)

The final_notebook.ipynb regroups all the main results.

## How to run 
pip install -r requirements.txt
jupyter notebook final_notebook.ipynb
