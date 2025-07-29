# AutochaosNet
A Hyperparameter-Free Neurochaos Learning Algorithm
## Overview

**Neurochaos Learning (NL)** is a brain-inspired classification framework that utilizes chaotic dynamics to transform input data into meaningful features.  
**AutoChaosNet** is a novel, hyperparameter-free variant of the NL algorithm that eliminates the need for both training and parameter optimization.

This repository contains two simplified versions of AutoChaosNet:

- **TM AutoChaosNet**: Uses **TraceMean** as the only feature.
- **TM FR AutoChaosNet**: Uses **TraceMean** and **Firing Rate** as features.

Both variants are evaluated on 10 standard datasets.

## ⚙️ How to Run

For both versions:

1. Install dependencies:
   
pip install numpy scikit-learn pandas numba

3. Navigate to either TM_AutoChaosNet/ or TM_FR_AutoChaosNet/.

4. Run the main script: TM_testing.py inside the corresponding dataset folder to execute the classification pipeline.

## Results
The models were tested across 10 datasets. F1 Scores are computed for all datasets.

⏱️ Timing Metrics (e.g., average time per iteration) were computed only on the following datasets:

Iris

Seeds

Statlog

Sonar
