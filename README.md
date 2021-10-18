# HFT-LOB

**Introduction**

This repository contains the datasets and codes described in the paper "...". We propose a system model contains training, predicting and trading and evaluate it on the simulation data and the empirical data.

**Data**

In our paper, we evaluate our system model on both simulation dataset and historical stock data of Chinese A-share markets.

- The simulation dataset: We simulate three datasets with different financial environments and one dataset under a long-term and mixed period by a zero intelligence agent-based model with the [codes](https://github.com/JackBenny39/pyziabm).
- The CS-20 dataset: [Benchmark dataset](https://github.com/HKGSAS) is publicly available and downloaded.
- The proprietary dataset of CS-100: Provided by fintech company TradeMaster.

**Code**

The codes written by Python are provided in the [fold](Code) for the simulation dataset under a long-term and mixed period.

- DCNN.py: Labelling, training and predicting.
- invest_strat_op.py: Trading strategy with optimization and without optimization.
