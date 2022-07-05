# Deep LOB Trading

**Introduction**

This repository contains the datasets and codes described in the paper "Deep LOB Trading: Half a second please!". We propose a system model contains training, predicting and trading, and evaluate it on the simulation data and the empirical data.

**Data**

In our paper, we evaluate our system model on both simulation dataset and historical stock data of Chinese A-share market.

- The simulation dataset: We simulate three hypothetical market sentiments (uptrend, downtrend, and flat) and one dataset under a mixture of different market sentiments by a zero intelligence agent-based model with the [codes](https://github.com/JackBenny39/pyziabm). If you would like to reproduce our trading system, you could download the simulation datasets [here](https://drive.google.com/drive/folders/1gQw7WtzuEdF2yMlgO9cC66brJ4MUtPl9?usp=sharing) and have a taste.
- The CS-20 dataset: [Benchmark dataset](https://github.com/hkgsas/LOB) is publicly available and downloaded.
- The proprietary dataset of CS-100: Provided by the Fintech company [TradeMaster](https://www.trademastertech.com).

**Code**

The codes written by Python are provided in the [fold](Code) for the simulation dataset under a mixture of different market sentiments.

- DCNN.py: Labelling, training and predicting.
- invest_strat_op.py: Trading strategy with optimization and without optimization.
