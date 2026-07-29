# Impact of Training Data Contamination on Unsupervised Time Series Anomaly Detection

## Overview
The growth of IoT sensor data facilitates data-driven monitoring and unsupervised time series anomaly detection in distributed and cyber-physical systems. In addition to classical statistical approaches, reconstruction-based neural networks are a promising method to detect anomalies in unlabeled time series. However, reconstruction-based methods rely on a strong assumption which is often neglected in practice: Training data must not be contaminated. Particularly in large datasets, guaranteeing completely anomaly-free training data is difficult. Thus, this study investigates how contamination in the training set influences anomaly detection performance quantified by the point-wise F1-score. We evaluate increasing levels of training data contamination using three state-of-the-art models - TimesNet, USAD and TranAD - together with a Transformer block, Autoencoder, iForest and z-scores as baselines. The findings reveal that several leading models are highly sensitive to even low levels of contamination. All reconstruction-based neural networks except USAD halved their F1-score obtained with anomaly-free training data on at least one dataset at training data contamination levels between 2.5\% and 5\%. USAD halved its clean-data performance at a contamination level of 30\% for the first time. Due to the ubiquity of contaminated real-world datasets and the low contamination-resilience demonstrated by state-of-the-art models, we firstly advocate to evaluate future models also for contamination-resilience and secondly to strive for architectures which focus on robustness to contamination.

## Usage
This section describes how to reproduce the results.

1. We use Python v3.14.4 and PyTorch v2.11.0. Install remaining dependencies: `python3 -m pip install -r requirements.txt`.

2. Download and unzip datasets from https://www.thedatum.org/datasets/TSB-AD-U.zip and https://www.thedatum.org/datasets/TSB-AD-M.zip. The downloaded datasets are expected to reside in folder `./dataset/raw/TSB-AD-U` and `./dataset/raw/TSB-AD-M`, respectively.

3. Create dataset splits, i.e. train-val-test-splits, using the downloaded datasets and our proposed training data contamination protocol: `./create_TSB-AD.sh`. The resulting dataset splits reside in `dataset/TSB-AD-Anomaly`.

4. Run experiments using the previously generated dataset splits: `python3 run_experiment.py experiments/tsb`.

5. Run evaluation script to generate tables and plots: `python3 run_summary_tsb2.py --ds-root .`.

6. Investigate results in `results/TSB-AD-Anomaly-summary`.
