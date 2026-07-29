#!/bin/bash
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-U -n 8 --split 0.7 0.15 0.15
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-M -n 8 --split 0.7 0.15 0.15

python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.0/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.0
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.025/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.025
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.05/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.05
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.075/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.075
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.1/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.1
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.2/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.2
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.3/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.3
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-U -o dataset/TSB-AD-Anomaly/TSB-AD-U/0.4/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.4

python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.0/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.0
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.025/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.025
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.05/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.05
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.075/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.075
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.1/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.1
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.2/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.2
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.3/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.3
python3 data/preprocess_tsb.py -i dataset/raw/TSB-AD-M -o dataset/TSB-AD-Anomaly/TSB-AD-M/0.4/ -n 8 --split 0.7 0.15 0.15 --anomaly-ratio-train 0.4
