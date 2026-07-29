import numpy as np
import pandas as pd


def msl():
    print('MSL')
    print(np.load('../dataset/MSL/MSL_train.npy').shape)
    print(np.load('../dataset/MSL/MSL_test.npy').shape)
    print(np.load('../dataset/MSL/MSL_test_label.npy').shape)


def psm():
    print('PSM')
    print(pd.read_csv('../dataset/PSM/train.csv').shape)
    print(pd.read_csv('../dataset/PSM/test.csv').shape)
    print(pd.read_csv('../dataset/PSM/test_label.csv').shape)


def smap():
    print('SMAP')
    print(np.load('../dataset/SMAP/SMAP_train.npy').shape)
    print(np.load('../dataset/SMAP/SMAP_test.npy').shape)
    print(np.load('../dataset/SMAP/SMAP_test_label.npy').shape)


def smd():
    print('SMD')
    print(np.load('../dataset/SMD/SMD_train.npy').shape)
    print(np.load('../dataset/SMD/SMD_test.npy').shape)
    print(np.load('../dataset/SMD/SMD_test_label.npy').shape)


if __name__ == '__main__':
    msl()
    psm()
    smap()
    smd()
