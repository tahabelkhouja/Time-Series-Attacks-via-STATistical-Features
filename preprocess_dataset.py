# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:54:10 2019

@author: BkTaha
"""
import os
import json
import numpy as np
import pickle as pkl
import pandas as pd

from scipy.io import arff
from absl import app, flags

FLAGS = flags.FLAGS

def load_ucr_data(dataset_name, parent_file):
    #Extract Data Dimensions
    dim_df = pd.read_csv("DatasetUVDimensions.csv")
    ds_idx = dim_df[dim_df["problem"]==dataset_name].index[0]
    ds_trn_size = int(dim_df.at[ds_idx, "numTrainCases"])
    ds_tst_size = int(dim_df.at[ds_idx, "numTestCases"])
    ds_seg_size = int(dim_df.at[ds_idx, "seriesLength"])
    #Extract TrainData
    X_train = np.zeros((ds_trn_size, 1, ds_seg_size, 1))
    data_file = parent_file+"/"+dataset_name+"_TRAIN.arff"
    data, meta = arff.loadarff(data_file)
    train_data = data[meta.names()[:-1]] #everything but the last column
    train_data = np.array(train_data.tolist())
    X_train[:,:,:,0] = train_data.reshape((ds_trn_size, 1,ds_seg_size))
    #Extract TrainLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    train_lbl = data[meta.names()[-1]] #LastColumn
    train_lbl = np.array([ss.decode('ascii') for ss in train_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(train_lbl))):
        labels[y]=i
    y_train = np.array([labels[y] for y in train_lbl])
    
    #Extract TestData
    X_test = np.zeros((ds_tst_size, 1, ds_seg_size, 1))
    data_file = parent_file+"/"+dataset_name+"_TEST.arff"
    data, meta = arff.loadarff(data_file)
    test_data = data[meta.names()[:-1]] #everything but the last column
    test_data = np.array(test_data.tolist())
    X_test[:,:,:,0] = test_data.reshape((ds_tst_size, 1,ds_seg_size))
    #Extract TestLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    test_lbl = data[meta.names()[-1]] #LastColumn
    test_lbl = np.array([ss.decode('ascii') for ss in test_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(test_lbl))):
        labels[y]=i
    y_test = np.array([labels[y] for y in test_lbl])
    
    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]
    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]
    return X_train, y_train, X_test, y_test

def load_mv_ucr_data(dataset_name, parent_file):
    #Extract Data Dimensions
    dim_df = pd.read_csv("DatasetMVDimensions.csv")
    ds_idx = dim_df[dim_df["Problem"]==dataset_name].index[0]
    ds_trn_size = int(dim_df.at[ds_idx, "TrainSize"])
    ds_tst_size = int(dim_df.at[ds_idx, "TestSize"])
    ds_channel_nb = int(dim_df.at[ds_idx, "NumDimensions"])
    ds_seg_size = int(dim_df.at[ds_idx, "SeriesLength"])
    
    
    
    #Extract TrainData
    X_train = np.zeros((ds_trn_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        data_file = parent_file+"/"+dataset_name+"Dimension"+str(ch+1)+"_TRAIN.arff"
        data, meta = arff.loadarff(data_file)
        train_data = data[meta.names()[:-1]] #everything but the last column
        train_data = np.array(train_data.tolist())
        X_train[:,:,:,ch] = train_data.reshape((ds_trn_size, 1,ds_seg_size))
    #Extract TrainLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    train_lbl = data[meta.names()[-1]] #LastColumn
    train_lbl = np.array([ss.decode('ascii') for ss in train_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(train_lbl))):
        labels[y]=i
    y_train = np.array([labels[y] for y in train_lbl])
    
    #Extract TestData
    X_test = np.zeros((ds_tst_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        data_file = parent_file+"/"+dataset_name+"Dimension"+str(ch+1)+"_TEST.arff"
        data, meta = arff.loadarff(data_file)
        test_data = data[meta.names()[:-1]] #everything but the last column
        test_data = np.array(test_data.tolist())
        X_test[:,:,:,ch] = test_data.reshape((ds_tst_size, 1,ds_seg_size))
    #Extract TestLabels
    data, meta = arff.loadarff(open(data_file, "r"))
    test_lbl = data[meta.names()[-1]] #LastColumn
    test_lbl = np.array([ss.decode('ascii') for ss in test_lbl])
    labels = {}
    for i, y in enumerate(np.sort(np.unique(test_lbl))):
        labels[y]=i
    y_test = np.array([labels[y] for y in test_lbl])
    
    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]
    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]
    return X_train, y_train, X_test, y_test

    
def main(argv):
    dataset_zip_directory = "UCRDatasets/{}".format(FLAGS.dataset_name)
    try:
        os.makedirs("Dataset")
    except FileExistsError:
        pass
    if FLAGS.multivariate:
        X_train, y_train, X_test, y_test = load_mv_ucr_data(FLAGS.dataset_name, dataset_zip_directory)
    else:
        X_train, y_train, X_test, y_test = load_ucr_data(FLAGS.dataset_name, dataset_zip_directory)
        
    pkl.dump([X_train, y_train, X_test, y_test], open("Dataset/"+FLAGS.dataset_name+".pkl", "wb")) 
    with open('datasets_parameters.json') as jf:
        info = json.load(jf)
    info[FLAGS.dataset_name]={
            "path": "Dataset/"+FLAGS.dataset_name+".pkl",
        	 "SEG_SIZE": X_train.shape[2], 
        	 "CHANNEL_NB": X_train.shape[3],
        	 "CLASS_NB": len(np.unique(y_train))
        }
    with open('datasets_parameters.json', 'w') as jf:
        json.dump(info, jf, indent=2)
        
if __name__=="__main__":
    flags.DEFINE_string('dataset_name', 'SyntheticControl ', 'Dataset name')
    flags.DEFINE_boolean('multivariate', False, 'Dataset Is multivariate')
    app.run(main)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    