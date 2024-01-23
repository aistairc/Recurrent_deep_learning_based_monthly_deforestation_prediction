import os
import random
import numpy as np
import pandas as pd
import torch
import csv
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from network import LSTM_layer4
#from utils import print_model_info

# set config
SEED = 25
CV_NUM = 2
EPOCH_NUM = 10
WORKER_NUM = 30 # depends on the machine
SCORE = 'neg_log_loss'

# fix random seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# hyperparameter candidates
unit_num_range = [16, 32, 64, 128] 
batch_size_range = [64, 128, 256]
lr_range =  [0.0001, 0.001, 0.01, 0.1]
weight_decay_range = [0.0001, 0.001, 0.01, 0.1]

def grid_search(model_dir, area_list, test_period_list, period_info, mesh_ids, labels, features):
    
    #########################
    # set training datasets #
    #########################
    # find index of first testing period in source dataset
    first_test_ind = period_info.index(test_period_list[0])

    # set training dataset
    train_labels = labels[:,:first_test_ind]         # (N, P_train)
    train_features = features[:,:first_test_ind,:,:] # (N, P_train, L, D)
    #print('train_features shape: ', train_features.shape)

    # select data for balancing
    balanced_train_labels, balanced_train_features = [], []

    for area_ind in range(len(area_list)):

        # find mesh num of the area
        area_mesh_id = np.where((1000000*(area_ind+1)< mesh_ids) & (mesh_ids < 1000000 + 1000000*(area_ind+1)))
        area_train_labels = train_labels[area_mesh_id[0],:]
        area_train_features = train_features[area_mesh_id[0],:,:]

        # reshape labels and features
        area_train_labels = area_train_labels.reshape(area_train_labels.shape[0]*area_train_labels.shape[1])                                                                     # (N*P_train, )
        area_train_features = area_train_features.reshape(area_train_features.shape[0]*area_train_features.shape[1], area_train_features.shape[2], area_train_features.shape[3]) # (N*P_train, L, D)

        # find positive data where deforestation has occurred
        positive_ind = np.where(area_train_labels>0)
        positive_labels = area_train_labels[positive_ind[0]]
        positive_features = area_train_features[positive_ind[0],:,:]

        # extract as many negatives as positives
        negative_ind = np.where(area_train_labels==0)
        negative_ind = rng.choice(negative_ind[0], len(positive_ind[0]), replace=False, axis=0)
        negative_labels = area_train_labels[negative_ind]
        negative_features = area_train_features[negative_ind,:,:]

        temp_labels = np.concatenate([positive_labels, negative_labels], axis=0)
        temp_features = np.concatenate([positive_features, negative_features], axis=0)

        # concatenate labels and feats
        if area_ind==0:
            balanced_train_labels = temp_labels
            balanced_train_features = temp_features
        else:
            balanced_train_labels = np.append(balanced_train_labels, temp_labels, axis=0)
            balanced_train_features = np.append(balanced_train_features, temp_features, axis=0)

    print('balanced_train_labels: ', balanced_train_labels.shape)
    print('balanced_train_features: ', balanced_train_features.shape)

    X = balanced_train_features.astype(np.float32)
    y = balanced_train_labels.astype(np.int64)

    # standardization
    dim1, dim2, dim3 = X.shape[0], X.shape[1], X.shape[2]
    X = X.reshape(dim1, dim2*dim3)  # (N*P_train, L*D)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = X.reshape(dim1, dim2, dim3) # (N*P_train, L, D)

    #####################
    # construct network #
    #####################
    MyNet = LSTM_layer4()
    #print_model_info(MyNet)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("gpu or cpu: ", device)

    ###############
    # grid search #
    ###############
    param_grid = [{'net__module__unit_num1': [unit_num1], 'net__module__unit_num2': [unit_num2], 'net__module__unit_num3': [unit_num3], 'net__module__unit_num4': [unit_num4], 'net__batch_size': [batch_size], 'net__optimizer__lr': [lr],  'net__optimizer__weight_decay': [weight_decay]}
                   for unit_num1 in unit_num_range for unit_num2 in unit_num_range if (unit_num2 <= unit_num1) for unit_num3 in unit_num_range if (unit_num3 <= unit_num2) for unit_num4 in unit_num_range if (unit_num4 <= unit_num3) for batch_size in batch_size_range for lr in lr_range for weight_decay in weight_decay_range]
   
    net = NeuralNetClassifier(
        MyNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        max_epochs=EPOCH_NUM,
        device=device,
        verbose=False
    ) 

    pipe = Pipeline([
        ('net', net),
    ])

    grid = GridSearchCV(pipe, param_grid, scoring=SCORE, verbose=1, cv=KFold(n_splits=CV_NUM,shuffle=False), n_jobs=WORKER_NUM)
    grid_result = grid.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))

    # find best hyperparameters
    best_score = [grid_result.best_score_]
    best_hyperparams = grid_result.best_params_

    print('best score: ', best_score)
    print('best hyperparams: ', best_hyperparams)
    
    # save results of hyperparameter search
    grid_result = pd.DataFrame.from_dict(grid_result.cv_results_)
    grid_result.to_csv(model_dir+'grid_result.csv')

    # save hyperparameters
    with open(model_dir+'best_hyperparams.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        for k, v in best_hyperparams.items():
            writer.writerow([k, v])