import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from scipy import io
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from network import LSTM
from utils import print_model_info, DataSet

# set config
OUTPUT_DIM = 2
SEED = 25
EPOCH_NUM = 10
DROPOUT = 0.5

# fix random seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def check_prediction_performance(result_dir, model_dir, area_list, model_list, test_period_list, period_info, mesh_ids, labels, features, retraining=False):

    # load hyperparameters
    hyperparams = pd.read_csv(model_dir + 'best_hyperparams.csv', header=None, index_col=0)

    UNIT_NUM = [int(hyperparams.loc['net__module__unit_num1']), int(hyperparams.loc['net__module__unit_num2']), int(hyperparams.loc['net__module__unit_num3']), int(hyperparams.loc['net__module__unit_num4'])]
    BATCH_SIZE = int(hyperparams.loc['net__batch_size'])
    LEARNING_RATE = float(hyperparams.loc['net__optimizer__lr'])
    WEIGHT_DECAY = float(hyperparams.loc['net__optimizer__weight_decay'])

    # buffer
    lstm_preds = np.zeros([mesh_ids.shape[0], len(test_period_list)]) # (N, P_test)
    lr_preds = np.zeros([mesh_ids.shape[0], len(test_period_list)])
    accs = np.zeros([len(test_period_list), len(area_list), len(model_list)]) # (P_test, N_area, N_model)
    recalls = np.zeros([len(test_period_list), len(area_list), len(model_list)])
    precs = np.zeros([len(test_period_list), len(area_list), len(model_list)])
    f1_scores = np.zeros([len(test_period_list), len(area_list), len(model_list)])

    counter = 0

    for test_period in test_period_list:
        print('==========================')
        print('test period: ', test_period)

        #####################################
        # set testing and training datasets #
        #####################################
        test_ind = period_info.index(test_period)
        #print('train period: ', period_info[:test_ind])
        #print('test period: ', period_info[test_ind])

        test_label = labels[:,test_ind]         # (N, )
        test_feature = features[:,test_ind,:,:] # (N, L, D)
        test_feature = test_feature.reshape(test_feature.shape[0], test_feature.shape[1], test_feature.shape[2]) # (N, L, D)

        train_labels = labels[:,:test_ind]          # (N, P_train)
        train_features = features[:,:test_ind,:,:]  # (N, P_train, L, D)

        # select data for balancing
        balanced_train_labels, balanced_train_features = [], []

        for area_ind in range(len(area_list)):
            #print('area: ', area_list[area_ind])

            # find mesh num of the area
            area_mesh_id = np.where((1000000*(area_ind+1)< mesh_ids) & (mesh_ids < 1000000 + 1000000*(area_ind+1)))
            area_train_labels = train_labels[area_mesh_id[0],:]
            area_train_features = train_features[area_mesh_id[0],:,:]
            #print('total label shape: ', area_train_labels.shape)

            # reshape labels and features
            area_train_labels = area_train_labels.reshape(area_train_labels.shape[0]*area_train_labels.shape[1])                                                                     # (N*P_train, )
            area_train_features = area_train_features.reshape(area_train_features.shape[0]*area_train_features.shape[1], area_train_features.shape[2], area_train_features.shape[3]) # (N*P_train, L, D)

            # find positive data where deforestation has occurred
            positive_ind = np.where(area_train_labels>0)
            positive_labels = area_train_labels[positive_ind[0]]
            positive_features = area_train_features[positive_ind[0],:,:]
            #print('positive label shape: ', positive_labels.shape)

            # extract as many negatives as positives
            negative_ind = np.where(area_train_labels==0)
            negative_ind = rng.choice(negative_ind[0], len(positive_ind[0]), replace=False, axis=0)
            negative_labels = area_train_labels[negative_ind]
            negative_features = area_train_features[negative_ind,:,:]
            #print('selected negative label shape: ', negative_labels.shape)

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

        balanced_train_features = balanced_train_features.astype(np.float32)
        balanced_train_labels = balanced_train_labels.astype(np.int64)

        # standardization
        dim2, dim3 = balanced_train_features.shape[1], balanced_train_features.shape[2] # L, D
        balanced_train_features = balanced_train_features.reshape(balanced_train_features.shape[0], dim2*dim3)
        test_feature = test_feature.reshape(test_feature.shape[0], dim2*dim3)

        mu_train = np.mean(balanced_train_features, axis=0)
        sigma_train = np.std(balanced_train_features, axis=0)
        
        balanced_train_features = (balanced_train_features - mu_train) / sigma_train
        test_feature = (test_feature - mu_train) / sigma_train
        #print('mean: ', np.mean(balanced_train_features, axis=0))
        #print('std: ', np.std(balanced_train_features, axis=0))

        """
        # min-max normalization
        min_train = np.min(balanced_train_features, axis=0)
        max_train = np.max(balanced_train_features, axis=0)
        balanced_train_features = (balanced_train_features - min_train) / (max_train - min_train)
        test_feature = (test_feature - min_train) / (max_train - min_train)
        #print('min: ', np.min(balanced_train_features, axis=0))
        #print('max: ', np.max(balanced_train_features, axis=0))
        """

        balanced_train_features_2d = np.copy(balanced_train_features)
        test_feature_2d = np.copy(test_feature)

        balanced_train_features = balanced_train_features.reshape(balanced_train_features.shape[0], dim2, dim3)
        test_feature = test_feature.reshape(test_feature.shape[0], dim2, dim3)

        ############################################
        # construct network and logistic regressor #
        ############################################
        net = LSTM(balanced_train_features.shape[2], UNIT_NUM, OUTPUT_DIM, DROPOUT)
        #print_model_info(net)

        if not os.path.exists(model_dir+"LSTM_"+test_period+".pytorch") or retraining:
            print('--------------')
            print('start training')
            print('--------------')
            lr_model = LogisticRegression(penalty='none')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #print("gpu or cpu: ", device)

            ##########################################
            # set DataLoader for full-batch learning #
            ##########################################
            train_dataset = DataSet(balanced_train_features, balanced_train_labels, transform=transforms.ToTensor())
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
                )

            ##################
            # start training #
            ##################
            net.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            net.train()

            minibatch_loss_list = []
            epoch_loss_list = []

            for epoch in range(EPOCH_NUM):
                minibatch_loss = 0

                with tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)) as pbar:
                    pbar.set_description("[Epoch %d/%d]" % (epoch + 1, EPOCH_NUM))
                    for count, item in pbar:
                        mini_x, mini_y = item
                        mini_x, mini_y = mini_x.to(device), mini_y.to(device)
                        mini_y = mini_y.flatten()
                        mini_y = torch.tensor(mini_y, dtype=torch.int64)

                        optimizer.zero_grad()
                        outputs = net(mini_x)
                        loss = criterion(outputs, mini_y)
                        loss.backward()
                        optimizer.step()

                        minibatch_loss += loss.detach().cpu().numpy()
                        minibatch_loss_list.append(loss.detach().cpu().numpy())
                    epoch_loss_list.append(minibatch_loss / count)

            # draw the graph for loss by batch and epoch
            plt.plot(minibatch_loss_list)
            plt.plot(epoch_loss_list)

            plt.figure(figsize=[12,4])
            plt.subplot(1,2,1)
            plt.plot(minibatch_loss_list)
            plt.subplot(1,2,2)
            plt.plot(epoch_loss_list, "o--")
            plt.tight_layout()
            plt.savefig(model_dir+test_period+"_loss.jpg", dpi=300)
            plt.clf()
            plt.close()

            lr_model.fit(balanced_train_features_2d, balanced_train_labels)

            # save models
            torch.save(net.state_dict(), model_dir+"LSTM_"+test_period+".pytorch")
            pickle.dump(lr_model, open(model_dir+"LR_"+test_period+".sav", "wb"))
        else:
            print('-----------------')
            print('already trained')
            print('load weights')
            print('-----------------')

            net.load_state_dict(torch.load(model_dir+"LSTM_"+test_period+".pytorch"))
            lr_model = pickle.load(open(model_dir+"LR_"+test_period+".sav", "rb"))

        ############################################
        # evaluate model performance for each area #
        ############################################
        # set eval mode
        net.eval()
        device_cpu = torch.device("cpu")
        net.to(device_cpu)

        for area_ind in range(len(area_list)):
            print('area: ', area_list[area_ind])

            # find mesh num of the area
            area_mesh_id = np.where((1000000*(area_ind+1)< mesh_ids) & (mesh_ids < 1000000 + 1000000*(area_ind+1)))
            area_test_label = test_label[area_mesh_id[0]]
            area_test_feature = test_feature[area_mesh_id[0],:,:]
            area_test_feature_2d = test_feature_2d[area_mesh_id[0],:]

            test_y = area_test_label
            test_x = torch.tensor(area_test_feature).float()
            test_x_2d = np.copy(area_test_feature_2d)

            # deforestation prediction
            lstm_pred = net(test_x)
            lstm_pred = np.argmax(lstm_pred.detach().cpu().numpy(),axis=1)

            lr_pred = lr_model.predict(test_x_2d)

            # check performance in some indecies
            accs[counter,area_ind,0] = accuracy_score(test_y, lstm_pred)
            recalls[counter,area_ind,0] = recall_score(test_y, lstm_pred)#, average="macro")
            precs[counter,area_ind,0] = precision_score(test_y, lstm_pred)#, average="macro")
            f1_scores[counter,area_ind,0] = f1_score(test_y, lstm_pred)#, average="macro")

            accs[counter,area_ind,1] = accuracy_score(test_y, lr_pred)
            recalls[counter,area_ind,1] = recall_score(test_y, lr_pred)#, average="macro")
            precs[counter,area_ind,1] = precision_score(test_y, lr_pred)#, average="macro")
            f1_scores[counter,area_ind,1] = f1_score(test_y, lr_pred)#, average="macro")

            print('LSTM accuracy: ', accs[counter,area_ind,0])
            print('LSTM recall: ', recalls[counter,area_ind,0])
            print('LSTM precision: ', precs[counter,area_ind,0])
            print('LSTM f1 score: ', f1_scores[counter,area_ind,0])

            print('LR accuracy: ', accs[counter,area_ind,1])
            print('LR recall: ', recalls[counter,area_ind,1])
            print('LR precision: ', precs[counter,area_ind,1])
            print('LR f1 score: ', f1_scores[counter,area_ind,1])

            # concatenate preds
            if area_ind==0:
                temp_lstm_preds = lstm_pred
                temp_lr_preds = lr_pred
            else:
                temp_lstm_preds = np.append(temp_lstm_preds, lstm_pred, axis=0)
                temp_lr_preds = np.append(temp_lr_preds, lr_pred, axis=0)

        lstm_preds[:,counter] = temp_lstm_preds
        lr_preds[:,counter] = temp_lr_preds
        counter = counter + 1

    # save results
    mesh_ids_dammy = mesh_ids[:,np.newaxis]   
    lstm_preds = np.concatenate([mesh_ids_dammy, lstm_preds], axis=1)
    lr_preds = np.concatenate([mesh_ids_dammy, lr_preds], axis=1)
    np.savez(result_dir+'results', lstm_preds, lr_preds, accs, recalls, precs, f1_scores)
    io.savemat(result_dir+'results.mat', {"lstm_preds": lstm_preds, "lr_preds": lr_preds,
                                          "accs": accs, "recalls": recalls, "precs": precs, "f1_scores": f1_scores})


def summary_table(result_dir, area_list, model_list, test_period_list, eval_period_list):
    
    # load results
    npz = np.load(result_dir+"results.npz")
    accs, recalls, precs, f1_scores = npz['arr_2'], npz['arr_3'], npz['arr_4'], npz['arr_5']

    eval_ind = []
    for i in range(len(eval_period_list)):
        eval_ind.append(test_period_list.index(eval_period_list[i]))

    # make a summary table
    print('==============')
    print('summary table')
    print('==============')
    for area_ind in range(len(area_list)):
        print(area_list[area_ind])
        print('           acc    recall   precision  f1 score')
        for model_ind in range(len(model_list)):
            print(model_list[model_ind]+'_mean:  {:.3f}   {:.3f}     {:.3f}       {:.3f}'.format(np.mean(accs[eval_ind,area_ind,model_ind]), np.mean(recalls[eval_ind,area_ind,model_ind]), np.mean(precs[eval_ind,area_ind,model_ind]), np.mean(f1_scores[eval_ind,area_ind,model_ind])))
        for model_ind in range(len(model_list)):
            print(model_list[model_ind]+'_max:   {:.3f}   {:.3f}     {:.3f}       {:.3f}'.format(np.max(accs[eval_ind,area_ind,model_ind]), np.max(recalls[eval_ind,area_ind,model_ind]), np.max(precs[eval_ind,area_ind,model_ind]), np.max(f1_scores[eval_ind,area_ind,model_ind])))
        print('\n')


def visualize_prediction_performance(result_dir, area_list, test_period_list):
    
    # load results
    npz = np.load(result_dir+"results.npz")
    accs, recalls, precs, f1_scores = npz['arr_2'], npz['arr_3'], npz['arr_4'], npz['arr_5']
     
    # drow mean time-series plot
    pd_lstm_accs = pd.DataFrame({"Val": np.mean(accs[:,:,0], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lstm_recalls = pd.DataFrame({"Val": np.mean(recalls[:,:,0], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lstm_precs = pd.DataFrame({"Val": np.mean(precs[:,:,0], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lstm_f1_scores = pd.DataFrame({"Val": np.mean(f1_scores[:,:,0], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

    pd_lr_accs = pd.DataFrame({"Val": np.mean(accs[:,:,1], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lr_recalls = pd.DataFrame({"Val": np.mean(recalls[:,:,1], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lr_precs = pd.DataFrame({"Val": np.mean(precs[:,:,1], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
    pd_lr_f1_scores = pd.DataFrame({"Val": np.mean(f1_scores[:,:,1], axis=1)}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

    fig = plt.figure(figsize=(15,10))
    plt.rcParams['font.size'] = 14

    rain_start_datetime_2021 = datetime.datetime(2021, 10,1)
    rain_end_datetime_2021 = datetime.datetime(2022, 3,1)
    rain_start_datetime_2022 = datetime.datetime(2022, 10,1)
    rain_end_datetime_2022 = datetime.datetime(2023, 3,1)

    for fig_ind in range(4):
        ax = fig.add_subplot(2,2,fig_ind+1)
        ax.set_ylim(0, 1)
        if fig_ind == 0: # acc
            plt.plot(pd_lstm_accs, label="LSTM")
            plt.plot(pd_lr_accs, label="LR")
        elif fig_ind == 1: # recall
            plt.plot(pd_lstm_recalls, label="LSTM")
            plt.plot(pd_lr_recalls, label="LR")
        elif fig_ind == 2: # prec
            plt.plot(pd_lstm_precs, label="LSTM")
            plt.plot(pd_lr_precs, label="LR")
        elif fig_ind == 3: # f1 score
            plt.plot(pd_lstm_f1_scores, label="LSTM")
            plt.plot(pd_lr_f1_scores, label="LR")
        plt.gcf().autofmt_xdate()
        plt.axvspan(rain_start_datetime_2021, rain_end_datetime_2021, color="gray", alpha=0.3)
        plt.axvspan(rain_start_datetime_2022, rain_end_datetime_2022, color="gray", alpha=0.3)
        plt.grid()
        if fig_ind == 0:
            plt.legend(loc="upper right")
            plt.title("Monthly averaged acc")
        elif fig_ind == 1:
            plt.legend(loc="upper left")
            plt.title("Monthly averaged recall")
        elif fig_ind == 2:
            plt.legend(loc="lower left")
            plt.title("Monthly averaged specs")
        elif fig_ind == 3:
            plt.legend(loc="lower right")
            plt.title("Monthly averaged f1 score")
        
    fig.tight_layout()
    fig.savefig(result_dir+"average_result.jpg", dpi=300)
    fig.clf()

    # drow time-series plot for each area
    for area_ind in range(len(area_list)):
        pd_lstm_accs = pd.DataFrame({"Val": accs[:,area_ind,0]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lstm_recalls = pd.DataFrame({"Val": recalls[:,area_ind,0]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lstm_precs = pd.DataFrame({"Val": precs[:,area_ind,0]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lstm_f1_scores = pd.DataFrame({"Val": f1_scores[:,area_ind,0]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

        pd_lr_accs = pd.DataFrame({"Val": accs[:,area_ind,1]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lr_recalls = pd.DataFrame({"Val": recalls[:,area_ind,1]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lr_precs = pd.DataFrame({"Val": precs[:,area_ind,1]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))
        pd_lr_f1_scores = pd.DataFrame({"Val": f1_scores[:,area_ind,1]}, index=pd.Index(pd.date_range(test_period_list[0][0:4]+"/"+test_period_list[0][4:6]+"/01", periods=len(test_period_list), freq='MS'), name="Data"))

        fig = plt.figure(figsize=(15,10))
        plt.rcParams['font.size'] = 14

        for fig_ind in range(4):
            ax = fig.add_subplot(2,2,fig_ind+1)
            ax.set_ylim(0, 1)
            if fig_ind == 0: # acc
                plt.plot(pd_lstm_accs, label="LSTM")
                plt.plot(pd_lr_accs, label="LR")
            elif fig_ind == 1: # recall
                plt.plot(pd_lstm_recalls, label="LSTM")
                plt.plot(pd_lr_recalls, label="LR")
            elif fig_ind == 2: # prec
                plt.plot(pd_lstm_precs, label="LSTM")
                plt.plot(pd_lr_precs, label="LR")
            elif fig_ind == 3: # f1 score
                plt.plot(pd_lstm_f1_scores, label="LSTM")
                plt.plot(pd_lr_f1_scores, label="LR")
            plt.gcf().autofmt_xdate()
            plt.axvspan(rain_start_datetime_2021, rain_end_datetime_2021, color="gray", alpha=0.3)
            plt.axvspan(rain_start_datetime_2022, rain_end_datetime_2022, color="gray", alpha=0.3)
            plt.grid()
            if fig_ind == 0:
                plt.legend(loc="upper right")
                plt.title("Monthly averaged acc")
            elif fig_ind == 1:
                plt.legend(loc="upper left")
                plt.title("Monthly averaged recall")
            elif fig_ind == 2:
                plt.legend(loc="lower left")
                plt.title("Monthly averaged specs")
            elif fig_ind == 3:
                plt.legend(loc="lower right")
                plt.title("Monthly averaged f1 score")
            
        fig.tight_layout()
        fig.savefig(result_dir+ area_list[area_ind] + "_result.jpg", dpi=300)
        fig.clf()


# drow TP/TN/FP/FN plot for each area
def result(x):
    if x.actual == 1 and x.pred == 1:
        return "TP" # True Postive
    elif x.actual == 1 and x.pred == 0:
        return "FN" # False Negative
    elif x.actual == 0 and x.pred == 1:
        return "FP" # False Negative
    else:
        return "TN" # True Negative


def map_visualize_prediction_performance(data_dir, result_dir, area_list, test_period_list, mesh_ids):

    for area_ind in range(len(area_list)):

        ##########################
        # get objective variable #
        ##########################
        # load target area
        area = area_list[area_ind]
        true_defo = pd.read_csv(data_dir+"/"+area+"/label/event.csv", index_col=0, header=0)
        npz = np.load(result_dir+"results.npz")
        pred_defo_LSTM = npz['arr_0']

        area_mesh_id = np.where((1000000*(area_ind+1)< mesh_ids) & (mesh_ids < 1000000 + 1000000*(area_ind+1)))
        pred_defo_LSTM = pred_defo_LSTM[area_mesh_id[0]]
        
        ##########################
        # get mesh geojson file  #
        ##########################
        gdf_mesh = gpd.read_file(data_dir+"/"+area+"/label/1km_mesh.geojson")
        gdf_mesh = gdf_mesh.to_crs(4326)

        #################
        # export figure #
        #################
        for period_ind in range(len(test_period_list)):
            period = test_period_list[period_ind]

            # deforestation data
            df_actual = true_defo[[period]]
            df_actual = df_actual.set_axis(['actual'], axis=1)
            df_actual = df_actual.reset_index()

            # prediction data
            df_result = pd.DataFrame(pred_defo_LSTM)
            df_result=df_result.set_index([0]) 
            df_result.columns = test_period_list
            df_result = df_result[[period]]
            df_result = df_result.set_axis(['pred'], axis=1)
            df_result = df_result.reset_index()

            gdf_result_mesh_id = gpd.GeoDataFrame(pd.concat([df_actual,df_result,gdf_mesh], axis=1),crs=gdf_mesh.crs)
            gdf_result_mesh_id["result"]= np.nan
            gdf_result_mesh_id["result"] = gdf_result_mesh_id.apply(result, axis=1)

            palette = {'TP': 'blue','FP': 'red','FN': 'green','TN': 'grey'}
            gdf_result_mesh_id["Colors"] = gdf_result_mesh_id["result"].map(palette)
            
            custom_points = [Line2D([0], [0], marker="s", linestyle="none", markersize=10, color=color) for color in palette.values()]
            ax = gdf_result_mesh_id.plot(color = gdf_result_mesh_id["Colors"])
            leg_points = ax.legend(custom_points, palette.keys(),bbox_to_anchor=(1, 1), loc='upper left')
            ax.add_artist(leg_points)
            
            # add title
            #plt.title(target_area +" "+ target_pred +" "+ target_period, fontsize=18)
            plt.subplots_adjust(right=0.8)
            plt.title(area + "_"+ period)
            plt.tight_layout()        
            plt.savefig(result_dir+"/"+area+"_"+period+".png",dpi=400, facecolor="white", bbox_inches='tight') 