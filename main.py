## Main script for deforestation prediction
# checked OS: Windows 10 and Ubuntu 20.04

# Suguru Kanoga, 23-Jan.-2024
#  Artificial Intelligence Research Center, National Institute of Advanced
#  Industrial Science and Technology (AIST)
#  E-mail: s.kanouga@aist.go.jp

###################
# import packages #
###################
# public
import platform
import os
import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# private
from preprocessing import make_dataset
from hyperparameter_search import grid_search
from evaluate import summary_table, check_prediction_performance, visualize_prediction_performance, map_visualize_prediction_performance

############
# check OS #
############
os_name = platform.system()
if os_name == 'Windows':
    sl = '\\'
else:
    sl = '//'

##############
# set config #
##############
main_dir = os.getcwd()
data_dir = main_dir+sl+"data"
model_dir = main_dir+sl+"model"
result_dir = main_dir+sl+"result"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

area_list = ["porto_velho", "humaita", "altamira", # small-scale logging by poor farmers
             "vista_alegre_do_abuna","novo_progresso","sao_felix_do_xingu", # large-scale logging by large landowners
             "S6W57", "S7W57"] # mining development
feature_list = ["event", "square_meter"]
model_list = ["LSTM", "LR"]
test_period_list = ["202110","202111","202112",
                    "202201","202202","202203","202204","202205","202206","202207","202208","202209","202210","202211","202212",
                    "202301","202302","202303","202304","202305","202306","202307","202308","202309"]
eval_period_list = ["202206","202207","202208","202209",
                    "202306","202307","202308","202309"]

# dataset
DURATION = 12
REMAKING = False

# neural network
RETRAINING = True

#################
# preprocessing #
#################
# make dataset
if not os.path.exists(data_dir+sl+"dataset.npz") or REMAKING:
    make_dataset(sl, area_list, feature_list, data_dir, DURATION, objective_variable="event")

# load dataset
npz = np.load(data_dir+sl+"dataset.npz")
period_info, mesh_ids, labels, features = npz['arr_0'], npz['arr_1'], npz['arr_2'], npz['arr_3']
period_info = list(period_info)

print('areas: ', area_list)
print('preriod of dataset: ', period_info)
print('length of period: ', len(period_info))
print('total mesh ids: ', mesh_ids.shape)
print('shape of total labels: ', labels.shape)
print('shape of total features: ', features.shape)

#########################
# hyperparameter search #
#########################
if os.path.exists(model_dir+sl+"best_hyperparams.csv"):
    print('-----------------')
    print('already searched')
    print('-----------------')
else:
    print('-----------------')
    print('start hyperparameter search')
    print('-----------------')
    grid_search(model_dir+sl, area_list, test_period_list, period_info, mesh_ids, labels, features)

##############
# evaluation #
##############
if os.path.exists(result_dir+sl+"results.npz") and RETRAINING==False:
    print('-----------------')
    print('already evaluated')
    print('-----------------')
else:
    print('-----------------')
    print('start evaluation')
    print('-----------------')
    check_prediction_performance(result_dir+sl, model_dir+sl, area_list, model_list, test_period_list, period_info, mesh_ids, labels, features, retraining=RETRAINING)

#################
# visualization #
#################
summary_table(result_dir+sl, area_list, model_list, test_period_list, eval_period_list)
visualize_prediction_performance(result_dir+sl, area_list, test_period_list)
map_visualize_prediction_performance(data_dir+sl, result_dir+sl, area_list, test_period_list, mesh_ids)