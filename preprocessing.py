import pandas as pd
import numpy as np
#from scipy import io

# get period information, mesh ids, response variable (label), and explanatory variables (features)
def make_dataset(sl, area_list, feature_list, data_dir, DURATION, objective_variable):
    
    """
    N: total mesh num
    P: overall period length
    L: sequence length for prediction (=DURATION)
    D: feature dimension

    period_info: list        (P-L, )
    mesh_ids   : numpy array (N, )
    labels     : numpy array (N, P-L)
    features   : numpy array (N, P-L, L, D)
    """

    for area_ind in range(len(area_list)):

        # get response variable
        temp_label = pd.read_csv(data_dir+sl+area_list[area_ind]+sl+'label'+sl+objective_variable+'.csv', header=0)
        area_label = np.array(temp_label)
        area_mesh_id = area_label[:,0].astype(int)
        area_label = np.delete(area_label, 0, 1) # remove id info
        area_label = np.delete(area_label, np.s_[:DURATION], 1) # remove onset period

        # get explanatory variables
        for feature_ind in range(len(feature_list)):
            temp_feature = pd.read_csv(data_dir+sl+area_list[area_ind]+sl+'feature'+sl+feature_list[feature_ind]+'.csv', header=0)
            temp_feature = np.array(temp_feature)
            temp_feature = np.delete(temp_feature, 0, 1) # remove id info
            
            for seq_ind in range(area_label.shape[1]):
                temp_seq = temp_feature[:,seq_ind:seq_ind+DURATION]
                temp_seq = np.expand_dims(temp_seq, 1)
                temp_seq = np.expand_dims(temp_seq, -1)

                if seq_ind==0:
                    temp_seq_feature = temp_seq
                else:
                    temp_seq_feature = np.concatenate([temp_seq_feature, temp_seq], axis=1)
            
            if feature_ind==0:
                area_feature = temp_seq_feature
            else:
                area_feature = np.concatenate([area_feature, temp_seq_feature], axis=3)
            
        # respectively combine response/explanatory variables from different areas
        if area_ind==0:
            period_info = list(temp_label.columns.values)
            period_info.remove('id')
            del period_info[:DURATION]
            mesh_ids = area_mesh_id
            labels = area_label
            features = area_feature
        else:
            mesh_ids = np.concatenate([mesh_ids, area_mesh_id], axis=0)
            labels = np.concatenate([labels, area_label], axis=0)
            features = np.concatenate([features, area_feature], axis=0)
        
    # save data
    np.savez(data_dir+sl+'dataset', period_info, mesh_ids, labels, features)
    #io.savemat(data_dir+sl+'dataset.mat', {"period_info": period_info, "mesh_ids": mesh_ids, "labels": labels, "features": features})