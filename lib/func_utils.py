
import os
import pandas as pd
import numpy as np

def compare_pdkeys_list(dataframe, list_keys):
    '''  Check with elements of the list are actual keys in the dataframe '''
    keys = dataframe.keys()
    avail_keys = [element for element in list_keys if element in keys]
    return avail_keys


def concatenate_processings(MatMat, EucMat, dict_df, config):
    MatConc = []; EucConc = []; df_conc = {}
    for p,proc in enumerate(config["Parameters"]["processing"]):
        if p==0:
            MatConc = MatMat[proc]
            EucConc = EucMat[proc]
            dict_df[proc]['proc'] = np.ones(np.shape(MatMat[proc])[2])*p
            df_conc = dict_df[proc]
            print('OK')
            
            #df_conc['proc'] = np.ones(np.shape(dict_df[proc])[0])*p
        else:
            MatConc = np.concatenate((MatConc, MatMat[proc]),axis=2)
            EucConc = np.concatenate((EucConc, EucMat[proc]),axis=2)
            dict_df[proc]['proc'] = np.ones(np.shape(MatMat[proc])[2])*p
            df_conc = pd.concat([df_conc, dict_df[proc]], axis=0)
            #df_conc['proc'] = np.ones(len(dict_df[proc]))*p
            
    return MatConc, EucConc, df_conc