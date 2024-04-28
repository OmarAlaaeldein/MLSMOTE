# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
random.seed(77)

def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.iloc[reference,:] - X.iloc[neighbor,:]
        new_X[i] = np.array(X.iloc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target

def SMOLTE_cat_wrapper(x_df, y_df, cat_col, nsamples):
    x_df_up = pd.DataFrame(columns=x_df.columns)
    y_df_up = pd.DataFrame(columns=y_df.columns)

    unique_cat_combs = x_df.groupby(cat_col).size().reset_index().rename(columns={0:'count'})[cat_col]
    num_cols = x_df.columns.drop(cat_col).tolist()
    for index, row in unique_cat_combs.iterrows():
        condition = (x_df[cat_col] == row).all(axis=1)

        subx = x_df[condition][num_cols].reset_index(drop=True)
        suby = y_df[condition].reset_index(drop=True)

        x_df_sub, y_df_sub = get_minority_samples(subx, suby)
        a, b = MLSMOTE(x_df_sub, y_df_sub, nsamples, 5)
        cats = pd.concat([row.to_frame().T]*len(a), ignore_index=True)
        a = pd.merge(cats, a, how='left', left_index=True, right_index=True)
        x_df_up = x_df_up.append(a, ignore_index=True)
        y_df_up = y_df_up.append(b, ignore_index=True)
    #y_df_up = y_df_up.astype(int)
    
    print('Number of new samples created: %d' %(len(y_df_up)))
    
    x_df_up = pd.concat([x_df, x_df_up], ignore_index=True)
    y_df_up = pd.concat([y_df, y_df_up], ignore_index=True)
    
    x_df_up = x_df_up.sample(len(x_df_up), random_state=1881).reset_index(drop=True)
    y_df_up = y_df_up.sample(len(y_df_up), random_state=1881).reset_index(drop=True)
    
    x_df_up[cat_col] = x_df_up[cat_col].astype(int)
    return x_df_up, y_df_up

if __name__=='__main__':
    """
    main function to use the MLSMOTE
    """
    X, y = create_dataset()                     #Creating a Dataframe
    X_res,y_res =MLSMOTE(X, y, 100)     #Applying MLSMOTE to augment the dataframe
